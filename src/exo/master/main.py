import hashlib
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

import anyio
from loguru import logger

from exo.master.placement import (
    add_instance_to_placements,
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.master.placement_utils import find_ip_prioritised
from exo.routing.event_router import (
    EventRouterBrokenResourceError,
    EventRouterClosedResourceError,
)
from exo.shared.apply import apply
from exo.shared.constants import EXO_EVENT_LOG_DIR, EXO_TRACING_ENABLED
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    ForwarderCommand,
    ForwarderDownloadCommand,
    ImageEdits,
    ImageGeneration,
    PlaceInstance,
    RequestEventLog,
    SendInputChunk,
    SetInstanceLink,
    TaskCancelled,
    TaskFinished,
    TestCommand,
    TextGeneration,
)
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    CustomModelCardAdded,
    CustomModelCardDeleted,
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    InputChunkReceived,
    InstanceDeleted,
    InstanceLinkCreated,
    InstanceLinkDeleted,
    LocalForwarderEvent,
    NodeGatheredInfo,
    NodeTimedOut,
    TaskCreated,
    TaskDeleted,
    TaskStatusUpdated,
    TraceEventData,
    TracesCollected,
    TracesMerged,
)
from exo.shared.types.instance_link import InstanceLink
from exo.shared.types.state import State
from exo.shared.types.tasks import (
    ImageEdits as ImageEditsTask,
)
from exo.shared.types.tasks import (
    ImageGeneration as ImageGenerationTask,
)
from exo.shared.types.tasks import (
    TaskId,
    TaskStatus,
)
from exo.shared.types.tasks import (
    TextGeneration as TextGenerationTask,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import InstanceId
from exo.utils.channels import Receiver, Sender
from exo.utils.disk_event_log import DiskEventLog
from exo.utils.event_buffer import MultiSourceBuffer
from exo.utils.task_group import TaskGroup


def _prefill_endpoint_for(state: State, decode_instance_id: InstanceId) -> str | None:
    decode = state.instances.get(decode_instance_id)
    if decode is None:
        return None
    decode_node = next(iter(decode.shard_assignments.node_to_runner.keys()), None)
    if decode_node is None:
        return None

    sources: set[InstanceId] = set()
    for link in state.instance_links.values():
        if decode_instance_id in link.decode_instances:
            sources.update(link.prefill_instances)
    sources.discard(decode_instance_id)

    in_flight = {TaskStatus.Pending, TaskStatus.Running}
    task_counts: dict[InstanceId, int] = {
        src_id: sum(
            1
            for task in state.tasks.values()
            if task.instance_id == src_id and task.task_status in in_flight
        )
        for src_id in sources
    }
    for src_id in sorted(sources, key=lambda sid: task_counts[sid]):
        instance = state.instances.get(src_id)
        if instance is None:
            continue
        for node_id, runner_id in instance.shard_assignments.node_to_runner.items():
            port = state.prefill_server_ports.get(runner_id)
            if port is None:
                continue
            ip = find_ip_prioritised(
                decode_node, node_id, state.topology, state.node_network, ring=True
            )
            if ip is None:
                continue
            return f"{ip}:{port}"
    return None


def _balancer_instance_label(state: State, instance_id: InstanceId) -> str:
    """'shortid(node1,node2)' label for load-balancer diagnostics."""
    inst = state.instances.get(instance_id)
    if inst is None:
        return str(instance_id)[:8]
    names: list[str] = []
    for node_id in inst.shard_assignments.node_to_runner:
        ident = state.node_identities.get(node_id)
        names.append(ident.friendly_name if ident is not None else str(node_id)[:8])
    return f"{str(instance_id)[:8]}({','.join(names) if names else '?'})"


def _log_balancer_choice(
    state: State,
    model: str,
    instance_task_counts: dict[InstanceId, int],
    chosen_id: InstanceId,
) -> None:
    """Always-on diagnostics: candidate replicas, their in-flight task counts,
    and which one the least-loaded balancer picked. Never raises."""
    try:
        ranked = sorted(instance_task_counts.items(), key=lambda kv: kv[1])
        parts = [
            f"{_balancer_instance_label(state, iid)} inflight={count}"
            + (" <=CHOSEN" if iid == chosen_id else "")
            for iid, count in ranked
        ]
        logger.info(f"[balancer] model={model} | " + " | ".join(parts))
    except Exception as exc:  # diagnostics must never break the master loop
        logger.warning(f"[balancer] diagnostics failed: {exc!r}")


# Round-robin cursor per model, used only to break ties between replicas that
# are equally (least) loaded. Master is a single process and the command
# processor handles commands one at a time, so no locking is needed.
_BALANCER_RR: dict[str, int] = {}

# Conversation -> instance affinity so consecutive turns of one conversation
# land on the replica that holds its KV prefix cache. A re-prefill of a long
# context costs minutes on Apple Silicon; queueing briefly behind a couple of
# in-flight tasks is cheaper, hence AFFINITY_MAX_OVERLOAD. Same single-process
# assumption as _BALANCER_RR.
_BALANCER_AFFINITY: "OrderedDict[str, InstanceId]" = OrderedDict()
_AFFINITY_MAX_ENTRIES = 256
_AFFINITY_MAX_OVERLOAD = 2


def _conversation_affinity_key(task_params: TextGenerationTaskParams) -> str | None:
    """Stable fingerprint of a conversation across its turns: instructions plus
    the first two messages (system prompt + first user message). Later turns
    append messages, so this prefix identifies the conversation; different
    sessions differ in their first user message. Never raises."""
    try:
        parts: list[str] = []
        if task_params.instructions:
            parts.append(str(task_params.instructions))
        msgs: list[Any] = list(task_params.chat_template_messages or [])
        if msgs:
            for m in msgs[:2]:
                parts.append(f"{m.get('role', '')}\x1f{m.get('content', '')}")
        else:
            for im in list(task_params.input or [])[:2]:
                parts.append(f"{im.role}\x1f{im.content}")
        if not parts:
            return None
        blob = "\x00".join(parts)[:4000]
        return hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()[:16]
    except Exception:
        return None


def _remember_affinity(key: str, instance_id: InstanceId) -> None:
    _BALANCER_AFFINITY[key] = instance_id
    _BALANCER_AFFINITY.move_to_end(key)
    while len(_BALANCER_AFFINITY) > _AFFINITY_MAX_ENTRIES:
        _BALANCER_AFFINITY.popitem(last=False)


def _select_balanced_instance(
    instance_task_counts: dict[InstanceId, int],
    model: str,
    affinity_key: str | None = None,
) -> InstanceId:
    """Pick the least in-flight instance; round-robin among those tied at the
    minimum so sequential traffic alternates across replicas instead of always
    pinning to the same one. Under real concurrency this is still least-loaded.

    When affinity_key is set and its remembered replica is within
    _AFFINITY_MAX_OVERLOAD of the least-loaded count, stick to it so the
    conversation reuses that replica's KV prefix cache instead of re-prefilling
    its whole history elsewhere."""
    min_count = min(instance_task_counts.values())

    if affinity_key is not None:
        remembered = _BALANCER_AFFINITY.get(affinity_key)
        if (
            remembered is not None
            and remembered in instance_task_counts
            and instance_task_counts[remembered] <= min_count + _AFFINITY_MAX_OVERLOAD
        ):
            _BALANCER_AFFINITY.move_to_end(affinity_key)
            logger.info(
                f"[balancer] affinity {affinity_key} -> {str(remembered)[:8]} "
                f"(inflight={instance_task_counts[remembered]}, min={min_count})"
            )
            return remembered

    tied = sorted(iid for iid, c in instance_task_counts.items() if c == min_count)
    if len(tied) == 1:
        chosen = tied[0]
    else:
        idx = _BALANCER_RR.get(model, 0) % len(tied)
        _BALANCER_RR[model] = idx + 1
        chosen = tied[idx]

    if affinity_key is not None:
        _remember_affinity(affinity_key, chosen)
    return chosen


class Master:
    def __init__(
        self,
        node_id: NodeId,
        session_id: SessionId,
        *,
        command_receiver: Receiver[ForwarderCommand],
        event_sender: Sender[Event],
        local_event_receiver: Receiver[LocalForwarderEvent],
        global_event_sender: Sender[GlobalForwarderEvent],
        download_command_sender: Sender[ForwarderDownloadCommand],
    ):
        self.node_id = node_id
        self.session_id = session_id
        self.state = State()
        self._tg: TaskGroup = TaskGroup()
        self.command_task_mapping: dict[CommandId, TaskId] = {}
        self.command_receiver = command_receiver
        self.local_event_receiver = local_event_receiver
        self.global_event_sender = global_event_sender
        self.download_command_sender = download_command_sender
        self.event_sender = event_sender
        self._system_id = SystemId()
        self._multi_buffer = MultiSourceBuffer[SystemId, Event]()
        self._event_log = DiskEventLog(EXO_EVENT_LOG_DIR / "master")
        self._pending_traces: dict[TaskId, dict[int, list[TraceEventData]]] = {}
        self._expected_ranks: dict[TaskId, set[int]] = {}

    async def run(self):
        logger.info("Starting Master")

        try:
            async with self._tg as tg:
                tg.start_soon(self._event_processor)
                tg.start_soon(self._command_processor)
                tg.start_soon(self._plan)
        except* (EventRouterBrokenResourceError, EventRouterClosedResourceError):
            # Event router has been closed (try-star syntax handles error groups)
            pass
        finally:
            self._event_log.close()
            self.global_event_sender.close()
            self.local_event_receiver.close()
            self.command_receiver.close()

    async def shutdown(self):
        logger.info("Stopping Master")
        self._tg.cancel_tasks()

    async def _command_processor(self) -> None:
        with self.command_receiver as commands:
            async for forwarder_command in commands:
                try:
                    logger.info(f"Executing command: {forwarder_command.command}")

                    generated_events: list[Event] = []
                    command = forwarder_command.command
                    instance_task_counts: dict[InstanceId, int] = {}
                    match command:
                        case TestCommand():
                            pass
                        case TextGeneration():
                            # set-difference => prefill-only nodes
                            prefill_only: set[InstanceId] = set()
                            for link in self.state.instance_links.values():
                                prefill_only.update(link.prefill_instances)
                            for link in self.state.instance_links.values():
                                prefill_only.difference_update(link.decode_instances)

                            for instance in self.state.instances.values():
                                # NON-prefill-only instances matching the model ID
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                    and instance.instance_id not in prefill_only
                                ):
                                    # count in-flight tasks of that instance
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            # there are no NON-prefill-only instances matching this model ID
                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            decode_instance_id = _select_balanced_instance(
                                instance_task_counts,
                                command.task_params.model,
                                affinity_key=_conversation_affinity_key(
                                    command.task_params
                                ),
                            )
                            _log_balancer_choice(
                                self.state,
                                command.task_params.model,
                                instance_task_counts,
                                decode_instance_id,
                            )
                            task_id = TaskId()
                            params = command.task_params.model_copy(
                                update={
                                    "prefill_endpoint": _prefill_endpoint_for(
                                        self.state, decode_instance_id
                                    ),
                                }
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=TextGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=decode_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=params,
                                    ),
                                )
                            )
                            self.command_task_mapping[command.command_id] = task_id
                        case ImageGeneration():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            task_id = TaskId()
                            selected_instance_id = _select_balanced_instance(
                                instance_task_counts, command.task_params.model
                            )
                            _log_balancer_choice(
                                self.state,
                                command.task_params.model,
                                instance_task_counts,
                                selected_instance_id,
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageGenerationTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case ImageEdits():
                            for instance in self.state.instances.values():
                                if (
                                    instance.shard_assignments.model_id
                                    == command.task_params.model
                                ):
                                    in_flight = {TaskStatus.Pending, TaskStatus.Running}
                                    task_count = sum(
                                        1
                                        for task in self.state.tasks.values()
                                        if task.instance_id == instance.instance_id
                                        and task.task_status in in_flight
                                    )
                                    instance_task_counts[instance.instance_id] = (
                                        task_count
                                    )

                            if not instance_task_counts:
                                raise ValueError(
                                    f"No instance found for model {command.task_params.model}"
                                )

                            task_id = TaskId()
                            selected_instance_id = _select_balanced_instance(
                                instance_task_counts, command.task_params.model
                            )
                            _log_balancer_choice(
                                self.state,
                                command.task_params.model,
                                instance_task_counts,
                                selected_instance_id,
                            )
                            generated_events.append(
                                TaskCreated(
                                    task_id=task_id,
                                    task=ImageEditsTask(
                                        task_id=task_id,
                                        command_id=command.command_id,
                                        instance_id=selected_instance_id,
                                        task_status=TaskStatus.Pending,
                                        task_params=command.task_params,
                                    ),
                                )
                            )

                            self.command_task_mapping[command.command_id] = task_id

                            if EXO_TRACING_ENABLED:
                                selected_instance = self.state.instances.get(
                                    selected_instance_id
                                )
                                if selected_instance:
                                    ranks = set(
                                        shard.device_rank
                                        for shard in selected_instance.shard_assignments.runner_to_shard.values()
                                    )
                                    self._expected_ranks[task_id] = ranks
                        case DeleteInstance():
                            placement = delete_instance(command, self.state.instances)
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            for cmd in cancel_unnecessary_downloads(
                                placement, self.state.downloads
                            ):
                                await self.download_command_sender.send(
                                    ForwarderDownloadCommand(
                                        origin=self._system_id, command=cmd
                                    )
                                )
                            generated_events.extend(transition_events)
                        case PlaceInstance():
                            placement = place_instance(
                                command,
                                self.state.topology,
                                self.state.instances,
                                self.state.node_memory,
                                self.state.node_network,
                                self.state.node_backends,
                                download_status=self.state.downloads,
                                node_rdma_ctl=self.state.node_rdma_ctl,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case CreateInstance():
                            placement = add_instance_to_placements(
                                command,
                                self.state.topology,
                                self.state.instances,
                            )
                            transition_events = get_transition_events(
                                self.state.instances, placement, self.state.tasks
                            )
                            generated_events.extend(transition_events)
                        case SendInputChunk(chunk=chunk):
                            generated_events.append(
                                InputChunkReceived(
                                    command_id=chunk.command_id,
                                    chunk=chunk,
                                )
                            )
                        case TaskCancelled():
                            if (
                                task_id := self.command_task_mapping.get(
                                    command.cancelled_command_id
                                )
                            ) is not None:
                                generated_events.append(
                                    TaskStatusUpdated(
                                        task_status=TaskStatus.Cancelled,
                                        task_id=task_id,
                                    )
                                )
                            else:
                                logger.warning(
                                    f"Nonexistent command {command.cancelled_command_id} cancelled"
                                )
                        case TaskFinished():
                            if (
                                task_id := self.command_task_mapping.pop(
                                    command.finished_command_id, None
                                )
                            ) is not None:
                                generated_events.append(TaskDeleted(task_id=task_id))
                            else:
                                logger.warning(
                                    f"Finished command {command.finished_command_id} finished"
                                )

                        case AddCustomModelCard():
                            generated_events.append(
                                CustomModelCardAdded(model_card=command.model_card)
                            )
                        case DeleteCustomModelCard():
                            generated_events.append(
                                CustomModelCardDeleted(model_id=command.model_id)
                            )
                        case SetInstanceLink():
                            link = InstanceLink(
                                link_id=command.link_id,
                                prefill_instances=list(
                                    dict.fromkeys(command.prefill_instances)
                                ),
                                decode_instances=list(
                                    dict.fromkeys(command.decode_instances)
                                ),
                            )
                            generated_events.append(InstanceLinkCreated(link=link))
                        case DeleteInstanceLink():
                            generated_events.append(
                                InstanceLinkDeleted(link_id=command.link_id)
                            )
                        case RequestEventLog():
                            # We should just be able to send everything, since other buffers will ignore old messages
                            # rate limit to 1000 at a time
                            end = min(command.since_idx + 1000, len(self._event_log))
                            for i, event in enumerate(
                                self._event_log.read_range(command.since_idx, end),
                                start=command.since_idx,
                            ):
                                await self._send_indexed_event(
                                    IndexedEvent(idx=i, event=event)
                                )
                    for event in generated_events:
                        await self.event_sender.send(event)
                except ValueError as e:
                    logger.opt(exception=e).warning("Error in command processor")

    # These plan loops are the cracks showing in our event sourcing architecture - more things could be commands
    async def _plan(self) -> None:
        while True:
            # kill broken instances
            connected_node_ids = set(self.state.topology.list_nodes())
            for instance_id, instance in self.state.instances.items():
                for node_id in instance.shard_assignments.node_to_runner:
                    if node_id not in connected_node_ids:
                        await self.event_sender.send(
                            InstanceDeleted(instance_id=instance_id)
                        )
                        break

            # time out dead nodes
            for node_id, time in self.state.last_seen.items():
                now = datetime.now(tz=timezone.utc)
                if now - time > timedelta(seconds=30):
                    logger.info(f"Manually removing node {node_id} due to inactivity")
                    await self.event_sender.send(NodeTimedOut(node_id=node_id))

            # reap orphaned tasks: their instance is gone, so no terminal event
            # will ever arrive, and Pending/Running they'd count as in-flight
            # forever and skew the replica balancer toward the other instances
            live_instances = set(self.state.instances.keys())
            for task_id, task in self.state.tasks.items():
                if (
                    task.task_status in (TaskStatus.Pending, TaskStatus.Running)
                    and task.instance_id not in live_instances
                ):
                    logger.info(
                        f"Reaping orphaned task {task_id} (instance "
                        f"{str(task.instance_id)[:8]} no longer exists)"
                    )
                    await self.event_sender.send(TaskDeleted(task_id=task_id))

            await anyio.sleep(10)

    async def _event_processor(self) -> None:
        with self.local_event_receiver as local_events:
            async for local_event in local_events:
                # Discard all events not from our session
                if local_event.session != self.session_id:
                    continue
                self._multi_buffer.ingest(
                    local_event.origin_idx,
                    local_event.event,
                    local_event.origin,
                )
                for event in self._multi_buffer.drain():
                    if isinstance(event, TracesCollected):
                        await self._handle_traces_collected(event)
                        continue

                    logger.debug(f"Master indexing event: {str(event)[:100]}")

                    event = event.model_copy(
                        update={"_master_time_stamp": datetime.now(tz=timezone.utc)}
                    )
                    if isinstance(event, NodeGatheredInfo):
                        event = event.model_copy(
                            update={"when": str(datetime.now(tz=timezone.utc))}
                        )

                    indexed = IndexedEvent(event=event, idx=len(self._event_log))
                    self.state = apply(self.state, indexed)

                    self._event_log.append(event)
                    await self._send_indexed_event(indexed)

    # This function is re-entrant, take care!
    async def _send_indexed_event(self, event: IndexedEvent):
        # Convenience method since this line is ugly
        await self.global_event_sender.send(
            GlobalForwarderEvent(
                origin=self.node_id,
                origin_idx=event.idx,
                session=self.session_id,
                event=event.event,
            )
        )

    async def _handle_traces_collected(self, event: TracesCollected) -> None:
        task_id = event.task_id
        if task_id not in self._pending_traces:
            self._pending_traces[task_id] = {}
        self._pending_traces[task_id][event.rank] = event.traces

        if (
            task_id in self._expected_ranks
            and set(self._pending_traces[task_id].keys())
            >= self._expected_ranks[task_id]
        ):
            await self._merge_and_save_traces(task_id)

    async def _merge_and_save_traces(self, task_id: TaskId) -> None:
        all_trace_data: list[TraceEventData] = []
        for trace_data in self._pending_traces[task_id].values():
            all_trace_data.extend(trace_data)

        await self.event_sender.send(
            TracesMerged(task_id=task_id, traces=all_trace_data)
        )

        del self._pending_traces[task_id]
        if task_id in self._expected_ranks:
            del self._expected_ranks[task_id]
