from exo.shared.apply import apply_node_timed_out
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut
from exo.shared.types.profiling import NodeIdentity, SystemPerformanceProfile
from exo.shared.types.state import State


def test_apply_node_timed_out_prunes_identity_and_runtime_state():
    stale_node = NodeId("stale-node")
    live_node = NodeId("live-node")

    topology = Topology()
    topology.add_node(stale_node)
    topology.add_node(live_node)

    state = State(
        topology=topology,
        node_identities={
            stale_node: NodeIdentity(friendly_name="stale"),
            live_node: NodeIdentity(friendly_name="live"),
        },
        node_system={
            stale_node: SystemPerformanceProfile(gpu_usage=1.0),
            live_node: SystemPerformanceProfile(gpu_usage=0.5),
        },
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=stale_node), state)

    assert stale_node not in new_state.node_identities
    assert stale_node not in new_state.node_system
    assert not new_state.topology.contains_node(stale_node)
