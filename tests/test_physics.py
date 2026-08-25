"""Tests for brain/physics.py — force-directed physics simulation.

TDD: these tests define the contract for PhysicsSimulation.

Test graph: 5 nodes, 3 edges, 2 regions.  Positions are chosen so that the
spec's default parameters produce deterministic, stable behaviour:

  - Convergence test : nodes start near equilibrium (close to link_distance=30
    apart) so velocity damps out within 200 ticks.
  - Attraction test  : edge (0, 1) starts far apart (60 units > rest length 30),
    so the spring force strongly pulls them together on the very first tick —
    distance drops from 60 to ~21 in one step.
  - Region-gravity test : link and repulsion forces disabled so only region
    gravity acts; nodes must move toward their assigned region centres.
"""

import numpy as np

from brain.physics import PhysicsSimulation


# ---------------------------------------------------------------------------
# Helper: build a PhysicsSimulation with the tiny test graph
# ---------------------------------------------------------------------------

def _make_sim(positions, **kwargs):
    """Wrap the shared edges / region data and pass through any param overrides."""
    edges = [(0, 1), (1, 2), (2, 3)]

    # 12 region centres required by the constructor
    region_centers = np.zeros((12, 3), dtype=np.float32)
    region_centers[0] = [20.0, 10.0, 0.0]   # region 0
    region_centers[1] = [-20.0, -10.0, 0.0]  # region 1

    node_regions = np.array([0, 0, 0, 1, 1], dtype=np.int32)

    return PhysicsSimulation(
        positions=np.array(positions, dtype=np.float32),
        edges=edges,
        region_centers=region_centers,
        node_regions=node_regions,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1: Simulation converges
# ---------------------------------------------------------------------------

def test_simulation_converges():
    """Total per-tick displacement must be < 1.0 after 200 ticks.

    Nodes start close to their rest-length equilibrium so velocity damps out
    within the tick budget using the spec's default parameters.
    """
    # Nodes arranged roughly at link_distance=30 apart, near region centres
    positions = [
        [  0.0,   0.0,  0.0],
        [ 30.0,   0.0,  0.0],   # ~rest-length from node 0
        [ 45.0,  26.0,  0.0],   # ~rest-length from node 1
        [ 15.0,  26.0,  0.0],   # ~rest-length from node 2
        [-15.0,  -5.0,  0.0],   # isolated
    ]
    sim = _make_sim(positions)

    for _ in range(200):
        sim.tick()

    prev = sim.get_positions_f32().copy()
    sim.tick()
    after = sim.get_positions_f32()

    total_displacement = float(
        np.sum(np.linalg.norm(after - prev, axis=1))
    )
    assert total_displacement < 1.0, (
        f"Simulation did not converge after 200 ticks: "
        f"total_displacement={total_displacement:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Connected nodes attract
# ---------------------------------------------------------------------------

def test_connected_nodes_attract():
    """Connected nodes that start farther than link_distance must get closer.

    Edge (0, 1): node 0 at origin, node 1 at [60, 0, 0].
    Initial separation 60 >> link_distance=30, so the spring force
    strongly pulls both nodes inward on the very first tick.
    """
    positions = [
        [  0.0,   0.0,  0.0],
        [ 60.0,   0.0,  0.0],   # connected to 0 — clearly past rest length
        [ 30.0,  52.0,  0.0],
        [-60.0,   0.0,  0.0],
        [  0.0, -60.0,  0.0],
    ]
    sim = _make_sim(positions)

    pos_before = sim.get_positions_f32()
    d_before = float(np.linalg.norm(pos_before[0] - pos_before[1]))

    sim.tick()   # one tick is enough — spring force dominates at dist=60

    pos_after = sim.get_positions_f32()
    d_after = float(np.linalg.norm(pos_after[0] - pos_after[1]))

    assert d_after < d_before, (
        f"Connected nodes (0, 1) did not attract after 1 tick: "
        f"d_before={d_before:.2f}, d_after={d_after:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Region gravity pulls node toward its region centre
# ---------------------------------------------------------------------------

def test_region_gravity_pulls_toward_center():
    """Nodes must move closer to their assigned region centres.

    Link and repulsion forces are disabled so only region gravity acts.
    Node 0 (region 0, centre [20, 10, 0]) and node 3 (region 1, centre
    [-20, -10, 0]) both start at positions clearly away from their centres.
    """
    positions = [
        [  0.0,   0.0,  0.0],   # node 0, region 0 centre is [20, 10, 0]
        [ 30.0,   0.0,  0.0],
        [ 45.0,  26.0,  0.0],
        [-60.0,   0.0,  0.0],   # node 3, region 1 centre is [-20, -10, 0]
        [  0.0, -60.0,  0.0],
    ]
    sim = _make_sim(
        positions,
        region_gravity=0.5,
        link_strength=0.0,    # disable springs
        repel_strength=0.0,   # disable repulsion
        center_strength=0.0,  # disable global centre pull
    )

    region_centers = np.zeros((12, 3), dtype=np.float32)
    region_centers[0] = [20.0, 10.0, 0.0]
    region_centers[1] = [-20.0, -10.0, 0.0]

    pos_before = sim.get_positions_f32()
    d0_before = float(np.linalg.norm(pos_before[0] - region_centers[0]))
    d3_before = float(np.linalg.norm(pos_before[3] - region_centers[1]))

    for _ in range(30):
        sim.tick()

    pos_after = sim.get_positions_f32()
    d0_after = float(np.linalg.norm(pos_after[0] - region_centers[0]))
    d3_after = float(np.linalg.norm(pos_after[3] - region_centers[1]))

    assert d0_after < d0_before, (
        f"Node 0 did not move toward its region centre: "
        f"d_before={d0_before:.2f}, d_after={d0_after:.2f}"
    )
    assert d3_after < d3_before, (
        f"Node 3 did not move toward its region centre: "
        f"d_before={d3_before:.2f}, d_after={d3_after:.2f}"
    )
