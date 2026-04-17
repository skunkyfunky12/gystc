"""Assign initial 3D positions to nodes based on brain region centers."""

import numpy as np

from data.regions import REGIONS, COMMUNITY_TO_REGION


def assign_initial_positions(nodes):
    """Assign 3D positions based on brain region centers with random scatter.

    Each node is placed near its community's brain region center,
    with Gaussian scatter (std=15) for visual separation.

    Args:
        nodes: List of dicts, each with an optional "community" key (int).

    Returns:
        np.ndarray of shape (len(nodes), 3) with float32 positions.
    """
    rng = np.random.default_rng(42)
    positions = np.zeros((len(nodes), 3), dtype=np.float32)
    for i, node in enumerate(nodes):
        community = node.get("community", 0)
        region_idx = COMMUNITY_TO_REGION.get(community, 9)
        center = np.array(REGIONS[region_idx]["position"], dtype=np.float32)
        scatter = rng.normal(0, 15.0, size=3).astype(np.float32)
        positions[i] = center + scatter
    return positions
