import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from math import log10
from metrics import fspl_db, calculate_metrics


_cached_result = None


# setting positions of nodes randomly in the area
def pos(seed=42):
    global _cached_result

    if _cached_result is not None:
        return _cached_result

    # setting area
    ar = 100  # area of 100m x 100m

    # setting nodes
    nd = 20  # number of nodes

    # deterministic positions from a fixed seed
    rng = np.random.default_rng(seed)
    node_positions = []
    roles = []
    for i in range(nd):
        x, y = rng.uniform(0, ar, size=2)
        role = "T" if rng.random() < 0.5 else "R"
        node_positions.append((float(x), float(y)))
        roles.append(role)
        print(f"Node {i+1} placed at: ({x:.2f}, {y:.2f}) as {role}")

    _cached_result = (ar, nd, node_positions, roles)
    return _cached_result


# Build one-to-many transmit/receive links
def r_data(node_positions, roles=None):
    if roles is None:
        roles = ["R"] * len(node_positions)

    pairs = []
    for tx_idx, tx_role in enumerate(roles):
        if tx_role != "T":
            continue
        for rx_idx, rx_role in enumerate(roles):
            if tx_idx == rx_idx or rx_role != "R":
                continue
            pairs.append((tx_idx, rx_idx))

    output_lines = []
    for tx_idx, rx_idx in pairs:
        sinr, capacity, dist, fspl = calculate_metrics(tx_idx, rx_idx, node_positions)
        sinr_db = 10 * np.log10(sinr)
        cap_mbps = capacity / 1e6
        line = (
            f"Node {tx_idx:<4} [{roles[tx_idx]}] | Node {rx_idx:<4} [{roles[rx_idx]}] | "
            f"Dist: {dist:<7.2f}m | FSPL: {fspl:<7.2f}dB | SINR: {sinr_db:<8.2f}dB ({sinr:<10.2e}) | "
            f"Capacity: {capacity:<12.2e} bps ({cap_mbps:<8.2f} Mbps)"
        )
        output_lines.append(line)
        print(line)

    return pairs, output_lines