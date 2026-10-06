import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from math import log10
from metrics import fspl_db, calculate_metrics


# setting positions of nodes randomly in the area
def pos(seed=42):
    # setting area
    ar = 1000  # area of 1000m x 1000m

    # setting nodes
    nd = 200  # number of nodes

    # positions stay deterministic for a given seed
    position_rng = np.random.default_rng(seed)
    role_rng = np.random.default_rng(seed)

    node_positions = []
    roles = []
    for i in range(nd):
        x, y = position_rng.uniform(0, ar, size=2)
        role = "T" if role_rng.random() < 0.5 else "R"
        node_positions.append((float(x), float(y)))
        roles.append(role)
        print(f"Node {i+1} placed at: ({x:.2f}, {y:.2f}) as {role}")

    return ar, nd, node_positions, roles


# Building 1:1 transmiter/receiver links at random positions
def r_data(node_positions, roles=None, seed=None):
    if roles is None:
        roles = ["R"] * len(node_positions)

    rng = np.random.default_rng(seed)
    tx_indices = [idx for idx, role in enumerate(roles) if role == "T"]
    receiver_indices = [idx for idx, role in enumerate(roles) if role == "R"]
    pairs = []

    if receiver_indices:
        available_receivers = receiver_indices.copy()
        rng.shuffle(available_receivers)

        for tx_idx in tx_indices:
            if not available_receivers:
                break
            rx_idx = available_receivers.pop()
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