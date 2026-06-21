import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from math import log10
from metrics import pos, fspl_db, calculate_metrics


#setting positions of nodes randomly in the area
def pos():
    #setting area
    ar = 100 #rea of 100m x 100m

    # setting nodes
    nd = 20  #number of nodes
    
    # setting positions of nodes randomly in the area
    positions = np.random.rand(nd, 2) * ar #putting nodes in radom pos
    node_positions = []
    for i in range(nd):
        x = np.random.uniform(0, ar)
        y = np.random.uniform(0, ar)
        node_positions.append((x, y))
        print(f"Node {i+1} placed at: ({x:.2f}, {y:.2f})")
    return ar, nd, node_positions


# Random pairing: randomly pair available nodes
def r_data(node_positions):
    # Random pairing: shuffle nodes and pair sequentially
    available_nodes = list(range(len(node_positions)))
    rnd.shuffle(available_nodes)
    pairs = []
    
    for i in range(0, len(available_nodes) - 1, 2):
        pairs.append((available_nodes[i], available_nodes[i + 1]))
    
    # Calculate metrics for each pair and collect output
    output_lines = []
    for tx_idx, rx_idx in pairs:
        sinr, capacity, dist, fspl = calculate_metrics(tx_idx, rx_idx, node_positions)
        sinr_db = 10*np.log10(sinr)
        cap_mbps = capacity/1e6
        line = f"Node {tx_idx:<4} | Node {rx_idx:<4} | Dist: {dist:<7.2f}m | FSPL: {fspl:<7.2f}dB | SINR: {sinr_db:<8.2f}dB ({sinr:<10.2e}) | Capacity: {capacity:<12.2e} bps ({cap_mbps:<8.2f} Mbps)"
        output_lines.append(line)
        print(line)
    
    return pairs, output_lines