import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from math import log10
from Phase_2 import pos, fspl_db, calculate_metrics


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


# Greedy pairing: pair closest available nodes. This will have to change, The user will have to choose the pairs, but for now i will just pair the closest nodes. I will also calculate the metrics for each pair and print them out.
def r_data(node_positions):
    # Greedy pairing: pair closest available nodes
    available_nodes = list(range(len(node_positions)))
    pairs = []
    
    while len(available_nodes) >= 2:
        best_dist = np.inf
        best_pair = None
        for i in range(len(available_nodes)):
            for j in range(i + 1, len(available_nodes)):
                dist = np.linalg.norm(np.array(node_positions[available_nodes[i]]) - np.array(node_positions[available_nodes[j]]))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (available_nodes[i], available_nodes[j])
        if best_pair:
            pairs.append(best_pair)
            available_nodes.remove(best_pair[0])
            available_nodes.remove(best_pair[1])
    
    # Calculate metrics for each pair
    for tx_idx, rx_idx in pairs:
        sinr, capacity, dist = calculate_metrics(tx_idx, rx_idx, node_positions)
        print(f"Pair ({tx_idx}, {rx_idx}): SINR = {sinr:.2f}, Capacity = {capacity:.2f} bps, Distance = {dist:.2f} m")
    
    return pairs