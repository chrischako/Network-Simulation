import numpy as np
import matplotlib.pyplot as plt
import random as rnd

C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm
ar = 100 #rea of 100m x 100m
nd = 20  #number of nodes
np.random.seed(42)

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

# Free space path loss in dB
def fspl_db(distance_m, frequency_hz):
    if distance_m <= 0.1: return 0
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / C)

# Calculate SINR, capacity, and distance for a given pair of nodes
def calculate_metrics(tx_idx, rx_idx, pos):
    #Signal
    dist_sig = np.linalg.norm(np.array(pos[tx_idx]) - np.array(pos[rx_idx])) #distance
    loss_sig = fspl_db(max(dist_sig, 0.1), FREQ) #path loss
    p_sig_watts = 10**((P_TX_DBM - loss_sig - 30) / 10) #P = power

    #Interference
    interference_watts = 0
    for i in range(len(pos)):
        if i != tx_idx and i != rx_idx:
            dist_i = np.linalg.norm(np.array(pos[i]) - np.array(pos[rx_idx]))
            loss_i = fspl_db(max(dist_i, 0.1), FREQ)
            p_int_watts = 10**((P_TX_DBM - loss_i - 30) / 10)
            interference_watts += p_int_watts

    noise_watts = 10**((NOISE_DBM - 30) / 10) #noise converts from wats to mwats
    sinr_linear = p_sig_watts / (interference_watts + noise_watts) #sinr
    capacity = BW * np.log2(1 + sinr_linear) #shannon_capacity
    
    return sinr_linear, capacity, dist_sig

# Greedy pairing: pair closest available nodes. This will have to change, The user will have to choose the pairs, but for now i will just pair the closest nodes. I will also calculate the metrics for each pair and print them out.
def r_data(node_positions): #add a specification for who talks with who and the name of the file which will save the data, also save the data in the correct format (one line)
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

# maybe i need to save the data in only one file, but for now i will save in three different files
def save_data(node_positions, pairs, metrics):
    np.save('node_positions.npy', node_positions)
    np.save('pairs.npy', pairs)
    np.save('metrics.npy', metrics)
    
    with open('node_positions.txt', 'w') as f:
        for pos in node_positions:
            f.write(f"{pos[0]:.2f}, {pos[1]:.2f}\n")
    
    with open('pairs.txt', 'w') as f:
        for pair in pairs:
            f.write(f"{pair[0]}, {pair[1]}\n")
    
    with open('metrics.txt', 'w') as f:
        for metric in metrics:
            f.write(f"SINR: {metric[0]:.2f}, Capacity: {metric[1]:.2f} bps, Distance: {metric[2]:.2f} m\n")
