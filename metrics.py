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
    loss_sig = fspl_db(max(dist_sig, 0.1), FREQ) #path loss (FSPL in dB)
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
    
    return sinr_linear, capacity, dist_sig, loss_sig

# Random pairing: randomly pair available nodes
def r_data(node_positions): #add a specification for who talks with who and the name of the file which will save the data, also save the data in the correct format (one line)
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
        line = f"Nd {tx_idx:<4} | Nd {rx_idx:<4} | Dist: {dist:<7.2f}m | FSPL: {fspl:<7.2f}dB | SINR: {sinr_db:<8.2f}dB ({sinr:<10.2e}) | Capacity: {capacity:<12.2e} bps ({cap_mbps:<8.2f} Mbps)"
        output_lines.append(line)
        print(line)
    
    return pairs, output_lines

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

# Save all results from all tries to a single file
def save_all_results(trial_num, output_lines, append=True):
    mode = 'a' if append else 'w'
    with open('all_results.txt', mode) as f:
        if not append or trial_num == 1:
            # Write header only on first trial or if not appending
            f.write(f"{'Sender':<8} | {'Receiver':<8} | {'Distance':<10} | {'FSPL (dB)':<10} | {'SINR (dB & Linear)':<20} | {'Shannon Capacity (bps & Mbps)':<35}\n")
            f.write("-" * 130 + "\n")
        f.write(f"\n=== Trial {trial_num} ===\n")
        for line in output_lines:
            f.write(line + "\n")
        f.write("-" * 130 + "\n")
