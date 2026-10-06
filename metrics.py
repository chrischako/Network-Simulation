import os
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm
ar = 1000 #rea of 100m x 100m
nd = 200  #number of nodes
#np.random.seed(42)
'''
#setting positions of nodes randomly in the area
def pos():
    #setting area
    ar = 100 #rea of 100m x 100m

    # setting nodes
    nd = 20  #number of nodes
    
    # setting positions of nodes randomly in the area they need to be stable for the whole simulation, so i will set them in a random way but they will be the same for all the tries
    #positions = np.random.rand(nd, 2) * ar #putting nodes in radom pos
    
    node_positions = []
    for i in range(nd):
        x = np.random.uniform(0, ar)
        y = np.random.uniform(0, ar)
        node_positions.append((x, y))
        print(f"Node {i+1} placed at: ({x:.2f}, {y:.2f})")
    return ar, nd, node_positions
    '''
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


def _make_result_dtype():
    return np.dtype([
        ("trial_num", np.int32),
        ("t_count", np.int32),
        ("r_count", np.int32),
        ("tx_idx", np.int32),
        ("rx_idx", np.int32),
        ("distance", np.float64),
        ("fspl", np.float64),
        ("sinr_db", np.float64),
        ("capacity", np.float64),
    ])


# Saving all results from all tries to text and numpy files.
def save_all_results(trial_num, pairs, roles, node_positions, output_lines=None, append=True, txt_path="results.txt", npy_path="results.npy"):
    t_count = roles.count("T")
    r_count = roles.count("R")
    records = []

    for tx_idx, rx_idx in pairs:
        sinr, capacity, dist, fspl = calculate_metrics(tx_idx, rx_idx, node_positions)
        sinr_db = 10 * np.log10(sinr)
        records.append((trial_num, t_count, r_count, tx_idx, rx_idx, dist, fspl, sinr_db, capacity))

    text_mode = "a" if append else "w"
    with open(txt_path, text_mode) as f:
        if not append or trial_num == 1:
            f.write("Trial | T | R | distance | fspl | sinr_db | capacity\n")
        f.write(f"Trial {trial_num}: T={t_count}, R={r_count}\n")
        for tx_idx, rx_idx in pairs:
            sinr, capacity, dist, fspl = calculate_metrics(tx_idx, rx_idx, node_positions)
            sinr_db = 10 * np.log10(sinr)
            f.write(f"T={tx_idx}, R={rx_idx} | distance={dist:.4f} m | fspl={fspl:.4f} dB | sinr={sinr_db:.4f} dB | capacity={capacity:.6e} bps\n")
        f.write("-" * 100 + "\n")

    new_dtype = _make_result_dtype()
    if append and os.path.exists(npy_path):
        try:
            existing = np.load(npy_path, allow_pickle=False)
            if existing.size == 0 or existing.dtype.names != new_dtype.names:
                new_array = np.array(records, dtype=new_dtype)
            else:
                new_array = np.concatenate([existing, np.array(records, dtype=new_dtype)])
        except Exception:
            new_array = np.array(records, dtype=new_dtype)
    else:
        new_array = np.array(records, dtype=new_dtype)

    np.save(npy_path, new_array)

    if output_lines is not None:
        for line in output_lines:
            print(line)
