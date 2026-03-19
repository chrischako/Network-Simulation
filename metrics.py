import numpy as np
import matplotlib.pyplot as plt

nodes = 20 # Must be even
area = 100.0
C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm

def fspl_db(distance_m, frequency_hz):
    if distance_m <= 0.1: return 0
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / C)

def calculate_metrics(tx_idx, rx_idx, pos):
    #Signal
    dist_sig = np.linalg.norm(pos[tx_idx] - pos[rx_idx]) #distance
    loss_sig = fspl_db(max(dist_sig, 0.1), FREQ) #path loss
    p_sig_watts = 10**((P_TX_DBM - loss_sig - 30) / 10) #P = power
    
    #Interference
    interference_watts = 0
    for i in range(len(pos)):
        if i != tx_idx and i != rx_idx:
            dist_i = np.linalg.norm(pos[i] - pos[rx_idx])
            loss_i = fspl_db(max(dist_i, 0.1), FREQ)
            p_int_watts = 10**((P_TX_DBM - loss_i - 30) / 10)
            interference_watts += p_int_watts

    noise_watts = 10**((NOISE_DBM - 30) / 10) #noise converts from wats to mwats
    
    sinr_linear = p_sig_watts / (interference_watts + noise_watts) #sinr
    capacity = BW * np.log2(1 + sinr_linear) #shannon_capacity
    
    return sinr_linear, capacity, dist_sig

#maybe i should find the pairs and then calculate the metrics
#we use a greedy approach to pair closest nodes
available_nodes = list(range(nodes))
pairs = []
"""
while len(available_nodes) >= 2:
    best_dist = np.inf
    best_pair = (None, None)
    
    #find the closest pair
    for i in range(len(available_nodes)):
        for j in range(i + 1, len(available_nodes)):
            n1, n2 = available_nodes[i], available_nodes[j]
            d = np.linalg.norm(positions[n1] - positions[n2])
            if d < best_dist:
                best_dist = d
                best_pair = (n1, n2)
    
    pairs.append(best_pair)
    available_nodes.remove(best_pair[0])
    available_nodes.remove(best_pair[1])
"""