import numpy as np
import matplotlib.pyplot as plt

nodes = 20 # Must be even
area = 100.0
C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm

#fspl = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
#d = distance, f = frequency, c = speed of light
#sinr = P / (I + N) 
#P = power of the incoming signal of interest, I= Interference, N=Noise
#shannon_cap = B * log2(1 + SINR) 
# B = Bandwidth


positions = np.random.rand(nodes, 2) * area #putting nodes in radom pos

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
    
    return sinr_linear, capacity, dist_sig, loss_sig

#Random pairing: randomly pair nodes
available_nodes = list(range(nodes))
np.random.shuffle(available_nodes)
pairs = []

for i in range(0, len(available_nodes) - 1, 2):
    pairs.append((available_nodes[i], available_nodes[i + 1]))

#Visualization
plt.figure(figsize=(10, 8))
print(f"{'Sender':<8} | {'Receiver':<8} | {'Distance':<10} | {'FSPL (dB)':<10} | {'SINR (dB & Linear)':<20} | {'Shannon Capacity (bps & Mbps)':<35}")
print("-" * 130)

for tx, rx in pairs:
    sinr_lin, cap, dist, fspl = calculate_metrics(tx, rx, positions)
    sinr_db = 10 * np.log10(sinr_lin)
    cap_mbps = cap/1e6
    
    print(f"Node {tx:<4} | Node {rx:<4} | Dist: {dist:<7.2f}m | FSPL: {fspl:<7.2f}dB | SINR: {sinr_db:<8.2f}dB ({sinr_lin:<10.2e}) | Capacity: {cap:<12.2e} bps ({cap_mbps:<8.2f} Mbps)")
    
    # Plotting the pair
    p1, p2 = positions[tx], positions[rx]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.3) # Connection line
    plt.scatter(p1[0], p1[1], color='green', s=80, label='Sender' if tx == pairs[0][0] else "")
    plt.scatter(p2[0], p2[1], color='red', s=80, label='Receiver' if rx == pairs[0][1] else "")
    plt.text(p1[0]+1, p1[1]+1, f"S{tx}", fontweight='bold')
    plt.text(p2[0]+1, p2[1]+1, f"R{rx}", fontweight='bold')


plt.title("Exclusive Network Pairs (Sender -> Receiver)")
plt.xlim(-5, area+5); plt.ylim(-5, area+5)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.show()