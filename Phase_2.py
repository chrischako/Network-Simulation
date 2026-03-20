import numpy as np
import matplotlib.pyplot as plt
#from metrics import fspl_db, calculate_metrics

nodes = 20 # Must be even
area = 100.0
C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm

# setting seed
np.random.seed(42)
area = np.zeros((100, 100))

def pos(area):
    # setting area
    ar = float(input("Enter the size of the area: "))
    while ar <= 0:
        print("Area must be positive. Please enter a valid area.")
        ar = float(input("Enter the size of the area: "))

    # setting nodes
    nd = int(input("Enter the number of nodes: "))
    while nd % 2 != 0:
        print("Number of nodes must be even. Please enter an even number.")
        nd = int(input("Enter the number of nodes: "))
    
    # setting positios of nodes
    node_positions = []
    for i in range(nd):
        x = np.random.uniform(0, ar)
        y = np.random.uniform(0, ar)
        xi, yi = int(x), int(y)
        if xi >= area.shape[0]:
            xi = area.shape[0] - 1
        if yi >= area.shape[1]:
            yi = area.shape[1] - 1
        area[xi][yi] = 1  # marking the position of the node
        node_positions.append((x, y))
        print(f"Node {i+1} placed at: ({x:.2f}, {y:.2f})")
    return ar, nd, node_positions

def fspl_db(distance_m, frequency_hz):
    if distance_m <= 0.1: return 0
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) + 20 * np.log10(4 * np.pi / C)

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

def r_data(node_positions):
    # Find all unique pairs and their distances
    pairs = []
    for i in range(len(node_positions)):
        for j in range(i + 1, len(node_positions)):
            dist = np.linalg.norm(np.array(node_positions[i]) - np.array(node_positions[j]))
            pairs.append(((i, j), dist))

    # Sort pairs by distance (ascending)
    pairs.sort(key=lambda x: x[1])

    # Number of pairs is half the number of nodes
    num_pairs = len(node_positions) // 2
    closest_pairs = pairs[:num_pairs]

    for (i, j), dist in closest_pairs:
        sinr, capacity, dist = calculate_metrics(i, j, node_positions)
        print(f"Closest Pair ({i}, {j}): SINR = {sinr:.2f}, Capacity = {capacity:.2f} bps, Distance = {dist:.2f} m")


if __name__ == "__main__":
    ar, nd, node_positions = pos(area)
    r_data(node_positions)
