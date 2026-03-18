import numpy as np
import matplotlib.pyplot as plt

C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm

#setting seed
np.random.seed(42)
area = np.random.rand(100, 100)

def pos(area, nodes):
    #setting area
    ar = float(input("Enter the size of the area: "))
    while ar <= 0:
        print("Area must be positive. Please enter a valid area.")
        ar = float(input("Enter the size of the area: "))

    #setting nodes
    nd = int(input("Enter the number of nodes: "))
    while nd % 2 != 0:
        print("Number of nodes must be even. Please enter an even number.")
        nd = int(input("Enter the number of nodes: "))

    #position of nodes
    for i in range(nd):
        x = np.random.uniform(0, ar)
        y = np.random.uniform(0, ar)
        area[int(x)][int(y)] = 1 #marking the position of the node
        print(f"Node {i+1} placed at: ({x:.2f}, {y:.2f})") #position is stable if the area and nodes are the same
    return (ar, nd)

#def fspl_db(distance_m, frequency_hz):
#def metrics(tx_idx, rx_idx, pos):
#def r_data():
