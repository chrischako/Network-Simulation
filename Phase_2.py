import numpy as np
import matplotlib.pyplot as plt

nodes = 20 # Must be even
area = 100.0
C = 3e8 
FREQ = 2.4e9 #Frequency 
BW = 20e6 #Bandwidth
P_TX_DBM = 20 #Power of the incoming signal
NOISE_DBM = -90  #Noise power in dBm

#setting seed
np.random.seed(42)
area = np.random.rand(100, 100)

def pos(area, nodes):
    ar = float(input("Enter the size of the area: "))
    nd = int(input("Enter the number of nodes: "))
    while nd % 2 != 0:
        print("Number of nodes must be even. Please enter an even number.")
        nd = int(input("Enter the number of nodes: "))
    return (ar, nd)

print(pos(area, nodes))
#def r_data():