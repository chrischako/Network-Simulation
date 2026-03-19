import numpy as np
import matplotlib.pyplot as plt
from metrics import fspl_db, calculate_metrics

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

print(pos(area, 20))

#def r_data():
