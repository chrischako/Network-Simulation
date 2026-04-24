import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from Phase_2 import ar, nd, BW, P_TX_DBM, NOISE_DBM, pos
from Phase_2 import pos, r_data, save_data, calculate_metrics
from pos import pos, r_data

for i in range (50):        
    if __name__ == "__main__":
        ar, nd, node_positions = pos()
        pairs = r_data(node_positions)
        save_data(node_positions, pairs, [calculate_metrics(tx_idx, rx_idx, node_positions) for tx_idx, rx_idx in pairs])
        
        # Plot the nodes and pairs
        plt.figure(figsize=(8, 8))
        x_coords = [pos[0] for pos in node_positions]
        y_coords = [pos[1] for pos in node_positions]
        plt.scatter(x_coords, y_coords, c='blue', label='Nodes')
        
        for tx_idx, rx_idx in pairs:
            plt.plot([node_positions[tx_idx][0], node_positions[rx_idx][0]], 
                    [node_positions[tx_idx][1], node_positions[rx_idx][1]], 'r-')
        
        plt.xlim(0, ar)
        plt.ylim(0, ar)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Node Positions and Pairs')
        plt.legend()
        plt.grid(True)
        plt.show()