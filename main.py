import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from metrics import ar, nd, BW, P_TX_DBM, NOISE_DBM, pos
from metrics import pos, r_data, save_data, calculate_metrics, save_all_results
from pos import pos, r_data

for i in range(1, 51):        
    if __name__ == "__main__":
        ar, nd, node_positions = pos()
        pairs, output_lines = r_data(node_positions)
        metrics_data = [(calculate_metrics(tx_idx, rx_idx, node_positions)[:3]) for tx_idx, rx_idx in pairs]
        save_data(node_positions, pairs, metrics_data)
        save_all_results(i, output_lines, append=(i > 1))
        
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