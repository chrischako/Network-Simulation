import numpy as np
import matplotlib.pyplot as plt
from metrics import save_all_results
from pos import pos, r_data

if __name__ == "__main__":
    ar, nd, node_positions, roles = pos()
    pairs = []

    for i in range(1, 51):
        pairs, output_lines = r_data(node_positions, roles)
        save_all_results(i, pairs, roles, node_positions, output_lines=output_lines, append=(i > 1))

    # Plot the nodes and links once using the same fixed positions
    plt.figure(figsize=(8, 8))

    tx_indices = [idx for idx, role in enumerate(roles) if role == 'T']
    rx_indices = [idx for idx, role in enumerate(roles) if role == 'R']

    plt.scatter(
        [node_positions[idx][0] for idx in tx_indices],
        [node_positions[idx][1] for idx in tx_indices],
        c='green', s=80, label='Transmitters'
    )
    plt.scatter(
        [node_positions[idx][0] for idx in rx_indices],
        [node_positions[idx][1] for idx in rx_indices],
        c='red', s=80, label='Receivers'
    )

    for tx_idx, rx_idx in pairs:
        plt.plot(
            [node_positions[tx_idx][0], node_positions[rx_idx][0]],
            [node_positions[tx_idx][1], node_positions[rx_idx][1]],
            'k--', alpha=0.4
        )

    for idx, role in enumerate(roles):
        x, y = node_positions[idx]
        plt.text(x + 0.8, y + 0.8, f"{idx} ({role})")

    plt.xlim(0, ar)
    plt.ylim(0, ar)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Node Positions and Traffic Roles')
    plt.legend()
    plt.grid(True)
    plt.show()