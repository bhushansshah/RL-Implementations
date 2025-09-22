import matplotlib.pyplot as plt
import numpy as np

def plot_value(V, max_cars, title="State-Value Function", save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(V.T, origin='lower', cmap='viridis')
    
    cbar = plt.colorbar(im)
    cbar.set_label("Expected Return")
    
    ax.set_title(title)
    ax.set_xlabel("Cars at Location 1")
    ax.set_ylabel("Cars at Location 2")
    ax.set_xticks(np.arange(0, max_cars+1, 2))
    ax.set_yticks(np.arange(0, max_cars+1, 2))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_policy(policy, max_cars, title="Policy", save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(policy.T, origin='lower', 
                   cmap='coolwarm', 
                   vmin=-5, vmax=5)   # policy is 2D: (loc1, loc2)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Cars moved (positive = L1→L2, negative = L2→L1)')
    
    ax.set_title(title)
    ax.set_xlabel("Cars at Location 1")
    ax.set_ylabel("Cars at Location 2")
    ax.set_xticks(np.arange(0, max_cars+1, 2))
    ax.set_yticks(np.arange(0, max_cars+1, 2))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
