import matplotlib.pyplot as plt
import numpy as np

def plot_value(V, title="State-Value Function", save_path=None):
    fig, ax = plt.subplots()
    ax.plot(V[1:-1])  # Exclude terminal states
    ax.set_title(title)
    ax.set_xlabel("Capital")
    ax.set_ylabel("Value")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_policy(policy, title="Policy", save_path=None):
    fig, ax = plt.subplots()
    ax.plot(policy[1:-1])  # Exclude terminal states
    ax.set_title(title)
    ax.set_xlabel("Capital")
    ax.set_ylabel("Action")
    if save_path:
        plt.savefig(save_path)
    plt.show()
