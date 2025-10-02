import argparse
import numpy as np
from env.environment import GamblerEnv
from algorithm.value_iteration import ValueIteration
from utils.visualization import plot_policy, plot_value

def main(policy_plot_file_path, value_function_plot_file_path,
         policy_data_file_path, value_data_file_path, prob_head):
    env = GamblerEnv(prob_head=prob_head)
    solver = ValueIteration(env)
    policy, V = solver.run()

    # Convert defaultdicts to numpy arrays
    policy_matrix = np.zeros(env.winning_capital + 1)
    value_matrix = np.zeros(env.winning_capital + 1)
    for s in env.states():
        policy_matrix[s] = policy.action(s)
        value_matrix[s] = V[s]

    # Save raw arrays
    np.save(policy_data_file_path, policy_matrix)
    np.save(value_data_file_path, value_matrix)
    
    # Save plots
    plot_policy(policy_matrix, save_path=policy_plot_file_path, title=f"Optimal Policy (ph={prob_head})")
    plot_value(value_matrix, title=f"Optimal State-Value Function (ph={prob_head})", save_path=value_function_plot_file_path)


    print(f"Policy saved at {policy_data_file_path}")
    print(f"Value function saved at {value_data_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jack's Car Rental with Policy Iteration")
    parser.add_argument("--policy_plot_file_path", type=str, required=True, help="Path to save policy plot image")
    parser.add_argument("--value_function_plot_file_path", type=str, required=True, help="Path to save value function plot image")
    parser.add_argument("--policy_data_file_path", type=str, required=True, help="Path to save raw policy array (.npy)")
    parser.add_argument("--value_data_file_path", type=str, required=True, help="Path to save raw value array (.npy)")
    parser.add_argument("--prob_head", type=float, default=0.4, help="Probability of heads in the coin flip")
    args = parser.parse_args()

    main(
        args.policy_plot_file_path,
        args.value_function_plot_file_path,
        args.policy_data_file_path,
        args.value_data_file_path,
        args.prob_head
    )
