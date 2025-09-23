import argparse
import numpy as np
from env.environment import CarRentalEnv
from algorithm.policy_iteration import PolicyIteration
from utils.visualization import plot_policy, plot_value

def main(policy_plot_file_path, value_function_plot_file_path,
         policy_data_file_path, value_data_file_path):
    env = CarRentalEnv()
    solver = PolicyIteration(env)
    policy, V = solver.run()

    # Convert defaultdicts to numpy arrays
    policy_matrix = np.zeros((env.max_cars + 1, env.max_cars + 1))
    value_matrix = np.zeros((env.max_cars + 1, env.max_cars + 1))
    
    for i in range(env.max_cars + 1):
        for j in range(env.max_cars + 1):
            policy_matrix[i, j] = policy.policy[(i, j)]
            value_matrix[i, j] = V[(i, j)]
    
    # Save raw arrays
    np.save(policy_data_file_path, policy_matrix)
    np.save(value_data_file_path, value_matrix)
    
    # Save plots
    plot_policy(policy_matrix, max_cars=env.max_cars, save_path=policy_plot_file_path)
    plot_value(value_matrix, max_cars=env.max_cars, title="Optimal State-Value Function", save_path=value_function_plot_file_path)


    print(f"Policy saved at {policy_data_file_path}")
    print(f"Value function saved at {value_data_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jack's Car Rental with Policy Iteration")
    parser.add_argument("--policy_plot_file_path", type=str, required=True, help="Path to save policy plot image")
    parser.add_argument("--value_function_plot_file_path", type=str, required=True, help="Path to save value function plot image")
    parser.add_argument("--policy_data_file_path", type=str, required=True, help="Path to save raw policy array (.npy)")
    parser.add_argument("--value_data_file_path", type=str, required=True, help="Path to save raw value array (.npy)")

    args = parser.parse_args()

    main(
        args.policy_plot_file_path,
        args.value_function_plot_file_path,
        args.policy_data_file_path,
        args.value_data_file_path,
    )
