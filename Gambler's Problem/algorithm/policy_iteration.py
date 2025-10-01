import numpy as np
from .policy import Policy

class PolicyIteration:
    def __init__(self, env, theta=1e-2):
        self.env = env
        self.theta = theta
        # Initialize value function randomly
        self.V = np.random.rand(env.winning_capital + 1)
        self.policy = Policy(env)

    def policy_evaluation(self):
        counter = 0
        while True:
            delta = 0
            counter += 1
            for s in self.env.states():
                v = self.V[s]
                a = self.policy.action(s)
                self.V[s] = self.env.expected_return(s, a, self.V)
                delta = max(delta, abs(v - self.V[s]))
            print(f"  Policy Evaluation Iteration {counter}, delta={delta}")
            if delta < self.theta:
                break

    def policy_improvement(self):
        stable = True
        for s in self.env.states():
            old_action = self.policy.action(s)
            action_returns = {
                a: self.env.expected_return(s, a, self.V)
                for a in self.env.actions(s)
            }
            best_action = max(action_returns, key=action_returns.get)
            self.policy.update(s, best_action)
            if best_action != old_action:
                stable = False
        return stable

    def run(self):
        iteration = 0
        while True:
            iteration += 1
            print(f"Iteration {iteration}...")
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print("Policy converged.")
                break
        return self.policy, self.V
