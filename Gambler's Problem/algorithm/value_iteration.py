import numpy as np
from .policy import Policy

class ValueIteration:
    def __init__(self, env, theta=1e-2):
        self.env = env
        self.theta = theta
        # Initialize value function randomly
        self.V = np.zeros(env.winning_capital + 1)
        self.policy = Policy(env)

    def value_iteration(self):
        counter = 0
        while True:
            delta = 0
            counter += 1
            for s in self.env.states():
                v = self.V[s]
                action_returns = {
                    a: self.env.expected_return(s, a, self.V)
                    for a in self.env.actions(s)
                }
                if action_returns:
                    self.V[s] = max(action_returns.values())
                delta = max(delta, abs(v - self.V[s]))
            print(f"  Value Iteration Iteration {counter}, delta={delta}")
            if delta < self.theta:
                break

    def policy_improvement(self):
        for s in self.env.states():
            action_returns = {
                a: self.env.expected_return(s, a, self.V)
                for a in self.env.actions(s)
            }
            if action_returns:
                best_action = max(action_returns, key=action_returns.get)
                self.policy.update(s, best_action)
        return

    def run(self):
        self.value_iteration()
        self.policy_improvement()
        return self.policy, self.V
