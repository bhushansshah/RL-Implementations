from collections import defaultdict

class Policy:
    """Represents a deterministic policy mapping states to actions."""
    def __init__(self, env):
        self.policy = defaultdict(int)
        for s in env.states():
            self.policy[s] = 0

    def action(self, state):
        return self.policy[state]

    def update(self, state, action):
        self.policy[state] = action
