import numpy as np


class GamblerEnv:
    """Environment for Gambler's Problem."""
    def __init__(self, 
                 winning_capital=100,
                 prob_head=0.25):
        self.winning_capital = winning_capital
        self.prob_head = prob_head
        self.gamma = 1  # No discounting
    def states(self):
        return [capital for capital in range(self.winning_capital + 1)]

    def actions(self, state):
        capital = state
        if capital == 0 or capital == self.winning_capital:
            return []
        max_bet = min(capital, self.winning_capital - capital)
        return list(range(1, max_bet + 1))

    def expected_return(self, state, action, V):
        """Expected return for taking action in state given value function V."""
        capital = state
        bet = action

        # If the coin lands heads
        win_state = capital + bet
        win_reward = 1 if win_state == self.winning_capital else 0
        win_value = V[win_state]

        # If the coin lands tails
        lose_state = capital - bet
        lose_reward = 0
        lose_value = V[lose_state]

        expected_return = (self.prob_head * (win_reward + self.gamma * win_value) +
                           (1 - self.prob_head) * (lose_reward + self.gamma * lose_value))
        return expected_return