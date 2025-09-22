import numpy as np
from .poisson_cache import PoissonCache

class CarRentalEnv:
    """Environment for Jack's Car Rental problem (Sutton & Barto Example 4.2)."""
    def __init__(self, 
                 max_cars=20, 
                 max_move=5, 
                 rental_reward=10, 
                 move_cost=2,
                 discount=0.9,
                 lambdas=(3, 4, 3, 2)):
        self.max_cars = max_cars
        self.max_move = max_move
        self.rental_reward = rental_reward
        self.move_cost = move_cost
        self.discount = discount
        self.lam_requests = lambdas[:2]
        self.lam_returns = lambdas[2:]
        self.poisson_cache = PoissonCache(max_n=max_cars*2, lambdas=lambdas)

    def states(self):
        return [(i, j) for i in range(self.max_cars + 1)
                       for j in range(self.max_cars + 1)]

    def actions(self, state):
        cars1, cars2 = state
        actions = []
        for move in range(-self.max_move, self.max_move + 1):
            if (0 <= cars1 - move <= self.max_cars) and (0 <= cars2 + move <= self.max_cars):
                actions.append(move)
        return actions

    def expected_return(self, state, action, V):
        """Expected return for taking action in state given value function V."""
        cars1, cars2 = state
        cars1 -= action
        cars2 += action

        reward = -abs(action) * self.move_cost
        value = 0.0

        for req1 in range(0, self.max_cars + 1):
            for req2 in range(0, self.max_cars + 1):
                prob_r1 = self.poisson_cache.prob(req1, self.lam_requests[0])
                prob_r2 = self.poisson_cache.prob(req2, self.lam_requests[1])
                prob = prob_r1 * prob_r2

                real_r1 = min(cars1, req1)
                real_r2 = min(cars2, req2)
                reward_r = (real_r1 + real_r2) * self.rental_reward

                cars1_ = cars1 - real_r1
                cars2_ = cars2 - real_r2

                for ret1 in range(0, self.max_cars + 1 - cars1_):
                    for ret2 in range(0, self.max_cars + 1 - cars2_):
                        prob_ret1 = self.poisson_cache.prob(ret1, self.lam_returns[0])
                        prob_ret2 = self.poisson_cache.prob(ret2, self.lam_returns[1])
                        prob_ret = prob_ret1 * prob_ret2

                        new_cars1 = min(self.max_cars, cars1_ + ret1)
                        new_cars2 = min(self.max_cars, cars2_ + ret2)

                        prob_total = prob * prob_ret
                        value += prob_total * (reward + reward_r + self.discount * V[new_cars1, new_cars2])

        return value
