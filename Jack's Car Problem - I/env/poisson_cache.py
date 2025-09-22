import math

class PoissonCache:
    """Cache Poisson probabilities for efficiency."""
    def __init__(self, max_n=40, lambdas=(3, 4, 3, 2)):
        self.max_n = max_n
        self.cache = {}
        for lam in lambdas:
            for n in range(max_n + 1):
                self.cache[(n, lam)] = self._poisson(n, lam)

    def _poisson(self, n, lam):
        return math.exp(-lam) * (lam**n) / math.factorial(n)

    def prob(self, n, lam):
        return self.cache.get((n, lam), 0.0)
