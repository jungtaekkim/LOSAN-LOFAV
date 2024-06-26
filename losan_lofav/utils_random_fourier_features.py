import numpy as np


class RFF:
    def __init__(self, dim_problem, dim_features, random_state):
        self.dim_problem = dim_problem 
        self.dim_features = dim_features
        self.random_state = random_state

        self.W = self.get_W()
        self.b = self.get_b()

    def get_W(self):
        return self.random_state.randn(self.dim_problem, self.dim_features)

    def get_b(self):
        return self.random_state.uniform(low=0.0, high=2 * np.pi, size=(self.dim_features, ))

    def get_rff(self, X):
        return np.sqrt(2 / self.dim_features) * np.cos(np.matmul(X, self.W) + self.b)
