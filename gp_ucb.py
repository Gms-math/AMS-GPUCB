import math
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel


class GPUCBHelper:
    def __init__(self, beta=2.0):
        self.beta = beta
        self.X = []
        self.y = []
        self.is_fit = False

        kernel = (
            C(1.0, (1e-4, 1e4))
            * Matern(length_scale=1.0, nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e1))
        )

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=42,
        )

    def add_observation(self, x, y):
        self.X.append(np.array(x, dtype=float))
        self.y.append(float(y))

        # Only fit once we have enough data
        if len(self.X) >= 5:
            X = np.array(self.X, dtype=float)
            y = np.array(self.y, dtype=float)
            self.gp.fit(X, y)
            self.is_fit = True

    def predict(self, x):
        if not self.is_fit:
            return 0.0, 1.0

        X = np.array([x], dtype=float)
        mu, std = self.gp.predict(X, return_std=True)
        return float(mu[0]), float(std[0])

    def ucb(self, x):
        mu, sigma = self.predict(x)
        return mu + math.sqrt(self.beta) * sigma