"""Any portfolio assembling model that takes up past, current and future (predicted) information and assembles a portfolio."""
import numpy as np

class Markowitz:
    def __init__(self) -> None:
        pass

    def calculate_expected_returns(self, expected_prices, last_prices):
        self.expected_prices = expected_prices
        self.last_prices = last_prices
        self.expected_returns = (expected_prices-last_prices)/last_prices

    def calculate_variance_covariance_matrix(self, klines):
        i = len(klines.values())
        j = len(list(klines.values())[0]['close']) - 1
        returns = np.zeros(shape=(i, j))
        for k in range(0, i):
            returns[k] = np.diff(np.array(list(klines.values())[k]['close'].astype(float))[np.newaxis, :])/np.array(list(klines.values())[k]['close'].astype(float))[np.newaxis, :][:, :-1]
        self.returns = returns
        self.variance_covariance_matrix = np.cov(self.returns)

    def efficient_frontier(self):
        pass

    def optmize_sharpe_ratio(self):
        pass