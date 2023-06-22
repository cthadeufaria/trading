"""Any portfolio assembling model that takes up past, current and future (predicted) information."""
import numpy as np
import pandas as pd
from scipy import optimize
from pypfopt.efficient_frontier import EfficientFrontier

class Markowitz:
    def __init__(self, klines, X, rf=0.) -> None:
        i = len(klines.values())
        j = len(list(klines.values())[0]['close'])
        returns = np.zeros(shape=(i, j))
        for k in range(0, i):
            close = np.array(pd.concat([list(klines.values())[k]['close'], pd.Series(str(X[k, 0]))], ignore_index=True).astype(float))[np.newaxis, :]
            returns[k] = np.diff(close)/close[:, :-1]
        self.returns = returns
        self.variance_covariance_matrix = np.cov(self.returns)
        l = self.returns.shape[0]
        self.weights = np.ones(shape=(l, 1)) * (1/l)
        self.expected_returns = self.returns[:, -1][:, np.newaxis]
        self.rf = rf

    def optmize_sharpe_ratio(self, method, lower_bound, upper_bound):
        if method == 'scipy':
            def sharpe_ratio(weights, expected_returns, variance_covariance_matrix, rf):
                self.portfolio_expected_return = weights.T @ expected_returns
                self.portfolio_std = np.sqrt(weights.T @ variance_covariance_matrix @ weights)
                self.sharpe_ratio = (self.portfolio_expected_return - rf) / self.portfolio_std
                return -self.sharpe_ratio
            
            def constraints1(weights):
                A = np.ones(weights.shape)
                return A @ weights.T - 1
            
            def constraints2(weights):
                return sum(weights) - 1
            
            cons = (
                {'type': 'ineq', 'fun':constraints1},
                {'type': 'eq', 'fun':constraints2}
            )
            bnds = tuple([(lower_bound, upper_bound) for x in self.weights])
            opt = optimize.minimize (
                sharpe_ratio, 
                x0 = self.weights, 
                args = (self.expected_returns, self.variance_covariance_matrix, self.rf), 
                method = 'SLSQP', 
                bounds = bnds,
                constraints = cons,
                tol = 10**-3
            )
            self.weights = opt.x[:, np.newaxis]
        elif method == 'pypfopt':
            efficient_portfolio = EfficientFrontier(self.expected_returns, self.variance_covariance_matrix, weight_bounds=(lower_bound, upper_bound))
            weights = efficient_portfolio.max_sharpe() 
            clean_weights = efficient_portfolio.clean_weights() 
            print(clean_weights)
            efficient_portfolio.portfolio_performance(verbose=True)
        elif method == 'sga':
            pass
        else:
            raise Exception("Argument \"method\" has not an expected value.")