"""Any portfolio assembling model that takes up past, current and future (predicted) information."""
import numpy as np
import pandas as pd
from scipy import optimize
from pypfopt.efficient_frontier import EfficientFrontier

class Markowitz:
    """Class to implement Markowitz portfolio model."""
    def __init__(self, klines, X, rf=0., filter_positive_returns=True, filter_n_bigger_returns=True, n=10) -> None:
        i = len(klines.values())
        j = len(list(klines.values())[0]['close'])
        tickers = list(klines.keys())
        returns = np.zeros(shape=(i, j))
        for k in range(0, i):
            close = np.array(pd.concat([list(klines.values())[k]['close'], pd.Series(str(X[k, 0]))], ignore_index=True).astype(float))[np.newaxis, :]
            returns[k] = np.diff(close)/close[:, :-1]
        if filter_positive_returns:
            returns = np.where(returns[:, -1][:, np.newaxis] <= 0, np.zeros(shape=(returns[0, :].shape))[np.newaxis, :], returns)
        if filter_n_bigger_returns:
            threshold = returns[:, -1][np.argsort(returns[:, -1])][-(n+1)]
            returns = np.where(returns[:, -1][:, np.newaxis] <= threshold, np.zeros(shape=(returns[0, :].shape))[np.newaxis, :], returns)
        keys = [False if a==0. else True for a in returns[:, -1]]
        self.tickers = [t for (t, k) in zip(tickers, keys) if k==True]
        self.returns = returns[~np.all(returns == 0, axis=1)]
        self.variance_covariance_matrix = np.cov(self.returns)
        # if P_variance:
        #     diag = [t for (t, k) in zip(np.diagonal(P), keys) if k==True]
        #     np.fill_diagonal(self.variance_covariance_matrix, diag)
        self.expected_returns = self.returns[:, -1][:, np.newaxis]
        l = self.expected_returns.shape[0]
        w = np.random.rand(l)
        self.weights = w/sum(w)
        # self.weights = np.ones(shape=(l, 1)) * (1/l)
        self.rf = rf

    def sharpe_gradient(self):
        pass

    def optmize_sharpe_ratio(self, method, lower_bound=0, upper_bound=1):
        if method == 'scipy':
            def sharpe_ratio(weights, expected_returns, variance_covariance_matrix, rf):
                self.portfolio_expected_return = (weights.T @ expected_returns)[0]
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
    
    def random_generator(self, n=1000):
        l = self.expected_returns.shape[0]
        random_portfolio_expected_return = []
        random_portfolio_std = []
        random_sharpe_ratio = []
        random_weights = []
        for _ in range(int(n)):
            w = np.random.rand(l)
            weights = w/sum(w)
            random_portfolio_expected_return.append((weights.T @ self.expected_returns)[0])
            random_portfolio_std.append(np.sqrt(weights.T @ self.variance_covariance_matrix @ weights))
            random_sharpe_ratio.append((random_portfolio_expected_return[-1] - self.rf) / random_portfolio_std[-1])
            random_weights.append(weights)
        return random_portfolio_expected_return, random_portfolio_std, random_sharpe_ratio, random_weights
