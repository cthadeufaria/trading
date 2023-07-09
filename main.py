"""Main trading logic."""
from information import Information
from functions import KalmanFilter, PortfolioFilter
from model import GLang
from portfolio import Markowitz
import matplotlib.pyplot as plt
import numpy as np

# get data from exchange
binance = Information()
binance.ping()
binance.tickers_list(market='BUSD')
tickers = binance.tickers
# tickers = ['BTCBUSD', 'BNBBUSD', 'DOGEBUSD', 'ETHBUSD', 'COMPBUSD', 'WAVESBUSD', 'AUDBUSD', 'GBPBUSD']
c = binance.candlestick(tickers, interval='1d')

# primary filter
filter = PortfolioFilter()
klines = filter.length(c)

# initialize objects
close_price = np.array([float(df['close'][0]) for df in list(klines.values())])[:, np.newaxis]
variance = np.diagflat([(float(df['close'][0]) - ((float(df['high'][0]) - float(df['low'][0])) / 2)) for df in list(klines.values())])
glang = GLang(close_price, variance)
filter = KalmanFilter()
d = len(klines)

# Kalman filter initial conditions
X = glang.mu
P = glang.P
A = np.eye(d)
Q = glang.Q
B = np.zeros((d, d))
U = np.zeros(d)[:, np.newaxis]
Y = glang.mu
H = np.eye(d)

# X, P prediction iterations
iterations = len(list(klines.values())[0]['close'])
for k in range(1, iterations):
    last_volume = np.array([float(df['volume'][k-1]) for df in list(klines.values())])[:, np.newaxis]
    current_volume = np.array([float(df['volume'][k]) for df in list(klines.values())])[:, np.newaxis]
    glang.update(last_volume, current_volume, P)
    Y = np.array([float(df['close'][k]) for df in list(klines.values())])[:, np.newaxis]
    if k == iterations - 1:
        print('Last iteration average percent state prediction error: ', np.mean(((Y-X)/Y)*100))
    X, P = filter.predict(X, P, A, Q, B, U)
    X, P = filter.update(X, P, Y, H, glang.R)

# Build Markowitz portfolio
markowitz = Markowitz(klines, X)
markowitz.optmize_sharpe_ratio(method='scipy', lower_bound=0., upper_bound=1.)
print('Linearly optimized portfolio: ', {markowitz.tickers[i]: markowitz.weights[i][0] for i in list(np.where(markowitz.weights >= 0.01)[0])})
print('Sharpe ratio: ', markowitz.sharpe_ratio)
print('Expected return: ', markowitz.portfolio_expected_return)
print('Expected risk: ', markowitz.portfolio_std, "\n")

# Build highest Sharpe ratio random portfolio
random_portfolio_expected_return, random_portfolio_std, random_sharpe_ratio, random_weights = markowitz.random_generator(n=10e5)
id = np.where(random_sharpe_ratio == max(random_sharpe_ratio))[0][0]
print('Best random portfolio: ', {markowitz.tickers[i]: random_weights[id][i] for i in list(np.where(random_weights[id] >= 0.01)[0])})
print('Sharpe ratio: ', random_sharpe_ratio[id])
print('Expected return: ', random_portfolio_expected_return[id])
print('Expected risk: ', random_portfolio_std[id], "\n")

# Plot random portfolios and optimized Sharpe portfolios
plt.figure(figsize=(10, 7))
plt.scatter(random_portfolio_std, random_portfolio_expected_return, c=random_sharpe_ratio, marker='o', alpha=0.3)
plt.scatter(markowitz.portfolio_std, markowitz.portfolio_expected_return, color='red', marker='*')
plt.scatter(random_portfolio_std[id], random_portfolio_expected_return[id], color='orange', marker='*')
plt.show()

print('================ Portfolio allocation finished successfully ================')


"""
TODO 
.check mean percent state prediction error peaks
.test which time period in binance.candlestick has the best predictions
.test variance prediction error
.implement Markowitz portfolio model []
    .check scipy optimization through plotting of efficient frontier [ok]
    .check maximum sharpe portfolio weights and tickers for random portfolio with many assets [ok]
    .how to use P matrix in Markowitz model? Why it's not positive semi-definite if replaced the diagonal by np.diagonal(P)? []
    .try gradient ascent for sharpe ratio optimization method [] 5
    .check if 'weights' matrix has the same order of assets as klines
    .filter n bigger expected returns [ok] 1
    .how to incorporate a risk free asset? based on usd [] 2
.implement PMPT optimization
.update to a faster code
    .binance.candlestick
    .binance.tickers
.calculate mse
.what's the minimum volatility portfolio?
.buy portfolio [] 4
.implement auto update on risk free asset
.find hour of day with more trading volume
"""

"""
DONE
.expand price/variance prediction logic for multiple assets [ok]
   .update Glang model for accomodating arrays [ok]
.clean klines items that don't have 1k observations after candlestick method [ok]
.create option to remove negative self.returns [ok]
    .remove asset data accordingly [ok]
.what's the best time to buy and sell? what are the datetimes of the klines in CET? [ok] 3
"""