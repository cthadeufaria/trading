"""Main trading logic."""
from information import Information
from controller import KalmanFilter
from model import GLang
import numpy as np
import matplotlib.pyplot as plt

binance = Information()
binance.ping()
binance.tickers_list(market='BUSD')
tickers = binance.tickers
# tickers = ['BTCBUSD', 'BNBBUSD'] 
c = binance.candlestick(tickers, interval='1d')
klines = {k:v for k, v in c.items() if len(v) == 1000} # clean klines

# initialize objects
close_price = np.array([float(df['close'][0]) for df in list(klines.values())])[:, np.newaxis]
variance = np.diagflat([(float(df['close'][0]) - ((float(df['high'][0]) - float(df['low'][0])) / 2)) for df in list(klines.values())])
glang = GLang(close_price, variance)
filter = KalmanFilter()
d = len(klines)

# initial conditions
X = glang.mu
P = glang.P
A = np.eye(d)
Q = glang.Q
B = np.zeros((d, d))
U = np.zeros(d)[:, np.newaxis]
Y = glang.mu
H = np.eye(d)

for k in range(1, len(list(klines.values())[0]['close'])):
    last_volume = np.array([float(df['volume'][k-1]) for df in list(klines.values())])[:, np.newaxis]
    current_volume = np.array([float(df['volume'][k]) for df in list(klines.values())])[:, np.newaxis]
    glang.update(last_volume, current_volume, P)
    Y = np.array([float(df['close'][k]) for df in list(klines.values())])[:, np.newaxis]
    print(
        'Mean percent state prediction error: ', np.mean(((Y-X)/Y)*100), 
        "\n"
    )
    X, P = filter.predict(X, P, A, Q, B, U)
    X, P = filter.update(X, P, Y, H, glang.R)

# TODO 
# Check mean percent state prediction error peaks
# test which time period in binance.candlestick has the best predictions
# test variance prediction error
# expand price/variance prediction logic for multiple assets [ok]
    # update Glang model for accomodating arrays [ok]
# implement Markowitz portfolio model
# update binance.tickers to a faster code
# clean klines items that don't have 1k observations after candlestick method [ok]


