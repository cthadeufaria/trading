"""Main trading logic."""
from information import Information
from functions import KalmanFilter
from model import GLang
import numpy as np
import matplotlib.pyplot as plt

binance = Information()
binance.ping()
# binance.tickers_list(market='BUSD')

tickers = ['BTCBUSD'] # 1st test only for BTC/BUSD - 1D Kalman Filter
# tickers = binance.tickers

klines = binance.candlestick(tickers, interval='1d')

# initialize objects
glang = GLang(
    float(klines[tickers[0]]['close'][0]), 
    (float(klines[tickers[0]]['close'][0]) - ((float(klines[tickers[0]]['high'][0]) + float(klines[tickers[0]]['low'][0])) / 2))
)
filter = KalmanFilter()
d = len(tickers)

# initial conditions
X = np.array(np.ones(d) * glang.mu)[:, np.newaxis]
P = np.array(np.eye(d) * glang.P)
A = np.eye(d)
Q = np.array(np.eye(d) * glang.Q)
B = np.array([[0]])
U = np.array([[0]])
Y = np.array(np.ones(d) * glang.mu)[:, np.newaxis]
H = np.eye(d)
# R = np.array(np.ones(d) * glang.R)[:, np.newaxis]

# for ticker in klines:
for k in range(1, len(klines[tickers[0]]['close'])):
    for n in klines.items():
        glang.update(float(n[1]['volume'][k-1]), float(n[1]['volume'][k]), P)
        R = np.array(np.ones(d) * glang.R)
        Y = np.array(np.ones(d) * float(n[1]['close'][k]))[:, np.newaxis]
        print('Percent state prediction error: ', ((Y-X)/Y)*100)
        print(X, Y, P, R, float(n[1]['volume'][k-1]), float(n[1]['volume'][k]))
        X, P = filter.predict(X, P, A, Q, B, U)
        X, P = filter.update(X, P, Y, H, R)

# TODO 
# test variance prediction error
# expand price/variance prediction logic for multiple assets
# implement Markowitz portfolio model


