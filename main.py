"""Main trading logic."""
from information import Information

binance = Information()
binance.ping()
binance.tickers_list(market='BUSD')
klines = binance.candlestick(binance.tickers)