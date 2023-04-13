"""Main trading logic."""
from information import Information

binance = Information()
binance.ping()
binance.tickers_list(market='BUSD')