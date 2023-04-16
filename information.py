"""Any information channel."""
from connection import Connection
import pandas as pd
import requests

class Information(Connection):
    def __init__(self) -> None:
        super().__init__()

    def tickers_list(self, market=None) -> None:
        r1 = requests.get(self.endpoints['exchange_info'], auth=(self.auth_dict['key'], self.auth_dict['skey']))
        # get all tickers list where dict['symbols']['status']=='TRADING':
        self.tickers = []
        for i in range(0, len(r1.json()['symbols'])): # TODO change for loop to list comprehension format
            if r1.json()['symbols'][1]['status'] == 'TRADING':
                if r1.json()['symbols'][i]['quoteAsset'] == market:
                    self.tickers.append(r1.json()['symbols'][i]['symbol'])
                elif market == None:
                    self.tickers.append(r1.json()['symbols'][i]['symbol'])
        print(self.tickers)

    def candlestick(self, tickers, limit=1000, interval='1h', endTime=None) -> dict:
        #   Kline intervals: m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
        #   1m / 3m / 5m / 15m / 30m / 1h / 2h / 4h / 6h / 8h / 12h / 1d / 3d / 1w / 1M
        data = {}
        if type(tickers) != list:
            tickers=[tickers]
        for t in tickers: # TODO change for loop to list comprehension format
            if endTime == None:
                url = self.endpoints['candlestick']+'?'+'symbol='+ t +'&interval='+interval+'&limit=' + str(limit)
            else:
                url = self.endpoints['candlestick']+'?'+'symbol='+ t +'&interval='+interval+'&limit=' + str(limit) + '&endTime=' + str(endTime)
            r4 = requests.get(url, auth=(self.auth_dict['key'], self.auth_dict['skey']))
            candle_columns=[
                'open_datetime', 
                'open', 'high', 
                'low', 'close', 
                'volume', 
                'close_datetime', 
                'quote_volume', 
                'n_trades', 
                'taker_buy_asset_vol', 
                'taker_buy_quote_vol', 
                'ignore'
            ]
            candle_info = pd.DataFrame(data=r4.json(), columns=candle_columns)
            data[t] = candle_info
        return data
