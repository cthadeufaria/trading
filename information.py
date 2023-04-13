"""Any information channel."""
from connection import Connection
import requests

class Information(Connection):
    def __init__(self) -> None:
        super().__init__(environment='production')

    def tickers_list(self, market=None) -> None:
        r1 = requests.get(self.endpoints['exchange_info'], auth=(self.auth_dict['key'], self.auth_dict['skey']))
        # get all tickers list where dict['symbols']['status']=='TRADING':
        self.tickers = []
        for i in range(0, len(r1.json()['symbols'])):
            if r1.json()['symbols'][1]['status'] == 'TRADING':
                if r1.json()['symbols'][i]['quoteAsset'] == market:
                    self.tickers.append(r1.json()['symbols'][i]['symbol'])
                elif market == None:
                    self.tickers.append(r1.json()['symbols'][i]['symbol'])
        print(self.tickers)
