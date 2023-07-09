"""Any connection channel with available exchanges."""
import os, requests, hmac, hashlib
from urllib.parse import urlencode

class Connection:
    def __init__(self, environment='production') -> None:
        self.environment = environment
        self.endpoints = {}
        endpoints = {
            'test' : '/api/v3/ping',
            'server_time' : '/api/v3/time',
            'exchange_info' : '/api/v3/exchangeInfo',
            'order_book' : '/api/v3/depth',
            'candlestick' : '/api/v3/klines',
            'avg_price' : '/api/v3/avgPrice',
            'best_price' : '/api/v3/ticker/bookTicker',
            'acc_info' : '/api/v3/account',
            'acc_snapshot' : '/sapi/v1/accountSnapshot',
            'price' : '/api/v3/ticker/price',
            'price_hist' : '/api/v3/historicalTrades',
            'order' : '/api/v3/order',
            'test_order' : '/api/v3/order/test',
            'trades' : '/api/v3/myTrades',
            'options_info' : '/vapi/v1/optionInfo',
            'options_order_book' : '/vapi/v1/depth',
            'options_mark_price' : '/vapi/v1/mark',
        }
        if self.environment == 'test':
            main_endpoint = 'https://testnet.binance.vision'
            self.auth_dict = {
                'key' : os.environ.get('TEST_KEY'),
                'skey' : os.environ.get('TEST_SKEY'),
            }
        elif self.environment == 'production':
            main_endpoint = 'https://api1.binance.com'
            options_endpoint = 'https://vapi.binance.com'
            self.auth_dict = {
                'key' : os.environ.get('SPOT_KEY'),
                'skey' : os.environ.get('SPOT_SKEY'),
            }
        # complete endpoints strings
        for endpoint in endpoints:
            if endpoint[0:7] == 'options':
                self.endpoints[endpoint] = options_endpoint + endpoints[endpoint]
            else:
                self.endpoints[endpoint] = main_endpoint + endpoints[endpoint]
        # print(self.endpoints)

    def ping(self) -> None:
        """Ping the api server."""
        r = requests.get(self.endpoints['test'])
        print('ping: ' + str(r))
        if str(r) == "<Response [200]>":
            pass
        else:
            raise Exception("Connection with exchange server unsuccessful.")

    def sha256_signature(self, endpoint_params) -> None:
        """Create hashed sign dict for instantiated object."""
        secret = self.auth_dict['skey']
        params = urlencode(endpoint_params)
        hashedsig = hmac.new(secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()
        self.hashedsig_dict = {
            "signature" : hashedsig
        }