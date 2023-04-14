import pandas as pd
import requests, time, hmac, hashlib
from urllib.parse import urlencode
from initialize import initialize

endpoints = initialize('prod')[0]
auth_dict = initialize('prod')[1]


def binance_data(): # used for debugging
    # random requests
    r2 = requests.get(endpoints['order_book']+'?'+'symbol=BNBBTC&limit=100', auth=(auth_dict['key'], auth_dict['skey']))
    r3 = requests.get(endpoints['avg_price']+'?'+'symbol=BNBBTC', auth=(auth_dict['key'], auth_dict['skey']))
    r7 = requests.get(endpoints['best_price']+'?'+'symbol=BNBBTC', auth=(auth_dict['key'], auth_dict['skey']))


def order_book(tickers, key):
    # key='spot' or key=0 -> spot info / key='option' or key=1 -> option info
    if key==0:
        endpoint=endpoints['order_book']
    elif key==1:
        endpoint=endpoints['options_order_book']

    order_book = {}
    for t in tickers:
        r = requests.get(endpoint+'?'+'symbol='+str(t)+'&limit=100', auth=(auth_dict['key'], auth_dict['skey']))
        order_book[t] = r.json()
    return order_book
    

def options_info():
    r = requests.get(endpoints['options_info'], auth=(auth_dict['key'], auth_dict['skey']))
    return r.json()


def markPrice(tickers):
    markPrice = {}
    for t in tickers:
        r = requests.get(endpoints['options_mark_price'] +'?'+'symbol='+str(t), auth=(auth_dict['key'], auth_dict['skey']))
        markPrice[t] = r.json()
    return markPrice


def ping():
    r = requests.get(endpoints['test'])
    print('ping: ' + str(r))
    return r


def get_timestamp():
    t = int(time.time()*1000)
    servertime = requests.get(endpoints['server_time'])
    st = servertime.json()['serverTime']
    return st, t


def tickers_list(market=None):
    r1 = requests.get(endpoints['exchange_info'], auth=(auth_dict['key'], auth_dict['skey']))

    # get all tickers list where dict['symbols']['status']=='TRADING':
    tickers = []
    for i in range(0, len(r1.json()['symbols'])):
        if r1.json()['symbols'][1]['status'] == 'TRADING':
            if r1.json()['symbols'][i]['quoteAsset'] == market:
                tickers.append(r1.json()['symbols'][i]['symbol'])
            elif market == None:
                tickers.append(r1.json()['symbols'][i]['symbol'])

    print(tickers)
    # print('status_code=' + str(r1.status_code) + ';' + str(r1.headers['content-type']))
    
    return tickers


def price_hist(tickers=['BNBBTC'], apikey=auth_dict['key']):
    data={}
    p1=[]
    p2=[]
    if type(tickers) != list:
        tickers=[tickers]
    for i in tickers:
        params = {
            'symbol' : i,
            'limit' : 1000,
        }
        headers = {
            "X-MBX-APIKEY" : apikey,
        }
        r = requests.get(endpoints['price_hist'], params=params, headers=headers).json()
        for j in r:
            p1.append(j['time'])
            p2.append(float(j['price']))
        data[i] = pd.DataFrame(index=p1.copy(), data=p2.copy(), columns=[i])
        p1.clear()
        p2.clear()
    return data


def candlestick(tickers, limit=1000, interval='1h', endTime = None):
#     Kline/Candlestick chart intervals:
# m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
#     1m / 3m / 5m / 15m / 30m / 1h / 2h / 4h / 6h / 8h / 12h / 1d / 3d / 1w / 1M

    data = {}
    if type(tickers) != list:
        tickers=[tickers]
    for t in tickers:
        if endTime == None:
            url = endpoints['candlestick']+'?'+'symbol='+ t +'&interval='+interval+'&limit=' + str(limit)
        else:
            url = endpoints['candlestick']+'?'+'symbol='+ t +'&interval='+interval+'&limit=' + str(limit) + '&endTime=' + str(endTime)
        r4 = requests.get(url, auth=(auth_dict['key'], auth_dict['skey']))
        candle_columns=['open_datetime', 'open', 'high', 'low', 'close', 'volume', 'close_datetime', 'quote_volume', 'n_trades', 'taker_buy_asset_vol', 'taker_buy_quote_vol', 'ignore']
        candle_info = pd.DataFrame(data=r4.json(), columns=candle_columns)
        # path = '/home/carlos/Documents/BTC_data_2/'+t+'.csv'
        # candle_info.to_csv(path)
        data[t] = candle_info
        # Plot
        # fig = go.Figure(data=[go.Candlestick(x=candle_info['open_datetime'], open=candle_info['open'], high=candle_info['high'], low=candle_info['low'], close=candle_info['close'])])
        # fig.show()
    return data


def sha256_signature(endpoint_params, skey=auth_dict['skey']):
    secret = skey
    params = urlencode(endpoint_params)
    hashedsig = hmac.new(secret.encode('utf-8'), params.encode('utf-8'), hashlib.sha256).hexdigest()
    hashedsig_dict = {
        "signature" : hashedsig
    }
    return hashedsig_dict


def account_info(apikey=auth_dict['key']):
    servertimeint = get_timestamp()[0]
    endpoint_params = {
        "timestamp" : servertimeint,
    }
    hashedsig_dict = sha256_signature(endpoint_params)
    endpoint_params.update(hashedsig_dict)
    userdata = requests.get(endpoints['acc_info'],
        params = endpoint_params,
        headers = {
            "X-MBX-APIKEY" : apikey,
        }
    )
    return userdata.json()


def account_snapshot(apikey=auth_dict['key']):
    servertimeint = get_timestamp()[0]
    endpoint_params = {
            "type" : "SPOT",
            "timestamp" : servertimeint,
        }
    hashedsig_dict = sha256_signature(endpoint_params)
    endpoint_params.update(hashedsig_dict)
    data = requests.get(endpoints['acc_snapshot'],
        params = endpoint_params,
        headers = {
            "X-MBX-APIKEY" : apikey,
        }
    )
    return data.json()


def trades(symbols, apikey=auth_dict['key']):
    FullData = []
    for symbol in symbols:
        servertimeint = get_timestamp()[0]
        endpoint_params = {
                "symbol" : symbol,
                "timestamp" : servertimeint,
            }
        hashedsig_dict = sha256_signature(endpoint_params)
        endpoint_params.update(hashedsig_dict)
        data = requests.get(endpoints['trades'],
            params = endpoint_params,
            headers = {
                "X-MBX-APIKEY" : apikey,
            }
        )
        FullData.append(data.json())
    return FullData


def test_order(symbol, side, type, quantity, apikey=auth_dict['key']):
    servertimeint = get_timestamp()[0]
    endpoint_params = {
            "symbol" : symbol,
            "side" : side,
            "type" : type,
            "quantity" : quantity,
            "timestamp" : servertimeint,
        }
    hashedsig_dict = sha256_signature(endpoint_params)
    endpoint_params.update(hashedsig_dict)
    data = requests.get(endpoints['test_order'],
        params = endpoint_params,
        headers = {
            "X-MBX-APIKEY" : apikey,
        }
    )
    return data.json()


def order(symbol, side, type, timeInForce, quantity, price, apikey=auth_dict['key']):
    servertimeint = get_timestamp()[0]
    endpoint_params = {
            "symbol" : symbol,
            "side" : side,
            "type" : type,
            "timeInForce" : timeInForce,
            "quantity" : quantity,
            "price" : price,
            "timestamp" : servertimeint,
        }
    hashedsig_dict = sha256_signature(endpoint_params)
    endpoint_params.update(hashedsig_dict)
    data = requests.get(endpoints['order'],
        params = endpoint_params,
        headers = {
            "X-MBX-APIKEY" : apikey,
        }
    )
    return data.json()


def balances():
    Na = {}
    # d = account_snapshot()['snapshotVos'][0]['data']['balances']
    # for i in d:
    #     Na[i['asset']] = float(i['free'])
    d = account_info()['balances']
    for i in d:
        if float(i['free']) > 0:
            Na[i['asset']] = float(i['free'])
    return Na


def prices():
    Pbuy = {}
    Psell = {}
    r = requests.get(endpoints['best_price'], auth=(auth_dict['key'], auth_dict['skey']))
    for i in r.json():
        Pbuy[i['symbol']] = i['askPrice']
        Psell[i['symbol']] = i['bidPrice']
    return Pbuy, Psell

    