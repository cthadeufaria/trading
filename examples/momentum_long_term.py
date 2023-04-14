# Portfolio 1 - Intermediate-Term Momentum
    # Identify universe (sample_space) (ok)
    # Remove outliers (beta) (ok) (momentum_outliers) (ok)
    # Momentum screen (momentum_quantity)
    # Momentum quality (momentum_quality)
    # Invest with conviction
        # Find best value assets
        # Define portfolio size

########################################################################################################

from handle_api import tickers_list, candlestick, prices, trades, balances
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import time, datetime, pygsheets, json


def sample_space(OneYearData, MinimumValue=0.5, MinimumVolume=10000, filter_period = 30):
    # MinimumValue = Minimum value of asset in USDT / MinimumVolume = Minimum average volume of asset in last 30 days
    # Exclude assets with less than 12 months of return data (ok)
    # Filter largest assets (ok)
    # Eliminate less liquid assets based on average daily volume (ok)

    EnoughReturnData = {k:v for k,v in OneYearData.items() if len(v) == 365}
    print('EnoughReturnData length = ' + str(len(EnoughReturnData)))

    LargestAssets = {k:v for k,v in EnoughReturnData.items() if sum(pd.to_numeric(v['high'][-filter_period:]))/filter_period >= MinimumValue}
    print('LargestAssets length = ' + str(len(LargestAssets)))

    # LargestAssets_2: use market_cap as variable
    LargestAssets_2 = {k:v for k,v in EnoughReturnData.items() if sum(pd.to_numeric(v['high'][-filter_period:]))/filter_period >= MinimumValue}
    print('LargestAssets length = ' + str(len(LargestAssets)))

    LiquidAssets = {k:v for k,v in LargestAssets.items() if sum(pd.to_numeric(v['volume'][-filter_period:]))/filter_period >= MinimumVolume}
    print('LiquidAssets length = ' + str(len(LiquidAssets)))

    last_decile_1 = decile(data = EnoughReturnData, filter_period = filter_period, column = 'high')[-1]
    ld_1_group = {k:v for k, v in EnoughReturnData.items() if sum(pd.to_numeric(v['high'][-filter_period:]))/filter_period >= last_decile_1}

    last_decile_2 = decile(data = EnoughReturnData, filter_period = filter_period, column = 'volume')[-1]
    ld_2_group = {k:v for k, v in EnoughReturnData.items() if sum(pd.to_numeric(v['volume'][-filter_period:]))/filter_period >= last_decile_2}

    print(ld_1_group.keys())
    print(ld_2_group.keys())

    return LiquidAssets


def beta(data, base_asset='BTCUSDT', p=0.3, filter_period = 30):
    # Eliminate assets with abs(beta) greater than 'p' (ok)

    df = pd.DataFrame()
    for s in data.keys():
        if len(df) == 0:
            df = pd.DataFrame(
                    data=(pd.to_numeric(data[s]['close'])-pd.to_numeric(data[s]['open']))/pd.to_numeric(data[s]['open']),
                    columns=[s]
                ).set_index(data[s]['open_datetime'])
        else:
            df = df.join(
                pd.DataFrame(
                    data=(pd.to_numeric(data[s]['close'])-pd.to_numeric(data[s]['open']))/pd.to_numeric(data[s]['open']),
                    columns=[s]
                ).set_index(data[s]['open_datetime']),
                how='left'
            )
    df.dropna(axis=1, inplace=True)

    beta = {}
    for c in df.columns:
        # Create arrays for x and y variables in the regression model
        x = np.array(df[c]).reshape((-1,1))
        y = np.array(df[base_asset])
        # Define the model and type of regression
        model = LinearRegression().fit(x, y)
        beta[c] = (model.coef_[0])
        print(str(c)+'s beta = '+str(model.coef_[0]))

    beta_abs = {k:abs(v) for k,v in beta.items()}
    first_decile = decile(data = beta_abs, filter_period = filter_period, column = None)[0]
    ld_3_group = {k:v for k, v in beta_abs.items() if v<=first_decile}

    print(ld_3_group.keys())

    beta = {k:v for k,v in beta.items() if abs(v)<=p}

    LowBetaAssets = {k:v for k,v in data.items() if (k == pd.Series(beta.keys())).any()}
    # LowBetaAssets['BTCUSDT'] = data['BTCUSDT']
    # LowBetaAssets['ETHUSDT'] = data['ETHUSDT']
    print('LowBetaAssets length: ' + str(len(LowBetaAssets)) + ' rows')

    return LowBetaAssets


def calculate_momentum(data, periods = [180, 270, 360]):
    # Calculate momentum for given time periods
    # periods = periods' list to calculate momentum
    MomentumData = {}
    info = {}
    for k in data.keys():
        for i in periods:
            length = len(data[k]['close'])
            end = length - 7
            beginning = max(end - i, 0)
            info[str(i)] = (pd.to_numeric(data[k]['close'][end]) - pd.to_numeric(data[k]['open'][beginning])) / pd.to_numeric(data[k]['open'][beginning])
        MomentumData[k] = info.copy()
        for i in periods:
            info.pop(str(i))
    
    return MomentumData


def momentum_outliers(data, MomentumData, screen = 0.0, periods = [180, 270]):
    # Remove assets with negative 6 and 9-month momentum measure from data
    MomentumDataNoOutliers = {k:v for k,v in MomentumData.items() if v[str(periods[1])] >= screen and v[str(periods[0])] >= screen}
    
    OutliersRemoved = {k:v for k,v in data.items() if (k == pd.Series(MomentumDataNoOutliers.keys())).any()}

    print('OutliersRemoved length = ' + str(len(OutliersRemoved)) + ' rows')

    return OutliersRemoved, MomentumDataNoOutliers


def momentum_quantity(data, MomentumDataNoOutliers):
    # Select assets with higher 12-month momentum
    # Create DataFrame for storing 12-month momentum data
    dfMomentum = pd.DataFrame(data=None, columns=['asset', 'momentum'], index=range(0, len(MomentumDataNoOutliers)))
    i = 0
    for k in MomentumDataNoOutliers.keys():
        dfMomentum['asset'][i] = k
        dfMomentum['momentum'][i] = MomentumDataNoOutliers[k]['360']
        i+=1
    
    # Filter positive 12-month momentum
    dfMomentum = dfMomentum[dfMomentum['momentum']>1.0]

    CleanData = {k:v for k,v in data.items() if (k == dfMomentum['asset']).any()}

    print('CleanData length = ' + str(len(CleanData)) + ' rows')

    return CleanData


def momentum_quality(data, n=20, p=0.5, cut=0):
    # ID = sign(PRET) * (% months negative - % months positive)
    # -1 <= ID <= 1
    # -1 = high quality momentum / 1 = bad quality momentum
    quality = {}
    for d in data.keys():
        data[d]['return'] = pd.to_numeric(data[d]['close'])-pd.to_numeric(data[d]['open'])
        data[d]['return_n'] = pd.to_numeric(data[d]['close'])-pd.to_numeric(data[d]['open']).shift(n-1)
        data[d]['return_n'][data[d]['return_n'] >= 0] = 1
        data[d]['return_n'][data[d]['return_n'] < 0] = -1
        data[d]['n_neg'] = 0.0
        data[d]['n_pos'] = 0.0
        for l in range(0, len(data[d])):
            if l < (n-1):
                pass
            else:
                for i in range(0, n):
                    if data[d]['return'][l-i] < 0:
                        data[d]['n_neg'][l] += 1
                    else:
                        data[d]['n_pos'][l] += 1
        quality[d] = (float(data[d]['return_n'][-1:]))*((float(data[d]['n_neg'][-1:])/12)-(float(data[d]['n_pos'][-1:])/12))
    
    quality = dict(sorted(quality.items(), key=lambda x: x[1]))
    # quality = dict(list(quality.items())[:math.ceil(len(quality)*p)])
    quality = {k:v for k,v in quality.items() if v<=cut}

    series = {}
    for (key, value) in data.items():
        if key in quality.keys():
            series[key] = value
    print('quality series: ' + str(len(series)) + ' rows')
    return series


def markovitz(data):
    ####################################################################################################################################################################################
    # get adjusted closing prices of 5 selected companies with Quandl
    # quandl.ApiConfig.api_key = '3QMrpN426duegrHv6o4v'
    # selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA']
    # data = quandl.get_table('WIKI/PRICES', ticker = selected,
    #                         qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
    #                         date = { 'gte': '2014-1-1', 'lte': '2016-12-31' }, paginate=True)

    # reorganise data pulled by setting date as index with
    # columns of tickers and their corresponding adjusted prices
    # clean = data.set_index('date')
    # table = clean.pivot(columns='ticker')
    ####################################################################################################################################################################################
    
    selected = list(data.keys())
    table = pd.DataFrame(None)
    for ticker in selected:
        t = data[ticker][['open_datetime', 'close']]
        t.set_index('open_datetime', inplace=True)
        t.columns = [ticker]
        t[ticker] = pd.to_numeric(t[ticker])

        if len(table) == 0:
            table = t.copy()
        else:
            table = t.join(table)

    table.dropna(axis=1, inplace=True)

    # calculate daily and annual returns of the stocks
    returns_daily = table.pct_change()
    returns_annual = returns_daily.mean() * 250

    # get daily and covariance of returns of the stock
    cov_daily = returns_daily.cov()
    cov_annual = cov_daily * 250

    # empty lists to store returns, volatility and weights of imiginary portfolios
    port_returns = []
    port_volatility = []
    sharpe_ratio = []
    stock_weights = []

    # set the number of combinations for imaginary portfolios
    num_assets = len(table.columns)
    num_portfolios = 50000

    #set random seed for reproduction's sake
    np.random.seed(101)

    # populate the empty lists with each portfolios returns,risk and weights
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    # a dictionary for Returns and Risk values of each portfolio
    portfolio = {'Returns': port_returns,
                'Volatility': port_volatility,
                'Sharpe Ratio': sharpe_ratio}

    # extend original dictionary to accomodate each ticker and weight in the portfolio
    for counter,symbol in enumerate(table.columns):
        portfolio[symbol] = [Weight[counter] for Weight in stock_weights]

    # make a nice dataframe of the extended dictionary
    df = pd.DataFrame(portfolio)

    # get better labels for desired arrangement of columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock for stock in table.columns]

    # reorder dataframe columns
    df = df[column_order]

    # find min Volatility & max sharpe values in the dataframe (df)
    min_volatility = df['Volatility'].min()
    max_sharpe = df['Sharpe Ratio'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
    min_variance_port = df.loc[df['Volatility'] == min_volatility]

    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.show()
    plt.savefig('plots/Markowitz Efficient Frontier.png')

    print(min_variance_port.T)
    print(sharpe_portfolio.T)

    return min_variance_port, sharpe_portfolio


def run_strategy():
    # Get tickers + data from specific market:
    t = tickers_list(market='USDT')
    data = candlestick(tickers=t, limit=365, interval='1d', endTime = 1637140000000)

    # Screen nº 1:
    a = sample_space(data)

    # Screen nº 2:
    # b = beta(a)
    b = beta(data)

    # Screen nº 3:
    c = calculate_momentum(b)
    d, MomentumDataNoOutliers = momentum_outliers(b, c)

    # Screen nº 4:
    e = momentum_quantity(d, MomentumDataNoOutliers)

    # Screen nº5:
    f = momentum_quality(e)
    
    return f


def decile(data, filter_period, column):
    l = list()
    for k in data.keys():
        if type(data[k]) == type(pd.DataFrame()):
            value = data[k][column]
            l.append(sum(pd.to_numeric(value[-filter_period:]))/filter_period)
        else:
            value = data[k]
            l.append(pd.to_numeric(value))
    var = np.array(l)
    p_range = np.arange(10, 100, 10)
    percentile = np.percentile(var, p_range)
    print(percentile)

    return percentile


def real_time_pnl(portfolio_keys=['00']):
    for key in portfolio_keys:
        portfolio = pd.read_csv('portfolios/' + key + '.csv')

        AllTickers = tickers_list()
        # holding = list(balances().keys())
        holding = list(portfolio['asset'])

        tickers = []
        for i in AllTickers:
            for j in holding:
                if i[0:len(j)] == j and (i[-3:] == 'BRL' or i[-3:] == 'BTC'):
                    tickers.append(i)
        
        TradeHistory = trades(tickers)

        pnl = {}
        for s in TradeHistory:
            price = []
            qty = []
            Time = []
            for v in s:
                if type(s) == type(list()):
                    if v['isBuyer'] == True:
                        price.append(v['price'])
                        qty.append(v['qty'])
                        Time.append(v['time'])
                        pnl[v['symbol']] = {'price': price, 'qty': qty, 'time': Time,}
                elif type([s]) == type(dict()):
                    print('code: ' + str(s['code']))
        while True:
            Psell = prices()[1]
            conversion = Psell['BTCBRL']

            print('\n\n')
            print(str(datetime.datetime.now()) + ' | Portfolio ' + str(key))
            print('\n')

            CurrentSum = 0.0
            BegginingSum = 0.0

            for k in pnl.keys():
                # sum of quantities times current price
                # sum(pd.to_numeric(pnl['LRCBTC']['qty']))*pd.to_numeric(Psell['LRCBTC'])
                # sum of each buy price times each quantity
                # sum(pd.to_numeric(pnl['LRCBTC']['price'])*pd.to_numeric(pnl['LRCBTC']['qty']))
                pnl[k]['pnl_total'] = \
                    sum(pd.to_numeric(pnl[k]['qty']))*pd.to_numeric(Psell[k]) - \
                    sum(pd.to_numeric(pnl[k]['price'])*pd.to_numeric(pnl[k]['qty']))

                pnl[k]['pnl_percent'] = \
                    ((sum(pd.to_numeric(pnl[k]['qty']))*pd.to_numeric(Psell[k])/ \
                    sum(pd.to_numeric(pnl[k]['price'])*pd.to_numeric(pnl[k]['qty']))) - 1) * 100

                print(
                    k + ': ' + str(round(pnl[k]['pnl_percent'], 2)) + '%'
                )

                if k[-3:] == 'BTC':
                    t = pd.to_numeric(conversion)
                else:
                    t = 1.0
                
                CurrentSum += sum(pd.to_numeric(pnl[k]['qty']))*pd.to_numeric(Psell[k])*t
                BegginingSum += sum(pd.to_numeric(pnl[k]['price'])*pd.to_numeric(pnl[k]['qty']))*t
            
            print('\n')
            print('PnL total (BRL): ' + str(CurrentSum - BegginingSum))
            print('PnL percentage total: ' + str(((CurrentSum/BegginingSum) - 1) * 100))

            time.sleep(3)
    

def portfolio_delta(portfolio_keys=['00'], limit=7, interval='1d'):
    for key in portfolio_keys:
        portfolio = pd.read_csv('portfolios/' + key + '.csv')

        AllTickers = tickers_list()
        holding = list(portfolio['asset'])

        tickers = []
        for i in AllTickers:
            for j in holding:
                if i[0:len(j)] == j and (i[-3:] == 'BRL' or i[-3:] == 'BTC'):
                    tickers.append(i)
        
        TradeHistory = trades(tickers)

        pnl = {}
        for s in TradeHistory:
            price = []
            qty = []
            Time = []
            for v in s:
                if v['isBuyer'] == True:
                    price.append(v['price'])
                    qty.append(v['qty'])
                    Time.append(v['time'])
                    pnl[v['symbol']] = {'price': price, 'qty': qty, 'time': Time,}

        t = list(pnl.keys())
        t.append('BTCBRL')
        data = candlestick(tickers=t, limit=limit, interval=interval)

        # plot_candlestick(data=data)
        # PortfolioData = data


def get_portfolio(portfolio_keys=['00']):
    for key in portfolio_keys:
        # this portfolio should be saved as output from run_strategy
        # this should be done in a different way if buying assets through api
        # create exception for "CTSIBTC": {"price": ["0.00002303", "0.00001603"], "qty": ["140.00000000", "81.00000000"], "time": [1628535432451, 1637159419580], "portfolio": "00"} / first info is not from portfolio 00
        portfolio = pd.read_csv('portfolios/' + key + '.csv')

        AllTickers = tickers_list()
        holding = list(portfolio['asset'])

        conversionBNB = candlestick('BTCBNB')

        tickers = []
        for i in AllTickers:
            for j in holding:
                if i[0:len(j)] == j and (i[-3:] == 'BRL' or i[-3:] == 'BTC'):
                    tickers.append(i)
        
        TradeHistory = trades(tickers)

        # INSERT: check if asset is not in sheets file before assigning to current portfolio

        pnl = {}
        for s in TradeHistory:
            price = []
            qty = []
            Time = []
            orderId = []
            for v in s:
                if v['isBuyer'] == True:
                    price.append(v['price'])
                    qty.append(v['qty'])
                    Time.append(v['time'])
                    orderId.append(v['orderId'])
                    pnl[v['symbol']] = {
                        'price': price, 
                        'qty': qty, 
                        'time': Time, 
                        # 'orderId' : orderId, 
                        'portfolio' : key,
                    }

        update_value(worksheetName='Portfolios', data=json.loads(str(pnl).replace('\'', '\"')))
        # gsheets('update_cell', spreadsheetName='Cryptocurrencies Report Spreadsheet', worksheetNumber=0, cell='A1', data=json.loads(str(pnl).replace('\'', '\"')))


def gsheets(action, data=None, cell=None, spreadsheetName=None, spreadsheetKey=None, spreadsheetLink=None, worksheetNumber=None, worksheetName=None):
    gc = pygsheets.authorize(service_file='client_secret.json')

    if spreadsheetName is not None:
        # 1. Open spreadsheet by name 
        sh = gc.open(spreadsheetName) # open spreadsheet
    elif spreadsheetKey is not None:
        # 2. Open spreadsheet by key
        sh = gc.open_by_key(spreadsheetKey)
    elif spreadsheetLink is not None:
        # 3. Open spredhseet by link
        sh = gc.open_by_link(spreadsheetLink)
    else:
        print('No spreadsheet specified')

    # Open worksheet:
    if worksheetNumber is not None:
        wk = sh[worksheetNumber] # Open first worksheet of spreadsheet
    elif worksheetName is not None:
        wk = sh.worksheet[worksheetName] # sheet1 is name of first worksheet

    # Actions:
    if action == 'update_cell':
        wk.update_value(cell, data)
    elif action == 'get_values':
        values = wk.get_all_values(include_tailing_empty=False)
    elif action == 'create_worksheet':
        if worksheetName in str(sh.worksheets()):
            print('Worksheet ' + worksheetName + ' already exists')
        else:
            sh.add_worksheet(worksheetName,rows=250, cols=20)

    return values


def plot_candlestick(data, key='percentage', width = .4, width2 = .05, col1 = 'green', col2 = 'red'):
    for k in data.keys():
        # convert data to numeric
        for x in data[k].columns:
            data[k][x] = pd.to_numeric(data[k][x])

        # set datetime index (causing plot error)
        # data[k].set_index('open_datetime', inplace=True)
        # data[k].index = pd.to_datetime(data[k].index,unit='ms')

        #define up and down prices
        up = data[k][data[k]['close']>=data[k]['open']]
        down = data[k][data[k]['close']<data[k]['open']]

        if key == 'percentage':
            # change perspective to percentage
            base = data[k]['open'][0]
            for x in up.columns:
                up[x] = (up[x]-base)/base
            for x in down.columns:
                down[x] = (down[x]-base)/base
        else:
            pass

        #create figure
        plt.figure()

        #plot up prices
        plt.bar(up.index,up.close-up.open,width,bottom=up.open,color=col1)
        plt.bar(up.index,up.high-up.close,width2,bottom=up.close,color=col1)
        plt.bar(up.index,up.low-up.open,width2,bottom=up.open,color=col1)

        #plot down prices
        plt.bar(down.index,down.close-down.open,width,bottom=down.open,color=col2)
        plt.bar(down.index,down.high-down.open,width2,bottom=down.open,color=col2)
        plt.bar(down.index,down.low-down.close,width2,bottom=down.close,color=col2)

        # define table data
        TableDataUp = up.close-up.open
        TableDataDown = down.close-down.open
        TableData = TableDataUp.append(TableDataDown)
        TableData.sort_index(inplace=True)
        TableData = TableData*100
        ColLabels = list(TableData.index)
        TableData = list(TableData)

        # Add a table at the bottom of the axes
        plt.table(cellText=[['%.2f' % i for i in TableData]],
                            rowLabels=['Daily delta (%)'],
                            # rowColours=colors,
                            colLabels=ColLabels,
                            loc='bottom')

        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)

        #rotate x-axis tick labels
        plt.xticks(ticks=[], rotation=45, ha='right')

        # set title
        plt.title(k)

        #display candlestick chart
        plt.show()


def create_worksheet(worksheetName, spreadsheetName='Cryptocurrencies Report Spreadsheet'):
    gc = pygsheets.authorize(service_file='client_secret.json')
    sh = gc.open(spreadsheetName)
    if worksheetName in str(sh.worksheets()):
        print('Worksheet ' + worksheetName + ' already exists; not created')
    else:
        sh.add_worksheet(worksheetName,rows=250, cols=20)


def set_dataframe(worksheetName, data, cell='A1', copy_index=True, spreadsheetName='Cryptocurrencies Report Spreadsheet', clear=False):
    gc = pygsheets.authorize(service_file='client_secret.json')
    sh = gc.open(spreadsheetName)
    wk = sh.worksheet_by_title(worksheetName)
    if clear == True:
        wk.clear('A1')
    wk.set_dataframe(df=data, start=cell, copy_index=True)
    print('DataFrame set into ' + worksheetName + ' worksheet')


def update_value(worksheetName, data, cell='A1', spreadsheetName='Cryptocurrencies Report Spreadsheet'):
    gc = pygsheets.authorize(service_file='client_secret.json')
    sh = gc.open(spreadsheetName)
    wk = sh.worksheet_by_title(worksheetName)
    wk.update_value(val=data, addr=cell)
    print('Value updated in cell ' + cell + ', worksheet ' + worksheetName)


def get_values(worksheetName, spreadsheetName='Cryptocurrencies Report Spreadsheet'):
    gc = pygsheets.authorize(service_file='client_secret.json')
    sh = gc.open(spreadsheetName)
    wk = sh.worksheet_by_title(worksheetName)
    values = wk.get_all_values(include_tailing_empty=False)
    return values


def pnl(tradingFee=0.00075):
    # sellPrices = prices()[1]
    portfolios = get_values('Portfolios')
    pctChange = pd.DataFrame()
    priceSum = pd.DataFrame()
    for p in portfolios:
        if len(p) != 0:
            # convert portfolios from list of lists to dictionary 
            data = json.loads(p[0])
            create_worksheet('Portfolio ' + data[list(data)[0]]['portfolio'])
            counter = 0
            conversion = pd.DataFrame(candlestick('BTCBRL')['BTCBRL'])
            conversion['open_datetime'] = pd.to_datetime(conversion['open_datetime'], unit='ms')
            conversion.set_index('open_datetime', inplace=True)

            for k in data.keys():
                create_worksheet(k)
                a = candlestick(k)
                b = pd.DataFrame(a[k])

                # FIX! datetimefilter only works if candlestick begins after assets' buying datetime
                datetimeFilter = pd.to_datetime(data[k]['time'][0], unit='ms').round(freq='H') - datetime.timedelta(hours=1)
                b['open_datetime'] = pd.to_datetime(b['open_datetime'], unit='ms')
                b['close_datetime'] = pd.to_datetime(b['close_datetime'], unit='ms')
                b = b[b['open_datetime'] >= datetimeFilter]

                buyPrice = sum(pd.to_numeric(data[k]['price'])*pd.to_numeric(data[k]['qty']))/sum(pd.to_numeric(data[k]['qty']))
                buyPrice = buyPrice*(1+tradingFee)
                
                b['close'].iloc[0] = buyPrice.copy()
                b['open'].iloc[0] = buyPrice.copy()
                b['open'] = pd.to_numeric(b['open']).copy()
                b['close'] = pd.to_numeric(b['close']).copy()

                if counter == 0:
                    pctChange.index = a[k]['open_datetime']
                    priceSum.index = a[k]['open_datetime']
                    counter += 1
                pctChangeIntermediate = pd.DataFrame()
                priceSumIntermediate = pd.DataFrame()

                pctChangeIntermediate[k] = (b['close'] - b['close'].iloc[0])/b['close'].iloc[0]
                pctChangeIntermediate.index = b['open_datetime'].copy()
                priceSumIntermediate[k] = b['close'].copy()
                priceSumIntermediate[k] = priceSumIntermediate[k]*sum(pd.to_numeric(data[k]['qty']))
                priceSumIntermediate.index = b['open_datetime'].copy()
                conversionFiltered = conversion['close'][conversion.index >= priceSumIntermediate.index.min()]
                
                # convert BTC values to BRL
                if k[-3:] == 'BTC':
                    conversionFiltered = pd.DataFrame(pd.to_numeric(conversion['close'])[conversion.index >= priceSumIntermediate.index.min()])
                    priceSumIntermediate = pd.DataFrame(priceSumIntermediate[k].multiply(conversionFiltered['close']), columns=[k], index=priceSumIntermediate.index)

                pctChange = pctChange.merge(pctChangeIntermediate, how='left', left_index=True, right_index=True)
                priceSum = priceSum.merge(priceSumIntermediate, how='left', left_index=True, right_index=True)

                set_dataframe(worksheetName=k, data=b, clear=True)
            
            pctChange.dropna(how='all', inplace=True)

            # FIX!! if all assets don't start at same date then pnl for portfolio will be an aproximate because of argument "how='any'"
            priceSum.dropna(how='any', inplace=True)
            priceSum.fillna(0.0, inplace=True)
            priceSum = priceSum.sum(axis=1)

            priceSum = pd.DataFrame(priceSum)
            priceSum['pct'] = (priceSum[0] - priceSum[0][0])/priceSum[0][0]
            priceSum.columns = ['price', 'pct_change']

            pctChange = pctChange.merge(priceSum, how='left', left_index=True, right_index=True)

            set_dataframe(worksheetName='Portfolio ' + data[list(data)[0]]['portfolio'], data=pctChange, clear=True)
            update_value(worksheetName='Portfolio ' + data[list(data)[0]]['portfolio'], data='Datetime', cell='A1')

                # HOW:
                    #1. get candlestick data for all assets in portfolio ok
                    #2. filter candlestick data for datetimes greater than asset buying datetime
                    #3. make asset buying as first row in candlestick data
                    #4. convert to BRL quote asset
                


# build_portfolio variables:
#   [0]: build new portfolio
#   [1]: calculate Markowitz for new portfolio
build_portfolio = [1, 0]
RealTime = 0
if __name__ == "__main__":
    
    if build_portfolio[0] == 1:
        f = run_strategy()
        print(f)
        if build_portfolio[1] == 1:
            g, h = markovitz(f)
    
    if RealTime == 1:
        real_time_pnl()
    elif RealTime == 2:
        portfolio_delta()

    # get_portfolio()
    pnl()


    # analyzing deciles

    # next steps:

        # Paste current portfolio data in Sheets
            # Get pnl views in sheets
                # for each asset separately
                # for all investments
                # for each portfolio

        # check date when portfolio was bought to know when to sell

        # adjust real_time_pnl and portfolio_delta to contemplate many portfolios at once
            # finish function get_portfolio()
