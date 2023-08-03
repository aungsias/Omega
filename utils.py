#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
import requests
import datetime as dt
import pytz
import scipy as sp
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import random
import os
import threading
import time
import warnings


from datetime import datetime
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.api import OLS as ols
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from itertools import combinations
from cycler import cycler
from matplotlib import cm

def fmt(x, metric):
    return f"{x:,.2f}" if metric == "dcm" else f"{100*x:,.2f}%" if metric == 'pct' else f"${x:,.2f}" if metric == 'dlr' else None

def set_color_cycler(cmap_name='Set1', num_colors=12, linewidth=.75):
    cmap = cm.get_cmap(cmap_name, num_colors)
    color_cycler = cycler(color=[cmap(i) for i in range(num_colors)])
    plt.rcParams["axes.prop_cycle"] = color_cycler
    mpl.rcParams['figure.figsize'] = (14, 4)
    mpl.rcParams['lines.linewidth'] = linewidth
    sns.set_style('whitegrid')
    warnings.filterwarnings('ignore')
    return [cmap(i) for i in range(num_colors)]

universe = {
    'Technology': {
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'MET', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'ADBE', 'IBM', 'ORCL', 'CRM'],
        'indices': ['XLK', 'VGT', 'FDN']
    },
    'Financial': {
        'tickers': ['JPM', 'BAC', 'V', 'MA', 'GS', 'MS', 'AXP', 'WFC', 'C', 'USB', 'BLK', 'PNC', 'SCHW'],
        'indices': ['XLF', 'VFH', 'KBE']
    },
    'Healthcare': {
        'tickers': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABT', 'MDT', 'LLY', 'AMGN', 'GILD', 'VRTX', 'BMY', 'SYK', 'ISRG'],
        'indices': ['XLV', 'VHT', 'IBB']
    },
    'Consumer': {
        'tickers': ['WMT', 'KO', 'PEP', 'PG', 'MCD', 'NKE', 'SBUX', 'MMM', 'CL', 'KMB', 'DIS', 'YUM', 'TGT'],
        'indices': ['XLY', 'XLP', 'VCR']
    },
    'Energy': {
        'tickers': ['XOM', 'CVX', 'COP', 'EOG', 'PSX', 'MPC', 'VLO', 'OXY', 'HES', 'PXD', 'SLB', 'KMI', 'WMB'],
        'indices': ['XLE', 'VDE', 'FENY']
    },
    'Industrials': {
        'tickers': ['GE', 'HON', 'UNP', 'MMM', 'BA', 'CAT', 'DE', 'CSX', 'EMR', 'ETN', 'LMT', 'FDX', 'ITW'],
        'indices': ['XLI', 'VIS', 'ITA']
    },
    'Utilities': {
        'tickers': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'XEL', 'WEC', 'PEG', 'FE', 'ES', 'ED'],
        'indices': ['XLU', 'VPU', 'IDU']
    },
    'Real Estate': {
        'tickers': ['SPG', 'PLD', 'EQIX', 'EQR', 'AVB', 'PSA', 'VTR', 'WELL', 'CCI', 'DLR', 'BXP', 'CBRE', 'O'],
        'indices': ['XLRE', 'VNQ', 'IYR']
    },
    'Materials': {
        'tickers': ['LIN', 'APD', 'ECL', 'PPG', 'NEM', 'IFF', 'SHW', 'LYB', 'ALB', 'CE', 'VMC', 'IP', 'CF'],
        'indices': ['XLB', 'VAW', 'RTM']
    }
}



key_periods = {
    'PostGFC-Recovery': {
        'start': '2010-01-01', 
        'end': '2012-01-01'
    },
    'TaperTantrum-OilCollapse': {
        'start': '2012-01-01', 
        'end': '2014-01-01'
    },
    'BullMarket-LowVolatility': {
        'start': '2013-01-01', 
        'end': '2015-01-01'
    },
    'TradeWar-IncreasedVolatility': {
        'start': '2014-01-01', 
        'end': '2016-01-01'
    },
    'COVID-19Pandemic': {
        'start': '2015-01-01', 
        'end': '2017-01-01'
    },
    'PostPandemic-Recovery': {
        'start': '2016-01-01', 
        'end': '2018-01-01'
    }
}

def call_universe(universe=universe):
    ticker_data = {sector: universe[sector]['tickers'] + universe[sector]['indices']  for sector in universe}
    df = pd.DataFrame.from_dict(ticker_data, orient='index').T
    df.index.name = '#'
    df.index += 1
    return df

def call_universe_tickers(universe=universe):
    ticker_data = {sector: universe[sector]['tickers'] for sector in universe}
    df = pd.DataFrame.from_dict(ticker_data, orient='index').T
    df.index.name = '#'
    df.index += 1
    return df

def get_tickers():
    ticker_list = universe.copy()
    tickers = []
    for sector in ticker_list:
        tickers.extend(ticker_list[sector]['tickers'])
    return tickers

def random_from_universe(sector, choices=4):
    tickers = random.sample(universe[sector]['tickers'], choices)
    index = random.choice(universe[sector]['indices'])
    return tickers, index

def get_api():
    package_directory = os.path.dirname(os.path.abspath(__file__))
    api_file_path = os.path.join(package_directory, "api_fmp.txt")

    with open(api_file_path, "r") as f:
        API_KEY = f.read().strip()
    return API_KEY

API_KEY = get_api()

def get_now():
    return dt.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')

def get_today():
    return dt.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')

def get_pricing(stock_symbol, API_KEY=API_KEY, start=None, end=None, period=None, column='Close', yahoo=True):

    max_retries = 5
    base_url = "https://financialmodelingprep.com/api/v3/historical-price-full/"

    for retry in range(max_retries):
        try:
            if not yahoo:
                if period == 'max':
                    api_url = f"{base_url}{stock_symbol}?apikey={API_KEY}&serietype=line"
                else:
                    api_url = f"{base_url}{stock_symbol}?from={start}&to={end}&apikey={API_KEY}"
            
                response = requests.get(api_url)
                response.raise_for_status()  # Will raise an exception if the status is 4xx or 5xx

                df = pd.DataFrame(response.json()['historical'])
                df.rename(columns={'date':'Date','close':'Close','open':'Open','high':'High','low':'Low'}, inplace=True)
                df.set_index('Date', inplace=True)
                df.index = pd.to_datetime(df.index)
                if column is not None:
                    return df[column].iloc[::-1]
                else:
                    df.iloc[::-1]
            
            elif yahoo:
                prices = yf.Ticker(stock_symbol).history(start=start, end=end, period=period)
                prices.index = prices.index.tz_localize(None)
                return prices['Close']

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
            if retry < max_retries - 1:
                wait = 2 ** retry  # exponential backoff
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
                continue
            else:
                print("Max retries reached. Exiting...")
                return None
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting: {errc}")
            return None
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
            return None
        except requests.exceptions.RequestException as err:
            print(f"Something went wrong: {err}")
            return None

def get_returns(prices):
    return pd.DataFrame(np.log(prices).diff()).rename(columns={'Close': 'returns'})
    
def minute_data(ticker, precise=True, API_KEY=API_KEY, raw=False):
    if precise:
        url = f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={API_KEY}'
        response = requests.get(url)
        rawdata = response.json()
        df = pd.json_normalize(rawdata)
        df = df.iloc[::-1]
        df = df.set_index(['date'])
    if not precise:
        df = yf.Ticker('AAPL').history(period='7d', interval='1m')
        df.index.name = 'Datetime'
        df = df.rename_axis('date')
        df = df.rename(columns={col: col.lower() for col in df.columns})
    return df if not raw else rawdata


# In[46]:


def dcf(ticker, API_KEY=API_KEY, short=True, raw=False):
    url = f'https://financialmodelingprep.com/api/v4/advanced_discounted_cash_flow?symbol={ticker}&apikey={API_KEY}'
    response = requests.get(url)
    rawdata = response.json()
    df = pd.json_normalize(rawdata)
    df['enterpriseValuePerShare'] = df.enterpriseValue / df.dilutedSharesOutstanding
    now = dt.datetime.now().strftime('%Y')
    df['state'] = 'Undervalued'
    df.loc[df.price > df.enterpriseValuePerShare * 1.05, 'state'] = 'Overvalued'
    df.loc[(df.price > df.enterpriseValuePerShare * .95) & (
                df.price < df.enterpriseValuePerShare * 1.05), 'state'] = 'Neutral'
    extdf = df[['year', 'symbol', 'enterpriseValuePerShare', 'equityValuePerShare',
                'price', 'state', 'beta', 'ebitda', 'ebit', 'revenue',
                'netDebt', 'equityValue', 'enterpriseValue',
                'dilutedSharesOutstanding', 'capitalExpenditurePercentage']]
    shtdf = df[['year', 'symbol', 'enterpriseValuePerShare', 'price', 'state', 'beta']]
    shtdf = shtdf.set_index('year')
    extdf = extdf.set_index('year')
    currsht = shtdf.loc[[now]]
    currext = extdf.loc[[now]]
    if not raw:
        return currsht if short else currext
    if raw:
        return rawdata


# In[47]:


def fair_value(ticker, API=API_KEY, equity=False, raw=False):
    if not raw:
        data = dcf(ticker, short=False)
        fair_value_equity = data['equityValuePerShare'][0]
        fair_value_enterprise = data['enterpriseValuePerShare'][0]
        return fair_value_enterprise if not equity else fair_value_equity
    if raw:
        data = dcf(ticker, short=False, raw=raw)
        return data


# In[48]:


def current_price(ticker, API_KEY=API_KEY, raw=False, date=False):
    try:
        url = f'https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={API_KEY}'
        response = requests.get(url)
        response.raise_for_status()
        rawdata = response.json()
        if not date:
            price = rawdata[0]['price']
        if date:
            price = pd.json_normalize(rawdata)
    except IndexError:
        pass
    except KeyError:
        pass
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.JSONDecodeError:
        pass
    return price if not raw else rawdata

def regime_change(ticker, bp, start, end):

    Y = get_pricing(ticker, start=start, end=end)['Close']
    y1 = Y[:bp]
    y2 = Y[bp:]

    X = np.arange(len(Y))
    x1 = np.arange(bp)
    x2 = np.arange(len(Y)-bp)

    A, B = ols(Y, X)
    a1, b1 = ols(y1, x1)
    a2, b2 = ols(y2, x2)

    Y_HAT = pd.Series(A + B*X, index=Y.index)
    y_hat1 = pd.Series(a1 + b1*x1, index=y1.index)
    y_hat2 = pd.Series(a2 + b2*x2, index=y2.index)

    Y.plot()
    Y_HAT.plot(color='orange', linewidth=1.5)
    y_hat1.plot(color='g', linewidth=1.5)
    y_hat2.plot(color='g', linewidth=1.5)
    plt.show()

def mpl_cast(style):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    style_file_path = os.path.join(package_directory, f"{style}.mplstyle")

    if os.path.exists(style_file_path):
        mpl.rc_file(style_file_path)
    else:
        raise ValueError(f"Style '{style}' not found in the 'Omega' package.")

def residual_analysis(resids, exog, sig_level = 0.01):

    normality_test = jarque_bera(resids)
    hskd_test = het_breuschpagan(resids, exog)
    autocorr_test = durbin_watson(resids)

    vals = [normality_test[1], hskd_test[-1], autocorr_test]


    if vals[0] < sig_level:
        print('Residuals likely not normal.')
    else:
        print('Residuals likely normal.')
    if vals[1] < sig_level:
        print('Residuals likely heteroscedastic.')
    else:
        print('Residuals likely not heteroscedastic.')
    if vals[2] < 2:
        print('Residuals likely autocorrelated')
    else:
        print('Residuals are likely not autocorrelated')

def chow(ticker, period='max', start=None, end=None, cut=2, sig_lvl=0.05, verbose=True, plot=False):

    returns = (1 + get_pricing(ticker, period=period, start=start, end=end)['Close'].pct_change()[1:]).cumprod()
    
    bp = len(returns)//cut if cut == 2 else cut
        
    H1 = returns[:bp]
    H2 = returns[bp:]

    X = np.arange(len(returns))
    XH1 = np.arange(bp)
    XH2 = np.arange(len(returns)-bp)

    LR = sm.OLS(returns, sm.add_constant(X)).fit()
    LRH1 = sm.OLS(H1, sm.add_constant(XH1)).fit()
    LRH2 = sm.OLS(H2, sm.add_constant(XH2)).fit()

    RSD = LR.resid
    RSDH1 = LRH1.resid
    RSDH2 = LRH2.resid

    SSRSD = np.sum(np.power(RSD, 2))
    SSRSDH1 = np.sum(np.power(RSDH1, 2))
    SSRSDH2 = np.sum(np.power(RSDH2, 2))

    N = LR.nobs
    K = 2 # total number of estimated parameters = total number of independent variables + 1 (1 is for the constant)

    numerator = (SSRSD - (SSRSDH1 + SSRSDH2))/K
    denominator = (SSRSDH1 + SSRSDH2)/(N - 2*K)

    chow_statistic = numerator/denominator

    dfn  = K
    dfd = N-2*K
    interval = [0 + sig_lvl/2, 1 - sig_lvl/2]

    critical_value = sp.stats.f(dfn=dfn, dfd=dfd).ppf(interval[1])

    if verbose:

        if chow_statistic > critical_value:
            print('Reject null: There is a structural break in the dataset.')
        if chow_statistic < critical_value:
            print('Keep null: There is no structural break in the dataset.')

    if plot:

        A = LR.params[0]
        AH1 = LRH1.params[0]
        AH2 = LRH2.params[0]

        B = LR.params[1]
        BH1 = LRH1.params[1]
        BH2 = LRH2.params[1]

        YHAT = A + B*X
        YHATH1 = AH1 + BH1*XH1
        YHATH2 = AH2 + BH2*XH2

        YHAT = pd.Series(YHAT, index=returns.index)
        YHATH1 = pd.Series(YHATH1, index=H1.index)
        YHATH2 = pd.Series(YHATH2, index=H2.index)

        returns.plot()
        YHAT.plot(color='red', linewidth=1.5)
        YHATH1.plot(color='g', linewidth=1.5)
        YHATH2.plot(color='g', linewidth=1.5)
        plt.show()

    return chow_statistic, critical_value

# FOR PAIRS TRADING IPYNB

def get_price_data(ticker, start, training_end, testing_end, training_prices, testing_prices, lock):
    data = get_pricing(ticker, start=start, end=testing_end)
    in_sample = data.loc[start:training_end]
    out_sample = data.loc[training_end:testing_end]
    with lock:
        training_prices[ticker] = in_sample
        testing_prices[ticker] = out_sample

def fetch_and_split_price_data(tickers, start, training_end, testing_end):
    training_prices = {}
    testing_prices = {}
    lock = threading.Lock()
    threads = []
    for ticker in tickers:
        t = threading.Thread(target=get_price_data, args=(ticker, start, training_end, testing_end, training_prices, testing_prices, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if set(training_prices.keys()) == set(tickers) and set(testing_prices.keys()) == set(tickers):
        return training_prices, testing_prices
    else:
        raise ValueError('Not all tickers were retrieved.')


# Finds the most cointegrated pairs from a list of tickers
def single_pair_cointegration(pair, price_data, sig_level):
    ticker1, ticker2 = pair
    S1 = price_data[ticker1]
    S2 = price_data[ticker2]
    if len(S1.index) == len(S2.index):
        t, p, c = coint(S1, S2)
    else:
        return None
    if p < sig_level:
        return pair
    else:
        return None

def find_cointegrated_pairs(price_data, sig_lvl=0.01, max_workers=90):
    cointegrated_pairs = []
    lock = threading.Lock()

    ticker_combs = list(combinations(list(price_data.keys()), r=2))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_pair_cointegration, pair, price_data, sig_lvl) for pair in ticker_combs]
        for future in wait(futures).done:
            pair = future.result()
            with lock:
                if pair is not None:
                    cointegrated_pairs.append(pair)

    # Sort the cointegrated pairs based on the p-value
    cointegrated_pairs = sorted(cointegrated_pairs, key=lambda x: x[1])

    return cointegrated_pairs


# Checks for stationarity based on the ADF test
def check_stationarity(x, c=0.01):
    test = adfuller(x)
    if test[1] < c:
        return f'{x.name} is likely stationary'
    else:
        return f'{x.name} is likely not stationary'
    
# Computes percent profits for trades
def percent_profits(entry_prices, exit_prices, short=False):
    if short:
        return [((purchase) - (repurchase)) 
                / purchase for purchase, repurchase in zip(entry_prices, exit_prices)]
    return [((sale) - (purchase))
            / purchase for purchase, sale in zip(entry_prices, exit_prices)]

# Compares returns from trading to a specified benchmark
def calculate_return(start_price, end_price):
    return end_price / start_price - 1

def long_y_short_x(trade_start, trade_end, X, Y):
    Y_BEG_P = Y.loc[trade_start]
    Y_END_P = Y.loc[trade_end]
    X_BEG_P = X.loc[trade_start]
    X_END_P = X.loc[trade_end]
    MAXLONG_Y_R = calculate_return(Y_BEG_P, Y_END_P)
    MAXSHORT_X_R = calculate_return(X_END_P, X_BEG_P)
    return MAXLONG_Y_R + MAXSHORT_X_R

def buy_hold_both(trade_start, trade_end, X, Y):
    Y_BEG_P = Y.loc[trade_start]
    Y_END_P = Y.loc[trade_end]
    X_BEG_P = X.loc[trade_start]
    X_END_P = X.loc[trade_end]
    MAXLONG_Y_R = calculate_return(Y_BEG_P, Y_END_P)
    MAXLONG_X_R = calculate_return(X_BEG_P, X_END_P)
    return MAXLONG_Y_R + MAXLONG_X_R

def sp_500(trade_start, trade_end):
    market_prices = get_pricing('SPY', start=trade_start, end=trade_end)
    MKT_START_P = market_prices.iloc[0]
    MKT_END_P = market_prices.iloc[-1]
    return calculate_return(MKT_START_P, MKT_END_P)

def benchmark_return(trade_start, trade_end, X, Y, benchmark='long y short x'):
    benchmark = benchmark.lower()
    if benchmark == 'long y short x':
        return long_y_short_x(trade_start, trade_end, X, Y)
    elif benchmark == 'buy hold both':
        return buy_hold_both(trade_start, trade_end, X, Y)
    elif benchmark == 's&p 500':
        return sp_500(trade_start, trade_end)

def risk_free(start, end):
    return (np.mean(get_pricing('^TNX', start=start, end=end))/100)/252

def equal_lengths(entx, extx, enty, exty):
    entx, extx = entx[:len(extx)], extx[:len(entx)]
    enty, exty = enty[:len(exty)], exty[:len(enty)]
    return entx, extx, enty, exty

def get_holding_period(ent, ext):
    if ent =='NaN' or ext =='NaN':
        return 'NaN'
    return (datetime.strptime(ext, '%Y-%m-%d') - datetime.strptime(ent, '%Y-%m-%d')).days

def calculate_hold_period(df, risk_free, trade_type, Y, X):
    for t in [f'{Y.name}', f'{X.name}']:
        df[f'{trade_type}_{t}_holdper'] = df.apply(
            lambda row: get_holding_period(
                row[f'{trade_type}_{t}_ent'], row[f'{trade_type}_{t}_exit']), axis=1
        )
        df[f'{trade_type}_{t}_rf'] = df.apply(
            lambda row: (1 + risk_free)**row[f'{trade_type}_{t}_holdper'] - 1, axis=1
        )
    return df

def calculate_excess_return(df, trade_type, Y, X):
    for t in [f'{Y.name}', f'{X.name}']:
        df[f'{trade_type}_{t}_xsr'] = df.apply(
            lambda row: row[f'{trade_type}_{t}_rets'] - row[f'{trade_type}_{t}_rf'], axis=1
        )
    return df

def calculate_sharpe(df, trade_type, Y, X):
    sharpe = []
    for t in [f'{Y.name}', f'{X.name}']:
        df.drop(columns=[f'{trade_type}_{t}_holdper'], inplace=True)
        avg_xsr = np.mean(df[f'{trade_type}_{t}_xsr'])
        std_xsr = np.std(df[f'{trade_type}_{t}_xsr'])
        sharpe_ratio = (avg_xsr/std_xsr)
        sharpe.append(sharpe_ratio)
    return np.mean(sharpe)

def trades_stats(enty, entx, exty, extx, yrs, xrs, risk_free, Y, X, type='short'):
    entx, extx, enty, exty = equal_lengths(entx, extx, enty, exty)

    data = {
        f'{type}_{Y.name}_ent': enty.index,
        f'{type}_{Y.name}_exit': exty.index,
        f'{type}_{Y.name}_rets': yrs,
        f'{type}_{X.name}_ent': entx.index,
        f'{type}_{X.name}_exit': extx.index,
        f'{type}_{X.name}_rets': xrs,
    }

    df = pd.DataFrame.from_dict(data, orient='index').T.dropna()

    df = calculate_hold_period(df, risk_free, type, Y, X)
    df = calculate_excess_return(df, type, Y, X)
    sharpe = calculate_sharpe(df, type, Y, X)

    df.index.name = f'{Y.name}-{X.name}'
    df.index += 1

    return df, sharpe

# Defining our trading logic as a function
def pairs_trade(Z, K=1):
    # Position tracker variable
    position = None
    # Trade dates
    long_dates = []
    short_dates = []
    exit_long_dates = []
    exit_short_dates = []   
    # Iterate through the rows
    for i in range(len(Z)):
        # Variables to make logic more readable
        z_score = Z[i]
        date = Z.index[i]
        # If not in position and the row's price is greater than / equal to the row's upper band value, enter short
        if position is None and z_score > K:
            position = 'short'
            short_dates.append(date)
         # If not in position and the row's price is less than / equal to the row's lower band value, enter long
        if position is None and z_score < -K:
            position = 'long'
            long_dates.append(date)
        # If currently in a short position and price hits or moves below our exit threshold for shorts, exit short
        if position == 'short' and z_score <= 0:
            position = None
            exit_short_dates.append(date)
        # If currently in a long position and price hits or moves below our exit threshold for longs, exit long
        if position == 'long' and z_score >= 0:
            position = None
            exit_long_dates.append(date)
    # Return the corresponding dates, which will then be used to find the corresponding prices
    return long_dates, short_dates, exit_long_dates, exit_short_dates

def process_pair(pair, out_sample_prices, window=40, ma_window=40, K=1):
    y = pair[0]
    x = pair[1]
    Y = out_sample_prices[y]
    X = out_sample_prices[x]
    Y.name = y
    X.name = x

    rolling_regression = RollingOLS(Y, sm.add_constant(X), window=window).fit()

    params = rolling_regression.params.drop(columns='const')
    params.rename(columns={f'{x}': f'B_{x}'}, inplace=True)
    B = params[f'B_{x}']

    C = Y - B*X

    C_MA = C.rolling(window=ma_window).mean()
    C_STD = C.rolling(window=ma_window).std()

    Z = (C - C_MA) / C_STD

    ENTLONG_D, ENTSHORT_D, EXTLONG_D, EXTSHORT_D = pairs_trade(Z, K=K)

    # Retrieving prices from longing and shorting Y and the corresponding exit dates
    ENTLONGY_P, EXTLONGY_P = Y.loc[ENTLONG_D], Y.loc[EXTLONG_D]
    ENTSHORTY_P, EXTSHORTY_P = Y.loc[ENTSHORT_D], Y.loc[EXTSHORT_D]

    # Retrieving prices from longing and shorting X and the corresponding exit dates
    ENTLONGX_P, EXTLONGX_P = X.loc[ENTSHORT_D], X.loc[EXTSHORT_D]
    ENTSHORTX_P, EXTSHORTX_P = X.loc[ENTLONG_D], X.loc[EXTLONG_D]

    # Computing percent profit per trade
    LONGY_RS, SHORTY_RS = percent_profits(ENTLONGY_P, EXTLONGY_P), percent_profits(ENTSHORTY_P, EXTSHORTY_P)
    LONGX_RS, SHORTX_RS = percent_profits(ENTLONGX_P, EXTLONGX_P), percent_profits(ENTSHORTX_P, EXTSHORTX_P)

    # Profits for Y and X trades
    Y_R = np.sum(LONGY_RS) + np.sum(SHORTY_RS)
    X_R = np.sum(LONGX_RS) + np.sum(SHORTX_RS)

    # Grand total profits
    TOT_R = Y_R + X_R
    start_of_trading, end_of_trading = C_MA.first_valid_index(), C_MA.index.max() 
    risk_free_rate = risk_free(start_of_trading, end_of_trading)

    long, long_sharpe = trades_stats(
        ENTLONGY_P, ENTLONGX_P, EXTLONGY_P, EXTLONGX_P, LONGY_RS, LONGX_RS, risk_free_rate, type='long', Y=Y, X=X
    )

    short, short_sharpe = trades_stats(
        ENTSHORTY_P, ENTSHORTX_P, EXTSHORTY_P, EXTSHORTX_P, SHORTY_RS, SHORTX_RS, risk_free_rate, type='short', Y=Y, X=X
    )

    pairs_trade_sharpe = np.mean(long_sharpe + short_sharpe)    
    return (pair, TOT_R, long, short, pairs_trade_sharpe)

def pairs_trade_basket(cointegrated_pairs, out_sample_prices, window=40, ma_window=40, K=1, max_workers=90):

    TOTAL_RETURNS = {}
    TRADES_STATS = {}
    SHARPES = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pair, pair, out_sample_prices, window, ma_window, K) for pair in cointegrated_pairs if len(out_sample_prices[pair[0]]) == len(out_sample_prices[pair[1]])]
        for future in wait(futures).done:
            pair, TOT_R, long, short, pairs_trade_sharpe = future.result()
            y, x = pair
            with lock:
                TOTAL_RETURNS[y, x] = TOT_R
                TRADES_STATS[y, x] = {'long': long, 'short': short}
                SHARPES[y, x] = pairs_trade_sharpe

    return TOTAL_RETURNS, TRADES_STATS, SHARPES

def pairs_trade_en_masse(cointegrated_pairs, out_sample_prices, window=40, ma_window=40, K=1):

    TOTAL_RETURNS = {}
    TRADES_STATS = {}
    SHARPES = {}

    for pair in cointegrated_pairs:
        
        y = pair[0][0]
        x = pair[0][1]
        Y = out_sample_prices[y]
        X = out_sample_prices[x]
        Y.name = y
        X.name = x
        
        rolling_regression = RollingOLS(Y, sm.add_constant(X), window=window).fit()

        params = rolling_regression.params.drop(columns='const')
        params.rename(columns={f'{x}': f'B_{x}'}, inplace=True)
        B = params[f'B_{x}']

        C = Y - B*X

        C_MA = C.rolling(window=ma_window).mean()
        C_STD = C.rolling(window=ma_window).std()

        Z = (C - C_MA) / C_STD
        
        ENTLONG_D, ENTSHORT_D, EXTLONG_D, EXTSHORT_D = pairs_trade(Z, K=K)

        ENTLONGY_P, EXTLONGY_P = Y.loc[ENTLONG_D], Y.loc[EXTLONG_D]
        ENTSHORTY_P, EXTSHORTY_P = Y.loc[ENTSHORT_D], Y.loc[EXTSHORT_D]
        ENTLONGX_P, EXTLONGX_P = X.loc[ENTSHORT_D], X.loc[EXTSHORT_D]
        ENTSHORTX_P, EXTSHORTX_P = X.loc[ENTLONG_D], X.loc[EXTLONG_D]

        LONGY_RS, SHORTY_RS = percent_profits(ENTLONGY_P, EXTLONGY_P), percent_profits(ENTSHORTY_P, EXTSHORTY_P)
        LONGX_RS, SHORTX_RS = percent_profits(ENTLONGX_P, EXTLONGX_P), percent_profits(ENTSHORTX_P, EXTSHORTX_P)

        Y_R = np.sum(LONGY_RS) + np.sum(SHORTY_RS)
        X_R = np.sum(LONGX_RS) + np.sum(SHORTX_RS)

        TOT_R = Y_R + X_R

        start_of_trading, end_of_trading = C_MA.first_valid_index(), C_MA.index.max() 
        risk_free_rate = risk_free(start_of_trading, end_of_trading)
        long, long_sharpe = trades_stats(
            ENTLONGY_P, ENTLONGX_P, EXTLONGY_P, EXTLONGX_P, LONGY_RS, LONGX_RS, risk_free_rate, type='long', Y=Y, X=X
        )
        short, short_sharpe = trades_stats(
            ENTSHORTY_P, ENTSHORTX_P, EXTSHORTY_P, EXTSHORTX_P, SHORTY_RS, SHORTX_RS, risk_free_rate, type='short', Y=Y, X=X
        )
        pairs_trade_sharpe = np.mean(long_sharpe + short_sharpe)

        TOTAL_RETURNS[y,x] = TOT_R
        TRADES_STATS[y,x] = {'long': long, 'short': short}
        SHARPES[y,x] = pairs_trade_sharpe

    return TOTAL_RETURNS, TRADES_STATS, SHARPES

def basket_bar_chart(data, title, ylabel, xlabel='Pairs', color='r'):
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))
    pairs = [f'{pair}' for pair in sorted_data.keys()]
    values = list([value for value in sorted_data.values()])
    sns.barplot(x=pairs, y=values, orient='v', color=color, width=1, edgecolor='black', alpha=.5)
    plt.grid(False)
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
    plt.title(f'{title}')
    plt.tight_layout()
    plt.xticks([]);

def basket_histogram(data, title, xlabel, color='r'):
    sns.histplot(data, bins=20, color=color, kde=True,alpha=.5, linewidth=1)
    plt.xlabel(f'{xlabel}')
    plt.title(f'{title}')
    plt.tight_layout()
    plt.grid(False)

def summarize_basket_trades(basket_sharpes, basket_returns):
    max_sharpe_key = max(basket_sharpes, key=basket_sharpes.get)
    min_sharpe_key = min(basket_sharpes, key=basket_sharpes.get)
    max_return_key = max(basket_returns, key=basket_returns.get)
    min_return_key = min(basket_returns, key=basket_returns.get)
    
    sharpe_values = list(basket_sharpes.values())
    return_values = list(basket_returns.values())
    
    data = {
        "Pair": [max_sharpe_key, max_return_key, min_sharpe_key, min_return_key, "Mean", "Median"],
        "Sharpe Ratio": [
            basket_sharpes[max_sharpe_key],
            basket_sharpes[max_return_key],
            basket_sharpes[min_sharpe_key],
            basket_sharpes[min_return_key],
            np.mean(sharpe_values),
            np.median(sharpe_values),
        ],
        "% Return": [
            basket_returns[max_sharpe_key]*100,
            basket_returns[max_return_key]*100,
            basket_returns[min_sharpe_key]*100,
            basket_returns[min_return_key]*100,
            np.mean(return_values)*100,
            np.median(return_values)*100,
        ],
        "Rank": ["Highest Sharpe", "Highest Return", "Lowest Sharpe", "Lowest Return", "Mean", "Median"],
    }

    df = pd.DataFrame.from_dict(data).set_index('Pair')
    return df


def fetch_price_data(prices, ticker, period, start, end, lock):
    data = get_pricing(ticker, start=start, end=end)
    with lock:
        prices[period] = data
def get_multiple_price_data(ticker, periods):
    price_data = {}
    lock = threading.Lock()
    threads = []
    for period, dates in periods.items():
        t = threading.Thread(target=fetch_price_data, args=(price_data, ticker, period, dates['start'], dates['end'], lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return ticker, price_data
def fetch_and_store_multiple_price_data(ticker, periods, prices, lock):
    _, price_data = get_multiple_price_data(ticker, periods)
    with lock:
        for period in periods:
            prices[period][ticker] = price_data[period]
def ccf_price_by_period(tickers, periods):
    prices = {period: {} for period in periods}
    lock = threading.Lock()
    threads = []
    for ticker in tickers:
        t = threading.Thread(target=fetch_and_store_multiple_price_data, args=(ticker, periods, prices, lock))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return tuple(prices[period] for period in periods)

class OptionData:

    def __init__(self, ticker):
        self.ticker = ticker
        self.lock = threading.Lock()

    @staticmethod
    def fetch(ticker, i):
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        if 0 <= i < len(options_dates):
            expiration_date = options_dates[i]
            options_data = stock.option_chain(expiration_date)
            call_options_df = options_data.calls
            put_options_df = options_data.puts
            relevant_columns = ['strike', 'lastTradeDate', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'openInterest']
            call_options_df = call_options_df[relevant_columns].copy()
            put_options_df = put_options_df[relevant_columns].copy()
            call_options_df['quoteDate'] = pd.to_datetime('today').strftime('%Y-%m-%d')
            call_options_df['expirationDate'] = expiration_date
            put_options_df['quoteDate'] = pd.to_datetime('today').strftime('%Y-%m-%d')
            put_options_df['expirationDate'] = expiration_date
            
            exp = pd.to_datetime(call_options_df['expirationDate'])
            quote = pd.to_datetime(call_options_df['quoteDate'])
            call_options_df['TYears'] =  (exp - quote).dt.days/252
            put_options_df['TYears'] = (exp - quote).dt.days/252
            call_options_df = call_options_df[['quoteDate', 'expirationDate', 'TYears'] + relevant_columns]
            put_options_df = put_options_df[['quoteDate', 'expirationDate', 'TYears'] + relevant_columns]
            return call_options_df.set_index('quoteDate'), put_options_df.set_index('quoteDate')
        
    def fetch_and_update(self, i, calls, puts):
        result = self.fetch(self.ticker, i)
        if result is not None:
            with self.lock:
                call_df, put_df = result
                calls.append(call_df)
                puts.append(put_df)
                
    def merge(self):
        calls, puts = [], []
        threads = []
        for i in range(20):
            thread = threading.Thread(target=self.fetch_and_update, args=(i, calls, puts))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return pd.concat(calls), pd.concat(puts)

    
# def pricing_thread_function(ticker, start, end, prices, lock, yahoo=True):
#     data = get_pricing(ticker, start=start, end=end, yahoo=yahoo)
#     with lock:
#         prices[ticker] = data

# def get_prices(tickers, start, end, yahoo):
#     prices = {}
#     lock = threading.Lock()
#     threads = []
#     for ticker in tickers:
#         t = threading.Thread(target=pricing_thread_function, args=(ticker, start, end, prices, lock, yahoo))
#         t.start()
#         threads.append(t)
#     for t in threads:
#         t.join()
#     return pd.DataFrame(prices).reindex(columns=tickers)

def pricing_thread_function(ticker, start, end, yahoo=False):
    data = get_pricing(ticker, start=start, end=end, yahoo=yahoo)
    return ticker, data

def get_prices(tickers, start, end, yahoo=False):
    prices = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(pricing_thread_function, ticker, start, end, yahoo): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker, data = future.result()
                prices[ticker] = data
            except Exception as exc:
                pass
    return pd.concat(prices, axis=1)

def get_snp():
    tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers.Symbol.to_list()

def get_dow():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
    return tickers.Symbol.to_list()

