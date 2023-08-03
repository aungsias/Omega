import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

def get_vline_x(date):
    vline_x = pd.to_datetime(date)
    vline_x_num = mdates.date2num(vline_x)  
    return vline_x, vline_x_num

def plot_with_fill(price_series, lagged_price_series, ticker, window, date='2023-01-01'):

    price_plot = lagged_price_series.plot(title=ticker, legend=False, color='#800000', grid=False)
    price_line = mlines.Line2D([], [], color='#800000', markersize=15, label=f'SMA{window}')
    price_series.plot(ax=price_plot.twinx(), color='b', legend=False, alpha=0.25, grid=False, yticks=[])
    lagged_line = mlines.Line2D([], [], color='b', markersize=15, label='Prices', alpha=0.25)

    vline_x, vline_x_num = get_vline_x(date)
    price_plot.axvline(x=vline_x, color='k', linewidth=2)
    xlim, ylim = price_plot.get_xlim(), price_plot.get_ylim()

    rect = patches.Rectangle((xlim[0], ylim[0]), vline_x_num-xlim[0], ylim[1]-ylim[0], facecolor='k', alpha=0.25)
    price_plot.add_patch(rect)
    price_plot.legend(handles=[lagged_line, price_line])
    plt.show()

def plot_with_factor(price_series, computed, ticker, factor, date='2023-01-01'):

    price_plot = computed.plot(title=ticker, legend=False, color='#800000', grid=False)
    price_line = mlines.Line2D([], [], color='#800000', markersize=15, label=f'{factor.name}')
    price_series.plot(ax=price_plot.twinx(), color='b', legend=False, alpha=0.25, grid=False, yticks=[])
    lagged_line = mlines.Line2D([], [], color='b', markersize=15, label='Prices', alpha=0.25)

    vline_x, vline_x_num = get_vline_x(date)
    price_plot.axvline(x=vline_x, color='k', linewidth=2)
    xlim, ylim = price_plot.get_xlim(), price_plot.get_ylim()

    rect = patches.Rectangle((xlim[0], ylim[0]), vline_x_num-xlim[0], ylim[1]-ylim[0], facecolor='k', alpha=0.25)
    price_plot.add_patch(rect)
    price_plot.legend(handles=[lagged_line, price_line])
    plt.show()

def backtest(prices_and_scores, ticker, year='2023', v=False):
    data = prices_and_scores.loc[(slice(None), ticker), :].reset_index().set_index('Date').loc[year].drop(columns='Equity')
    data['Returns'] = data['Prices'].pct_change()
    data['PrevMomentumScore'] = data['MomentumScores'].shift()
    data['MomentumSignChange'] = data.MomentumScores.apply(np.sign)
    data['Buy'] = ((data['MomentumSignChange'] == 1) & (data['PrevMomentumScore'] < 0) & (data['MomentumScores'] > 0)).astype(int).shift()
    data['Sell'] = ((data['MomentumSignChange'] == -1) & (data['PrevMomentumScore'] > 0) & (data['MomentumScores'] < 0)).astype(int).shift()
    data['Bought'] = np.nan
    data.loc[data['Buy'] == 1, 'Bought'] = 1
    data.loc[data['Sell'] == 1, 'Bought'] = 0
    data['Bought'] = data['Bought'].ffill().fillna(0)
    data['InvestmentReturns'] = data['Returns'] * data['Bought']
    data['CumReturns'] = (1 + data['InvestmentReturns']).cumprod() - 1
    bh = data.Prices.iloc[-1] / data.Prices.iloc[0] - 1
    ret = data.CumReturns.iloc[-1]
    alpha = ret - bh
    if v:
        print(ticker)
        print('-'*60)
        print(f'Buy-Hold Return: {bh:,.2f}\n'
            f'Algo Return: {ret:,.2f}\n'
            f'Algo Alpha: {alpha:,.2f}\n')
    return alpha, ret, bh
    
def backtest_en_masse(tickers, prices_and_scores, year='2023'):
    ticker_data = {}
    for ticker in tickers:
        ticker_dict = {}
        alpha, algo_ret, buy_hold_ret = backtest(prices_and_scores, ticker, year=year)
        ticker_dict['alpha'] = alpha
        ticker_dict['algo_return'] = algo_ret
        ticker_dict['buy_hold_return'] = buy_hold_ret
        ticker_data[ticker] = ticker_dict
    return pd.DataFrame(ticker_data)

def plot_backtest_hist(ticker_data):
    data = ticker_data.T    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(data['alpha'], bins=30, color='#800000', edgecolor='black')
    plt.title('Alpha')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.hist(data['algo_return'], bins=30, color='#6BB26A', edgecolor='black')
    plt.title('Algo Return')
    plt.xlabel('Value')
    plt.subplot(1, 3, 3)
    plt.hist(data['buy_hold_return'], bins=30, color='#6265FF', edgecolor='black')
    plt.title('Buy Hold Return')
    plt.xlabel('Value')
    plt.show()

def plot_price_and_momentum(prices_and_scores, ticker, year='2023'):
    data = prices_and_scores.loc[(slice(None), ticker), :].reset_index().set_index('Date').loc[year:].drop(columns='Equity').reset_index()
    _, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10), gridspec_kw={'height_ratios': [2, 1]})
    axes[0].plot(data['Date'], data['Prices'], label='Prices', linewidth=2, color='#800000')
    axes[0].set_title('Prices over Time')
    axes[0].set_ylabel('Prices')
    axes[0].legend()
    colors = ['#6BB26A' if x >= 0 else '#800000' for x in data['MomentumScores']]
    axes[1].plot(data['Date'], data['MomentumScores'], label='Momentum Scores', color='#6265FF')
    axes[1].set_title('Momentum Scores over Time')
    axes[1].set_ylabel('Momentum Scores')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
