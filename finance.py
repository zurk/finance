import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from pandas_datareader.data import DataReader
from pandas_datareader.moex import MoexReader


def corrfunc(x, y, **kws):
    r, _ = scipy.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    return ax


def plot_adjusted_price(ticker_price, figsize=(16, 9)):
    first_valid_index = ticker_price.apply(lambda x: max(x.notna().argmax(), (x != 0).argmax()))
    ticker_price = ticker_price / ticker_price.apply(lambda x: x[first_valid_index[x.name]])
    fig, ax = plt.subplots(figsize=figsize)
    ticker_price.plot(ax=ax)
    for name, price in ticker_price.iteritems():
        ax.annotate(xy=(price.index[-1], price.iloc[-1]), xytext=(5, 0), textcoords='offset points', text=name, va='center')
    ax.set_ylabel('Adjusted closing price ($)')
    return ax


def plot_instruments_correlation(ticker_price):
    g = sns.PairGrid(ticker_price, height=4.5)
    g.map_upper(plt.scatter, s=1, marker=".", alpha=0.25)
    g.map_diag(sns.distplot, kde=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(corrfunc)
    return g


def plot_mean_avg(ticker_price, windows=(20, 100)):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(ticker_price.index, ticker_price, label=ticker_price.name)
    for w in windows:
        rolling_ticker_price = ticker_price.rolling(window=w).mean()
        ax.plot(rolling_ticker_price.index, rolling_ticker_price, label='%d days rolling' % w)
        ax.set_xlabel('Date')
        ax.set_ylabel('Adjusted closing price ($)')
        ax.legend()
    return ax


def plot_accumulated_log_return(ticker_price):
    ticker_price = ticker_price / ticker_price.iloc[0, :]
    ticker_price = np.log(ticker_price)
    ax = ticker_price.plot(figsize=(16, 9))
    ax.set_ylabel('Accumulated log return')
    return ax


def correlation_matrix(tickers_price):
    corr = tickers_price.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(13, 13))
    ax = sns.heatmap(corr, mask=mask, square=True, annot=corr, vmin=-1, vmax=1, cmap='RdBu_r',
                     cbar=False)
    return ax


def getMoexData(symbol, start_date, end_date):
    t = MoexReader(symbol, start_date, end_date).read()
    t = t[~t.CLOSE.isna()]
    boardid = t.BOARDID.groupby(t.BOARDID).count().idxmax()
    t = t[t.BOARDID == boardid]
    t = t.drop_duplicates()
    t.index = pd.DatetimeIndex(t.index)
    t = t[~t.CLOSE.isna()]
    return t.CLOSE


def getOtherData(symbol, start_date, end_date):
    try:
        # Yahoo
        data = DataReader([symbol], 'yahoo', start_date, end_date)
        tickers_data = data["Close"]
    except:
        # iex
        data = DataReader([symbol], 'iex', start_date, end_date)
        tickers_data = data["close"]
    return tickers_data


def getData(symbols, start_date, end_date):
    res = pd.DataFrame(columns=symbols,
                       index=pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date)))
    for s in symbols:
        # try:
        #     res[s] = getMoexData(s, start_date, end_date)
        # except:
        res[s] = getOtherData(s, start_date, end_date)
        print(f"{s} First not NA value date: {res[s].index[~res[s].isna()].min()}")
    res[res == 0] = float("nan")

    return res.fillna(method="ffill")
