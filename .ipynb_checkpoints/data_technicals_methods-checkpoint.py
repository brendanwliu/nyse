import os
import time
import numpy as np
import pandas as pd

import datetime
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt

import math

def parser(x):
    return datetime.datetime.strptime(x,'%Y-%m-%d')
# dataset_gs_raw = pd.read_csv('gs_prices.csv')
# dataset_gs_raw = dataset_gs_raw.drop('symbol', axis = 1)
# print(dataset_gs_raw.head(3))
def extract_price(ticker):
    '''Takes the .csv from kaggle turns it into a 3 column file with only the one stock in question'''
    dataset_ex_df = pd.read_csv('prices.csv',header = 0, index_col = 'symbol',parse_dates = [0], date_parser = parser)
    dataset_raw = dataset_ex_df.loc[ticker,['close', 'date']]
    dataset_raw.to_csv('_'+ticker+'_prices.csv')
def prep_data(ticker):
    '''takes the extracted stock prices and drops the symbol column and returns the dataset'''
    dataset_ticker_raw = pd.read_csv('data_input/'+'_'+ticker+'_prices.csv')
    dataset_ticker_raw = dataset_ticker_raw.drop('symbol', axis = 1)
    return dataset_ticker_raw
#extract_price('GS')

def plot_price(dataset):
    plt.figure(figsize=(14, 5), dpi=100)
    plt.plot(dataset['date'], dataset['close'])
    plt.xlabel('Date')
    plt.xticks([])
    plt.ylabel('USD')
    plt.title('Figure 2: Stock price')
    plt.legend()
    plt.show()

def get_technicals_indic(dataset):
    '''we need to create technical indicators for the stock prices, we can add more, like RSI, later'''
    #7 and 21 day SMA
    dataset['ma7'] = dataset['close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['close'].rolling(window=21).mean()

    #create macd
    dataset['26ema'] = dataset['close'].ewm(span=26, adjust = False).mean()
    dataset['12ema'] = dataset['close'].ewm(span=12, adjust = False).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    #bollinger bands
    dataset['30_Day_STD'] = dataset['close'].rolling(window=30).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['30_Day_STD'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['30_Day_STD'] * 2)
    # dataset['20sd'] = pd.stats.moments.rolling_std(dataset['close'],20)
    # dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    # dataset['upper_band'] = dataset['ma21'] - (dataset['20sd']*2)

    #create ema
    # dataset['ema'] = dataset['close'].ewm(com = 0.5).mean()

    #momentum
    dataset['momentum'] = dataset['close'].pct_change()

    return dataset

def plotting_TI(dataset, days_vis):
    plt.figure(figsize=(16,10), dpi=100)
    numdates = dataset.shape[0]
    vis_macd = numdates-days_vis

    dataset = dataset.iloc[-days_vis:,:]
    x_ = range(3,dataset.shape[1])
    x_ = list(dataset.index)

    plt.subplot(2,1,1)
    plt.plot(dataset['ma7'], label = 'MA 7', color = 'g', linestyle='--')
    plt.plot(dataset['ma21'], label = 'MA 21', color = 'r', linestyle='--')
    plt.plot(dataset['upper_band'], label = 'Upper Band', color = 'c')
    plt.plot(dataset['lower_band'], label = 'Lower Band', color = 'c')
    plt.plot(dataset['close'], label = 'price', color = 'b')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical Indicators for stock for the past {} days'.format(days_vis))
    plt.ylabel('USD')
    plt.xticks([])
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(dataset['MACD'], label = 'MACD', linestyle='-.')
    plt.hlines(15, vis_macd, numdates, colors = 'g', linestyle = '--')
    plt.hlines(-15, vis_macd, numdates, colors = 'g', linestyle = '--')
    plt.plot(dataset['momentum'], label = 'Momentum', color = 'r', linestyle='-')
    plt.title('MACD')
    plt.xticks([])
    plt.legend()

    plt.show()
