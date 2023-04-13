import datetime
import pandas as pd
from tqdm import tqdm
import pickle
from numba import jit, cuda
import yfinance as yf

def getHistoricalData(stock_name):
    stock_name = stock_name + '.NS'
    time_interval = '3y'
    stock = yf.Ticker(stock_name)
    stock_data = stock.history(period=time_interval)
    print('sdcs')
    if stock_data.empty:
        return stock_data
    stock_data.index = stock_data.index.date
    return stock_data

def run():
    stock_symbols = pd.read_csv('resources/NIFTY_100.csv').loc[:,'Symbol']


    data = {}

    for i in tqdm(range(len(stock_symbols))):
        d = getHistoricalData(stock_symbols.iloc[i])
        if(not d.empty):
            data[stock_symbols.iloc[i]] = d
        

    return  data
