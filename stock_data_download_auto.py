import nsepy
import datetime
import pandas as pd
from tqdm import tqdm
import pickle

def getHistoricalData(stock_name):
    current_date = datetime.date.today()
    period = 365
    start = datetime.date.today()+datetime.timedelta(-period)

    stock_data = nsepy.get_history(stock_name, start=start, end=current_date)
    return stock_data



def run():
    stock_symbols = pd.read_csv('resources/NIFTY_100.csv').loc[:,'Symbol']


    data = {}

    for i in tqdm(range(len(stock_symbols))):
        d = getHistoricalData(stock_symbols.iloc[i])
        if(d.shape[0]>0):
            data[stock_symbols.iloc[i]] = d
        

    return  data;

