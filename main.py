# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 02:30:40 2022

@author: VAISHNAV
"""
import pandas as pd
from tqdm import tqdm
from stock_object import Stock
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import stock_data_download_auto as std
from numba import jit, cuda
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


#this will create a pkl files which contains data for the last 1 year of all the companies in NIFTY100
data = std.run()       
          

#function to create Stock Object for each stock in data.
#the object blueprint can be found in stock_object.py
def createStockObj():
    stocks = {}
    for symbol in tqdm(data):
        stocks[symbol]=Stock(symbol)
        stocks[symbol].addData(data[symbol])
        print(stocks[symbol].symbol)
        stocks[symbol].preProcess()
        stocks[symbol].createModel()
        stocks[symbol].predPrice()
        
    return stocks

#Creating a dictionairy consisting of Stock objects which contains the model for each stock
stocks_obj_list = createStockObj()

#Sorting the stocks by the predicted price
sorted_stocks = sorted(stocks_obj_list, key = lambda stock_name: stocks_obj_list[stock_name].percentage, reverse=True)

final_stock_pl_sorted = {}

for k in sorted_stocks:
    final_stock_pl_sorted[k] = stocks_obj_list[k].percentage


#saving the final predicted prices
with open('resources/pred_stocks.pkl','wb') as file:
    pickle.dump(final_stock_pl_sorted,file)
        

sym = 'PAYTM'
t = {sym:data[sym]}
data = t
ob = createStockObj()

ob[sym].percentage
