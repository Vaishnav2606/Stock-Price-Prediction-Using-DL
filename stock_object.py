# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:39:14 2022

@author: VAISHNAV
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dropout
from keras.metrics import Accuracy
from keras.callbacks import Callback



class MyThresholdCallback(Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        loss = logs["loss"]
        if loss <= self.threshold:
            self.model.stop_training = True

class Stock:
    
    def __init__(self,symbol):
        self.symbol = symbol
        self.data = ''
        self.model = ''
        self.timesteps = ''
        self.pred_start = ''
        self.pred_close = ''
        self.pred_avg_price = ''
        self.x_scaler = ''
        self.y_scaler = ''
        self.x_data = ''
        self.y_data = ''
        self.prev_price = ''
        self.percentage = ''
        
    def addData(self,data):
        self.data = data.loc[:,["Open","Low","Close","Volume"]].values
        self.timesteps = int(round(self.data.shape[0]*0.08,0))
        self.prev_price = self.data[self.data.shape[0]-1,2]
        
    def preProcess(self):
        self.x_scaler = MinMaxScaler(feature_range=(0,1))
        self.y_scaler = MinMaxScaler(feature_range=(0,1))
        x_scaled = self.x_scaler.fit_transform(self.data)
        y_scaled = self.y_scaler.fit_transform(self.data[:,[0,2,3]])
        
        x_train = []
        y_train = []

        for i in range(self.timesteps, x_scaled.shape[0]):
            x_train.append(x_scaled[i-self.timesteps:i, :])
            y_train.append(y_scaled[i,:])
            
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))
        self.x_data = x_train
        self.y_data = y_train
        
    def createModel(self):
        regressor = Sequential()

        #first layer
        regressor.add(CuDNNLSTM(units = 50, return_sequences = True, input_shape = (self.x_data.shape[1], 4)))
        regressor.add(Dropout(0.2)) # this means 20% of the neurons will be ignored at each iteration of training

        #second layer
        regressor.add(CuDNNLSTM(units = 50, return_sequences = True))
        regressor.add(Dropout(0.2))

        #third layer
        regressor.add(CuDNNLSTM(units = 25, return_sequences = True))
        regressor.add(Dropout(0.2))

        #fourth layer
        regressor.add(CuDNNLSTM(units = 25)) 
        regressor.add(Dropout(0.2))
        
        regressor.add(Dense(units=25))
        regressor.add(Dense(units=3))

        #compiling the RNN
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',  metrics=['accuracy'])

        #fitting the training data
        regressor.fit(self.x_data,self.y_data, batch_size=32, epochs = 100)
        
        self.model = regressor
        
    def predPrice(self):
        pred_input = self.data[len(self.data)-self.timesteps:]
        pred_input = pred_input.reshape(-1,4)
        pred_input = self.x_scaler.transform(pred_input)
        
        pred_input = np.array([pred_input[:self.timesteps,:]])
        pred_input =  np.reshape(pred_input, (pred_input.shape[0], pred_input.shape[1], 4))
        
        tom_price = self.y_scaler.inverse_transform(self.model.predict(pred_input))
        self.pred_start = tom_price[0][0]
        self.pred_close = tom_price[0][1]
        self.pred_avg_price = round((tom_price[0][0]+tom_price[0][1])/2,2)
        self.percentage = round(((self.pred_avg_price-self.prev_price)/self.prev_price),4)*100
        self.percentage = str(self.percentage) 