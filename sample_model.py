# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:42:40 2022

@author: Vaishnav
"""

#importing the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#importing the test set
dataset_train = pd.read_csv("resources/PAYTM.csv")
training_set = dataset_train.loc[:, ["Open Price","Prev Close","Total Traded Quantity","Close Price"]].values

#Feature Scaling
sc_x = MinMaxScaler(feature_range=(0,1))
sc_y = MinMaxScaler(feature_range=(0,1))


#Applying the feature scaling 
x_scaled = sc_x.fit_transform(training_set)
y_scaled = sc_y.fit_transform(training_set[:,[0,2,3]])

#Creating a Data structure with 38 timesteps and one output
x_train = []
y_train = []

timesteps = 38

for i in range(timesteps, x_scaled.shape[0]):
    x_train.append(x_scaled[i-timesteps:i, :])
    y_train.append(y_scaled[i,:])
    
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))

#Building RNN
regressor = Sequential()

#first layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 4)))
regressor.add(Dropout(0.2)) # this means 20% of the neurons will be ignored at each iteration of training

#second layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#third layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#fourth layer
regressor.add(LSTM(units = 50)) # return sequences will be false here since this is the last LSTM layers
regressor.add(Dropout(0.2))

regressor.add(Dense(units=3))

#compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the training data
regressor.fit(x_train,y_train, batch_size=32, epochs = 100)


#Getting the stock data for the next market day
dataset_total = dataset_train.loc[:, ["Open Price","Prev Close","Total Traded Quantity","Close Price"]]
inputs = dataset_total[len(dataset_total)-timesteps:].values
inputs = inputs.reshape(-1,4)
inputs = sc_x.transform(inputs)

pred = []

mn_input = np.array([inputs[:timesteps,:]])
mn_input =  np.reshape(mn_input, (mn_input.shape[0], mn_input.shape[1], 4))
tom_price = sc_y.inverse_transform(regressor.predict(mn_input))
#print(tom_price)
print('Price for the next market day:',round((tom_price[0][0]+tom_price[0][2])/2,2))

