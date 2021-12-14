import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Google_Stock_Price_Train.csv')

# training_set = df['Open'].values
training_set = df.iloc[:,1:2].values


#feature scalling 
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#getting input and output
x_train = training_set[0:-1]
y_train = training_set[1:]

#reshaping
x_train = np.reshape(x_train, (len(x_train), 1, 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM

regressor = Sequential()

#add LSTM
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

#add ouput layes
regressor.add(Dense(units=1))

#commmpile RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, epochs=200, batch_size=32)


test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

# test = sc.transform(test)
inputs = real_stock_price
inputs = sc.transform(inputs)

inputs = np.reshape(inputs, (len(inputs), 1, 1))
pred = regressor.predict(inputs)
pred = sc.inverse_transform(pred)



#measure performance

real_stock_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_train = real_stock_train.iloc[:,1:2].values

pred_train = pred = regressor.predict(x_train)
pred_train = sc.inverse_transform(pred_train)

plt.plot(real_stock_train, color='red',label='Real stock price')
plt.plot(pred_train, color='blue',label='Predicted stock price')
plt.title = 'Goole Sock Price Prediction'
plt.ylabel('Time')
plt.xlabel('Google stock Price')
plt.legend()
plt.show()

#evaluate model
import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, pred))








