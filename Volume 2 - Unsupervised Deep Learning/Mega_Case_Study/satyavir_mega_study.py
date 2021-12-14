import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

x = sc.fit_transform(x)

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(data=x, num_iteration=400)

#mean internuron distance MID
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()

marker =['o', 's']
colors = ['r', 'g']

for i, xt in enumerate(x):
    w = som.winner(xt)
    plot(w[0] + 0.5, w[1] + 0.5, marker[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor=None,
         markersize=10,
         markeredgewidth=2)

show()

#finding the fraud customer
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(3, 8)], mappings[(2, 4)]), axis=0)
frauds = sc.inverse_transform(frauds)

customers = dataset.iloc[:,1:].values

#fraud or not
is_frauds = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_frauds[i] = 1


#CNN model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(customers, is_frauds, batch_size = 1, epochs = 2)



# Predicting the probabilities
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:,0:1], y_pred), axis=1)
y_pred = y_pred[y_pred[:,1].argsort()]









