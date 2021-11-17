import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Churn_Modelling.csv')

column = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
          'IsActiveMember', 'EstimatedSalary', 'Exited']



df = df[column]

df = pd.get_dummies(df, drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


#build ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

#add  first hidden layes
classifier.add(Dense(input_dim=X.shape[1], units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)








