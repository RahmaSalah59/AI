# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 00:47:26 2025

@author: ascom
"""


import pandas as pd
dataset = pd.read_csv("diabetes.csv")

x = dataset.iloc[:,0:8].values
y = dataset.iloc[:,8].values


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 , random_state= 0)


from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(12, input_dim = 8 , activation='relu'))
model.add(Dense(8 , activation='relu'))
model.add(Dense(1 , activation='sigmoid'))


model.compile(loss= 'binary_crossentropy', optimizer= 'adam' , metrics=['accuracy'])

model.fit(x_train, y_train , epochs=100 ,batch_size=10)

scores = model.evaluate(x_test , y_test)
print(model.metrics_names[1] , scores[1]*100)