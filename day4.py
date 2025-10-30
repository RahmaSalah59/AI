# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,[0]].values
Y = dataset.iloc[:,1].values

#splitting trainning and testing
#train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.20)

#trainning algorithm
from sklearn.linear_model import LinearRegression
# class > object
regressor = LinearRegression()
#fit function (trainning)
regressor.fit(x_train, y_train)

# testing

y_predict = regressor.predict(x_test)

#evaluation mean square error or mean absolute error
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test,y_predict))

# عايزين نطبع المعادلة الخاصه بالمودل
A = regressor.coef_
b = regressor.intercept_

print("y = ",A ,"x + ",b)

# use model in real life
print(regressor.predict([[2]]))


#visualization of trainning data
import matplotlib.pyplot as plt 

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.xlabel("years of experinces")
plt.ylabel("Salary")


#visualization of testing data

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.xlabel("years of experinces")
plt.ylabel("Salary")





















