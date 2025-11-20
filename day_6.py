# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 01:21:36 2025

@author: ascom
"""


import pandas as pd
 
 # Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

x = dataset.iloc[:,2:4].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25 , random_state= 0)

#from sklearn.tree import DecisionTreeClassifier

#classifier = DecisionTreeClassifier()

from sklearn.svm import SVC

classifier = SVC(kernel ='linear')

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

 