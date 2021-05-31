# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:15:01 2021

@author: kshitiz
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.decomposition import PCA
import time, datetime


#loading the dataset
dataset = pd.read_csv('C:/Users/kshitiz/Desktop/project/Age/ageFeatures.csv')

x = dataset.iloc[:,:-1]
age = dataset['age']

x_train, x_test, age_train, age_test= train_test_split(x, age, test_size= 0.2, random_state=0)
#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Training Support Vector Machine
# svm=SVC(kernel='linear', C=1, decision_function_shape='ovo',random_state=42)
# svm.fit(x_train,age_train)
# print("Accuracy of svm classifier is: {} %".format(svm.score(x_test,age_test)*100))


#DEEP LEARNING
#Artificial Neural Networks
# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(24, activation='relu', input_shape=(24,)))
# Add one hidden layer 
model.add(Dense(output_dim=12, activation='relu'))
model.add(Dropout(p=0.2))
# Add an output layer 
model.add(Dense(1, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])    
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10) 
start = datetime.datetime.now()              
model.fit(x_train,age_train,epochs=150, batch_size=64, verbose=1,callbacks=[early_stop])
time.sleep(2)
end = datetime.datetime.now()
print("The time difference is",end-start)
score = model.evaluate(x_test, age_test,verbose=1)
print("Deep Learning",score)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred,0)



