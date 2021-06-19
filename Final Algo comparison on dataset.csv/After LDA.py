# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:12:15 2021

@author: Yashi
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time, datetime
from sklearn.decomposition import FastICA

#loading the dataset
dataset = pd.read_csv('dataset.csv')

#Encoding the categorical values - label colum
label_encoderGender = LabelEncoder()
dataset['Sex'] = label_encoderGender.fit_transform(dataset['Sex'])

x = dataset.iloc[:,:-3]
gender = dataset['Sex']

x_train, x_test, gender_train, gender_test= train_test_split(x, gender, test_size= 0.2, random_state=0)
#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


ica = FastICA(n_components=17)
x_train = ica.fit_transform(x_train)
x_test = ica.transform(x_test)

#DEEP LEARNING
#Artificial Neural Networks
# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(17, activation='relu', input_shape=(17,)))
# Add one hidden layer 
model.add(Dense(output_dim=10, activation='relu'))
model.add(Dropout(p=0.2))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])    
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10) 
start = datetime.datetime.now()              
model.fit(x_train,gender_train,epochs=150, batch_size=64, verbose=1,callbacks=[early_stop])
time.sleep(2)
end = datetime.datetime.now()
print("The time difference is",end-start)
score = model.evaluate(x_test, gender_test,verbose=1)
print("Deep Learning",score)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred,0)

