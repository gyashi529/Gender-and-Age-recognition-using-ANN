# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:03:21 2021

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
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

dataset = pd.read_csv('E:/Major Project/Age/dataset/ageNewFeatures.csv')


x = dataset.iloc[:,:-1]
age = dataset['age']

scaler = StandardScaler()
x = scaler.fit_transform(x)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(age)
encoded_age = encoder.transform(age)
dummy_age = np_utils.to_categorical(encoded_age)

x_train, x_test, age_train, age_test= train_test_split(x, dummy_age, test_size= 0.2, random_state=0)


model = Sequential()
# Add an input layer 
model.add(Dense(45, activation='relu', input_shape=(45,)))
# Add one hidden layer 
model.add(Dense(output_dim=300, activation='relu'))

model.add(Dense(output_dim=300, activation='relu'))

model.add(Dense(output_dim=300, activation='relu'))

model.add(Dense(output_dim=200, activation='relu'))

model.add(Dense(output_dim=200, activation='relu'))

model.add(Dense(output_dim=150, activation='relu'))

model.add(Dense(output_dim=150, activation='relu'))

# Add an output layer 
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])   
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)            
model.fit(x_train,age_train,epochs=200,batch_size=200, verbose=1,callbacks=[early_stop])
score = model.evaluate(x_test, age_test,verbose=1)
print("Deep Learning",score)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred,0)



