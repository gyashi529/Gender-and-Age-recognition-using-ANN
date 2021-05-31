# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:07:16 2021

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
from sklearn.decomposition import PCA
import time, datetime
from keras.utils import np_utils

#loading the dataset
dataset = pd.read_csv('ageNewFeatures.csv')

x = dataset.iloc[:,:-1]
age = dataset['age']

encoder = LabelEncoder()
encoder.fit(age)
encoded_age = encoder.transform(age)
dummy_age = np_utils.to_categorical(encoded_age)

x_train, x_test, age_train, age_test= train_test_split(x, age, test_size= 0.2, random_state=0)
#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Training Support Vector Machine
svm=SVC(random_state=42)
svm.fit(x_train,age_train)
print("Accuracy of svm classifier is: {} %".format(svm.score(x_test,age_test)*100))

#logistic regression
lr = LogisticRegression()
lr.fit(x_train,age_train)
print("Accuracy of Logistic Regression classifier is: {} %".format(lr.score(x_test,age_test)*100))

#KNN classifier
#First we have to find the best value for K

accuracyList = []
for each in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,age_train)
    accuracyList.append(knn.score(x_test,age_test))
    
plt.figure(figsize=(7,5))
plt.plot(range(1,40),accuracyList)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()

#using knn classifier with the value obtained
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,age_train)
print("Accuracy of KNN classifier is: {} %".format(knn.score(x_test,age_test)*100))


#using Naive Bayes
nb=GaussianNB()
nb.fit(x_train,age_train)
print("Accuracy of Naive Bayes classifier is {} %".format(nb.score(x_test,age_test)*100))

#Using Decision Tree

dt = DecisionTreeClassifier(max_depth=24,random_state=42)
dt.fit(x_train,age_train)
print("Accuracy of Decision Tree Classifier is {} %".format(dt.score(x_test,age_test)*100))

# #Using Random Forest Classifier
rf=RandomForestClassifier(n_estimators=10,random_state=42)
rf.fit(x_train,age_train)
print("Accuracy of Random Forest Classifier is {} %".format(rf.score(x_test,age_test)*100))


# #DEEP LEARNING
# #Artificial Neural Networks
# # Initialize the constructor
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

