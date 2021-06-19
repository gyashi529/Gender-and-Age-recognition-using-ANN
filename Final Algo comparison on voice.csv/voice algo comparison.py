# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:20:51 2021

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#loading the dataset
dataset = pd.read_csv('voice.csv')

#Encoding the categorical values - label colum
label_encoderGender = LabelEncoder()
dataset['label'] = label_encoderGender.fit_transform(dataset['label'])

x = dataset.iloc[:,:-1]
gender = dataset['label']

x_train, x_test, gender_train, gender_test= train_test_split(x, gender, test_size= 0.2, random_state=0)
#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Training Support Vector Machine
svm=SVC(random_state=42)
svm.fit(x_train,gender_train)
y_pred = svm.predict(x_test)
print("Accuracy of svm classifier is: {} %".format(svm.score(x_test,gender_test)*100))


#logistic regression
lr = LogisticRegression()
lr.fit(x_train,gender_train)
print("Accuracy of Logistic Regression classifier is: {} %".format(lr.score(x_test,gender_test)*100))

#KNN classifier
#First we have to find the best value for K

accuracyList = []
for each in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,gender_train)
    accuracyList.append(knn.score(x_test,gender_test))
    
plt.figure(figsize=(7,5))
plt.plot(range(1,20),accuracyList)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()

#using knn classifier with the value obtained
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,gender_train)
print("Accuracy of KNN classifier is: {} %".format(knn.score(x_test,gender_test)*100))


#using Naive Bayes
nb=GaussianNB()
nb.fit(x_train,gender_train)
print("Accuracy of Naive Bayes classifier is {} %".format(nb.score(x_test,gender_test)*100))

#Using Decision Tree

dt=DecisionTreeClassifier()
dt.fit(x_train,gender_train)
print("Accuracy of Decision Tree Classifier is {} %".format(dt.score(x_test,gender_test)*100))

#Using Random Forest Classifier
rf=RandomForestClassifier(n_estimators=10,random_state=42)
rf.fit(x_train,gender_train)
print("Accuracy of Random Forest Classifier is {} %".format(rf.score(x_test,gender_test)*100))


#DEEP LEARNING
#Artificial Neural Networks
# Initialize the constructor
model = Sequential()
# Add an input layer 
model.add(Dense(22, activation='relu', input_shape=(20,)))
# Add one hidden layer 
model.add(Dense(output_dim=50, activation='relu'))
model.add(Dropout(p=0.2))
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dropout(p=0.2))
model.add(Dense(output_dim=150, activation='relu'))
model.add(Dropout(p=0.2))
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dropout(p=0.2))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])                   
model.fit(x_train,gender_train,epochs=150, batch_size=64, verbose=1)
score = model.evaluate(x_test, gender_test,verbose=1)
print("Deep Learning",score)
y_pred = model.predict(x_test)
y_pred = np.round(y_pred,0)
y_pred = y_pred[:,0]


#2D Convolutional Network
print(dict(enumerate(label_encoderGender.classes_)))
y = dataset['label'].copy()
X = dataset.iloc[:,:-1].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)
X = tf.keras.preprocessing.sequence.pad_sequences(X, dtype=np.float, maxlen=25, padding='post')
X = X.reshape(-1, 5, 5)
X = np.expand_dims(X, axis=3)
print(X.shape)
plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(np.squeeze(X[i]))
    plt.axis('off')    
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
inputs = tf.keras.Input(shape=(X.shape[1], X.shape[2], X.shape[3]))
x = tf.keras.layers.Conv2D(16, 2, activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 1, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
print(model.evaluate(X_test, y_test))