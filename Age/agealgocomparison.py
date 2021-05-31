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
from sklearn.decomposition import PCA
import time, datetime

#loading the dataset
dataset = pd.read_csv('E:/Major Project/Age/dataset/ageFeatures.csv')


x = dataset.iloc[:,:-1]
age = dataset['age']

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
for each in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,age_train)
    accuracyList.append(knn.score(x_test, age_test))
    
plt.figure(figsize=(7,5))
plt.plot(range(1,20),accuracyList)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()

#using knn classifier with the value obtained
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,age_train)
print("Accuracy of KNN classifier is: {} %".format(knn.score(x_test,age_test)*100))


#using Naive Bayes
nb=GaussianNB()
nb.fit(x_train,age_train)
print("Accuracy of Naive Bayes classifier is {} %".format(nb.score(x_test,age_test)*100))

#Using Decision Tree

dt=DecisionTreeClassifier()
dt.fit(x_train,age_train)
print("Accuracy of Decision Tree Classifier is {} %".format(dt.score(x_test,age_test)*100))

#Using Random Forest Classifier
rf=RandomForestClassifier(n_estimators=10,random_state=42)
rf.fit(x_train,age_train)
print("Accuracy of Random Forest Classifier is {} %".format(rf.score(x_test,age_test)*100))


# pca = PCA(n_components=19)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)


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
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',
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

#2D Convolutional Network
# print(dict(enumerate(label_encoderGender.classes_)))

y = dataset['age'].copy()
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

modelcnn= tf.keras.Model(inputs, outputs)

print(model.summary())

modelcnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)

history = modelcnn.fit(
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
print(modelcnn.evaluate(X_test, y_test))

