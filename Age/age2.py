# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:58:16 2021

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
from sklearn.metrics import accuracy_score
#loading the dataset
dataset = pd.read_csv('E:/Major Project/Age/dataset/ageNewFeatures.csv')


x = dataset.iloc[:,:-1]
age = dataset['age']

x_train, x_test, age_train, age_test= train_test_split(x, age, test_size= 0.2, random_state=0)
#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Training Support Vector Machine
poly = SVC(kernel='poly', degree=3, C=1).fit(x_train, age_train)
age_pred = poly.predict(x_test)
print("Accuracy of svm classifier is: {} %".format(accuracy_score(age_test, age_pred)))

