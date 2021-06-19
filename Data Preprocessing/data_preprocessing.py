# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:49:33 2021

@author: Yashi
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#importing the dataset using pandas library
data = pd.read_csv('E:/Major Project/Data Preprocessing/dataset.csv')
#creating a matrix of independent variables
x = data.iloc[:,:-2].values
#separating the dependent variables - age and gender
#creating an array for age
age = data.iloc[:,23]
#creating an array for gender
gender = data.iloc[:,22]
#Checking if there are any null values in our dataset 
data.isna().sum()   #There are no null values
#Encoding the categorical data - gender (We set 0 for male and 1 for female) - using label Encoder
label_encoderGender = LabelEncoder()
data['Sex'] = label_encoderGender.fit_transform(data['Sex'])

#Splitting the dataset into test and train - ratio 20:80

x_train, x_test, gender_train, gender_test= train_test_split(x, gender, test_size= 0.2, random_state=0)
x_train, x_test, age_train, age_test= train_test_split(x, age, test_size= 0.2, random_state=0)  

#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)












