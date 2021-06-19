# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:32:16 2021

@author: Yashi
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

#importing the dataset using pandas library
data = pd.read_csv('E:/Major Project/Trying with both languages/dataset.csv')

#Encoding the categorical data - gender (We set 0 for male and 1 for female) - using label Encoder
label_encoderGender = LabelEncoder()
data['Sex'] = label_encoderGender.fit_transform(data['Sex'])

#creating a matrix of independent variables
x = data.iloc[:,:-2].values
#separating the dependent variables - age and gender
#creating an array for age
age = data.iloc[:,23]
#creating an array for gender
gender = data.iloc[:,22]
#Checking if there are any null values in our dataset 
data.isna().sum()   #There are no null values

#Splitting the dataset into test and train - ratio 20:80
x_train, x_test, gender_train, gender_test= train_test_split(x, gender, test_size= 0.2, random_state=0)
x_train, x_test, age_train, age_test= train_test_split(x, age, test_size= 0.2, random_state=0)  

#Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#checking if our dataset is linear or not
model = LinearRegression()
model.fit(x,gender)
model_predict = model.predict(x)
print(r2_score(gender,model_predict))  #r2_score is high, our dataset  is linear

#PCA Analysis
pca = PCA(n_components = 17)
pcaComponenets = pca.fit_transform(x_train,gender_train)
print(pca.explained_variance_)
scorePCA = pca.score(x_test,gender_test)

print("PCA score for gender",scorePCA)

pcaAge = pca.fit_transform(x_train,age_train)
scorePCAAge = pca.score(x_test,age_test)
print("PCA score for Age",scorePCAAge)

# #LDA Analysis for gender
# lda = LinearDiscriminantAnalysis()
# ldaTrain = lda.fit_transform(x_train,gender_train.ravel())
# scoreLDA = lda.score(x_test,gender_test)
# print("LDA score for gender", scoreLDA)

# #LDA Analysis for age
# ldaTrain2 = lda.fit_transform(x_train,age_train)
# scoreLDAAge = lda.score(x_test,age_test)
# print("LDA score for AGE ",scoreLDAAge)





































