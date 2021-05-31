# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 15:01:18 2021

@author: Yashi
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt



dataframe = pd.read_csv('E:/Major Project/Age/final algo comparison on ageNewFeatures.csv/ageNewFeatures.csv')

scaler = StandardScaler()
dataframe.iloc[:,:-1] = scaler.fit_transform(dataframe.iloc[:,:-1])


age = dataframe['age']

encoder = LabelEncoder()
encoder.fit(age)
encoded_age = encoder.transform(age)
dataframe['age'] = encoded_age


plt.subplots(1,1,figsize=(35,35))
#for i in range(26,45):
plt.subplot(5,5,1)
plt.title(dataframe.columns[44-20])
sns.kdeplot(dataframe.loc[dataframe['age'] == 0, dataframe.columns[44-20]], color= 'red', label='classA')
sns.kdeplot(dataframe.loc[dataframe['age'] == 1, dataframe.columns[44-20]], color= 'blue', label='classB')
sns.kdeplot(dataframe.loc[dataframe['age'] == 2, dataframe.columns[44-20]], color= 'yellow', label='classC')
plt.legend()
    



