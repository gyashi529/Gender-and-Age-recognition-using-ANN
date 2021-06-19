# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 22:53:28 2021

@author: Yashi
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt

dataframe = pd.read_csv('E:/Major Project/Trying with both languages/dataset.csv')


scaler = StandardScaler()
dataframe.iloc[:,:-2] = scaler.fit_transform(dataframe.iloc[:,:-2])

labelencoder = LabelEncoder()
dataframe['Sex'] = labelencoder.fit_transform(dataframe['Sex'])

gender = dataframe['Sex']

plt.subplots(1,1,figsize=(30,30))
#for i in range(1,23):
plt.subplot(5,5,1)
plt.title(dataframe.columns[21-2])
sns.kdeplot(dataframe.loc[dataframe['Sex'] == 0, dataframe.columns[21-2]], color= 'red', label='Female')
sns.kdeplot(dataframe.loc[dataframe['Sex'] == 1, dataframe.columns[21-2]], color= 'green', label='Male')
plt.legend()