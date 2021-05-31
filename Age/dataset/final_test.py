# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:51:47 2021

@author: Yashi
"""

from testcode import load_data

def aggregate_mean(df, column):
    return df[column].mean()

def test_aggregate_mean_feature_1():   
    data = load_data()
    expected = 0.28867513497970554
    result = aggregate_mean(data, "sd")
    assert expected == result