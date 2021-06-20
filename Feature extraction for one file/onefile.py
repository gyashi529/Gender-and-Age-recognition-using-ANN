# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:07:37 2021

@author: Yashi
"""

import librosa
import numpy as np
import scipy
import csv

#function to calculate statistical features

def describe_freq(freqs):
        mean = np.mean(freqs)
        std = np.std(freqs) 
        maxv = np.amax(freqs) 
        minv = np.amin(freqs) 
        median = np.median(freqs)
        skew = scipy.stats.skew(freqs)
        kurt = scipy.stats.kurtosis(freqs)
        q1 = np.quantile(freqs, 0.25)
        q3 = np.quantile(freqs, 0.75)
        mode = scipy.stats.mode(freqs)[0][0]
        iqr = scipy.stats.iqr(freqs)
    
        d = dict();
        d['mean'] = mean
        d['sd'] = std
        d['maxv'] = maxv
        d['minv'] = minv
        d['median'] = median
        d['skew'] = skew
        d['kurtosis'] = kurt
        d['q1'] = q1
        d['q3'] = q3
        d['mode'] = mode
        d['iqr'] = iqr
    
    
    
        return d
#END OF FUNCTION    

path = 'E:/Major Project/operation for one file/sen.wav'
x , sr = librosa.load(path)
freqs = np.fft.fftfreq(x.size)
result = describe_freq(freqs)

#creating an excel sheet

header = ['mean','sd','maxv','minv','median','skew','kurtosis','q1','q3','mode','iqr']
file = open('data.csv', 'w', newline ='')
with file: 
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader() 
    writer.writerow(result)
