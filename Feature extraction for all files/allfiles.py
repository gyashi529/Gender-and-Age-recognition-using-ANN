# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:31:28 2021

@author: Yashi
"""

import numpy as np
import scipy
import librosa 
import csv
directory_path = 'E:/Major Project/Attempt to generate csv file/Audio'
audio_files = librosa.util.find_files(directory_path, ext=['wav'])
audio_files = np.asarray(audio_files)
#START OF FUNCTION
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

file = open('datafiles.csv', 'a+', newline ='')

header = ['mean','sd','maxv','minv','median','skew','kurtosis','q1','q3','mode','iqr']
with file:
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader()
    for audio in audio_files:
        x , sr = librosa.load(audio)
        freqs = np.fft.fftfreq(x.size)
        result = describe_freq(freqs)
        writer.writerow(result)
        
    