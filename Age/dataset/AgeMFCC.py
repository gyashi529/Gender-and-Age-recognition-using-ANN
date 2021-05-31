# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:28:35 2021

@author: Yashi
"""

import numpy as np
import scipy
import librosa 
import csv
import pandas as pd

ds = pd.read_csv('Age_loc_id_label.csv')
audio_files = ds['Column2']
audio_files = np.asarray(audio_files)

#START OF FUNCTION
def describe_freq(mfcc):
        
        d = dict();
        d['mfcc1'] = np.mean(mfcc[0])
        d['mfcc2'] = np.mean(mfcc[1])
        d['mfcc3'] = np.mean(mfcc[2])
        d['mfcc4'] = np.mean(mfcc[3])
        d['mfcc5'] = np.mean(mfcc[4])
        d['mfcc6'] = np.mean(mfcc[5])
        d['mfcc7'] = np.mean(mfcc[6])
        d['mfcc8'] = np.mean(mfcc[7])
        d['mfcc9'] = np.mean(mfcc[8])
        d['mfcc10'] = np.mean(mfcc[9])
        d['mfcc11'] = np.mean(mfcc[10])
        d['mfcc12'] = np.mean(mfcc[11])
        d['mfcc13'] = np.mean(mfcc[12])
        d['mfcc14'] = np.mean(mfcc[13])
        d['mfcc15'] = np.mean(mfcc[14])
        d['mfcc16'] = np.mean(mfcc[15])
        d['mfcc17'] = np.mean(mfcc[16])
        d['mfcc18'] = np.mean(mfcc[17]) 
        d['mfcc19'] = np.mean(mfcc[18])
        d['mfcc20'] = np.mean(mfcc[19])
    
        return d

# #END OF FUNCTION

file = open('ageMFCCFeatures.csv', 'a+', newline ='')

header = ['mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20']
with file:
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader()
    for audio in audio_files:
        x , sr = librosa.load(audio, sr=22050,mono=True, duration=30)
        x, index = librosa.effects.trim(x)
        mfcc = librosa.feature.mfcc(x)      
        result = describe_freq(mfcc)
        writer.writerow(result)
        
    