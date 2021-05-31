# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:25:13 2021

@author: Yashi
"""

import numpy as np
import scipy
import librosa 
import csv
import pandas as pd

# ds = pd.read_csv('Age_loc_id_label.csv')
# audio_files = ds['Column2']
# audio_files = np.asarray(audio_files)

#START OF FUNCTION
def describe_freq(freqs,energy,rmse,zcr,tempo,meanmfcc,minmfcc,maxmfcc,chroma_freq,sc,sb,spectral_rolloff,tonnetz,stft,mel):
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
        d['energy'] = energy
        d['rmse'] = rmse
        d['zcr'] = zcr
        d['tempo'] = tempo
        d['meanmfcc'] = meanmfcc
        d['minmfcc'] = minmfcc
        d['maxmfcc'] = maxmfcc   
        d['chroma_freq'] = chroma_freq
        d['spectral_centroid'] = sc
        d['spectral_bandwidth'] = sb
        d['spectral_rolloff'] = spectral_rolloff
        d['tonnetz'] = tonnetz
        d['stft'] = stft
        d['mel'] = mel
    
        return d

# #END OF FUNCTION

file = open('ageFeatures2.csv', 'a+', newline ='')

header = ['mean','sd','maxv','minv','median','skew','kurtosis','q1','q3','mode','iqr','energy','rmse','zcr','tempo','meanmfcc','minmfcc','maxmfcc','chroma_freq','spectral_centroid','spectral_bandwidth','spectral_rolloff','tonnetz','stft','mel']
with file:
    writer = csv.DictWriter(file, fieldnames = header) 
    writer.writeheader()
    x , sr = librosa.load('E:/Major Project/cv-valid-train/cv-valid-train/sample-195774.mp3', sr=22050)
    fft = np.fft.fft(x)
    magnitude = np.abs(fft)
    freqs = np.linspace(0,sr,len(magnitude))
    freqs = freqs[:int(len(freqs)/2)]
    energy = np.sum(x**2)
    rmse = np.sqrt(np.mean(x**2))
    zcr = np.mean(librosa.zero_crossings(x, pad=False))
    tempo = librosa.beat.tempo(x)[0]
    mfcc = librosa.feature.mfcc(x)      
    meanmfcc=np.abs(np.mean(mfcc))
    minmfcc = np.abs(np.amin(mfcc))
    maxmfcc = np.amax(mfcc)
    chroma_freq = np.mean(librosa.feature.chroma_stft(x, sr))
    sc = np.mean(librosa.feature.spectral_centroid(x,sr))  
    sb = np.mean(librosa.feature.spectral_bandwidth(x,sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(x,sr))
    tonnetz = np.mean(librosa.feature.tonnetz(x, sr))
    stft = np.mean(np.abs(librosa.stft(x)))
    mel = np.mean(librosa.feature.melspectrogram(x, sr))
    result = describe_freq(freqs,energy,rmse,zcr,tempo,meanmfcc,minmfcc,maxmfcc,chroma_freq,sc,sb,spectral_rolloff,tonnetz,stft,mel)
    writer.writerow(result)
        
    