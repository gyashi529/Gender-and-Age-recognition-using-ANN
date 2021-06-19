# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:40:19 2021

@author: Yashi
"""

import numpy as np
import scipy
import librosa 
import csv
directory_path = 'E:/Major Project/Trying with both languages/Audio VV'
audio_files = librosa.util.find_files(directory_path, ext=['wav'])
# audio_files = np.asarray(audio_files)
# #START OF FUNCTION
# def describe_freq(freqs,energy,rmse,zcr,tempo,meanmfcc,minmfcc,maxmfcc,chroma_freq,sc,sb,spectral_rolloff):
#         mean = np.mean(freqs)
#         std = np.std(freqs) 
#         maxv = np.amax(freqs) 
#         minv = np.amin(freqs) 
#         median = np.median(freqs)
#         skew = scipy.stats.skew(freqs)
#         kurt = scipy.stats.kurtosis(freqs)
#         q1 = np.quantile(freqs, 0.25)
#         q3 = np.quantile(freqs, 0.75)
#         mode = scipy.stats.mode(freqs)[0][0]
#         iqr = scipy.stats.iqr(freqs)
#         d = dict();
#         d['mean'] = mean
#         d['sd'] = std
#         d['maxv'] = maxv
#         d['minv'] = minv
#         d['median'] = median
#         d['skew'] = skew
#         d['kurtosis'] = kurt
#         d['q1'] = q1
#         d['q3'] = q3
#         d['mode'] = mode
#         d['iqr'] = iqr 
#         d['energy'] = energy
#         d['rmse'] = rmse
#         d['zcr'] = zcr
#         d['tempo'] = tempo
#         d['meanmfcc'] = meanmfcc
#         d['minmfcc'] = minmfcc
#         d['maxmfcc'] = maxmfcc   
#         d['chroma_freq'] = chroma_freq
#         d['spectral_centroid'] = sc
#         d['spectral_bandwidth'] = sb
#         d['spectral_rolloff'] = spectral_rolloff
    
#         return d

# #END OF FUNCTION

# file = open('dataset.csv', 'a+', newline ='')

# header = ['mean','sd','maxv','minv','median','skew','kurtosis','q1','q3','mode','iqr','energy','rmse','zcr','tempo','meanmfcc','minmfcc','maxmfcc','chroma_freq','spectral_centroid','spectral_bandwidth','spectral_rolloff']
# with file:
#     writer = csv.DictWriter(file, fieldnames = header) 
#     writer.writeheader()
#     for audio in audio_files:
#         x , sr = librosa.load(audio)
#         energy = np.sum(x**2)
#         rmse = np.sqrt(np.mean(x**2))
#         zcr = sum(librosa.zero_crossings(x, pad=False))
#         tempo = librosa.beat.tempo(x)[0]
#         mfcc = librosa.feature.mfcc(x)
#         meanmfcc=np.mean(mfcc)
#         minmfcc = np.amin(mfcc)
#         maxmfcc = np.amax(mfcc)
#         chroma_freq = np.mean(librosa.feature.chroma_stft(x, sr))
#         sc = np.mean(librosa.feature.spectral_centroid(x,sr))  
#         sb = np.mean(librosa.feature.spectral_bandwidth(x,sr))
#         spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(x,sr))
#         freqs = np.fft.fftfreq(x.size)
#         result = describe_freq(freqs,energy,rmse,zcr,tempo,meanmfcc,minmfcc,maxmfcc,chroma_freq,sc,sb,spectral_rolloff)
#         writer.writerow(result)
        
    