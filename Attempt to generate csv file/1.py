# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:00:12 2021

@author: Yashi
"""

import numpy as np
import scipy
from glob import glob
import librosa 

directory_path = 'E:/Major Project/Attempt to generate csv file/Audio'

audio_files = glob(directory_path + '/*.wav')

for file in range(0,len(audio_files),1):
    audio = librosa.load(audio_files[file],sr=16000,mono=True)
    #freqs = np.fft.fftfreq(audio.size)
