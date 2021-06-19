# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:57:37 2021

@author: Yashi
"""

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy

# -1.2000000048873312
# -1.2000000011722805
# -1.200000005889354
file = "S_01_4003_VE.wav"

audio, sr = librosa.load(file, sr=22050)
librosa.display.waveplot(audio, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

#Shifting to frequency domain

fft = np.fft.fft(audio)

magnitude = np.abs(fft)
frequency = np.linspace(0,sr,len(magnitude))

left_frequency  = frequency[:int(len(frequency)/2)]
left_magnitude  = magnitude[:int(len(magnitude)/2)]

plt.plot(frequency,magnitude)
plt.xlabel(frequency)
plt.ylabel(magnitude)
plt.show()

kurt = scipy.stats.kurtosis(left_frequency)
freqs = np.mean(frequency)
