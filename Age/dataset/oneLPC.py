# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 17:05:30 2021

@author: Yashi
"""

import numpy as np
import scipy
import librosa 
import csv
import pandas as pd

audio = 'E:/Major Project/cv-valid-train/cv-valid-train/sample-000005.mp3'

x , sr = librosa.load(audio, sr=22050)
a = librosa.lpc(x, 2)
