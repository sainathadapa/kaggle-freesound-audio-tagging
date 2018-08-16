"""Trim leading and trailing silence"""

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


sampling_rate = 44100

train = pd.read_csv("./data/train.csv")
samp_subm = pd.read_csv("./data/sample_submission.csv")

# Ignoring the empty wavs
samp_subm['toremove'] = 0
samp_subm.loc[samp_subm.fname.isin([
    '0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav'
]), 'toremove'] = 1

print('Train...')
os.makedirs('./data/audio_train_trim', exist_ok=True)
for filename in tqdm(train.fname.values):
    x, sr = librosa.load('./data/audio_train/' + filename, sampling_rate)
    x = librosa.effects.trim(x)[0]
    np.save('./data/audio_train_trim/' + filename + '.npy', x)

print('Test...')
os.makedirs('./data/audio_test_trim', exist_ok=True)
for filename in tqdm(samp_subm.loc[lambda x: x.toremove == 0, :].fname.values):
    x, sr = librosa.load('./data/audio_test/' + filename, sampling_rate)
    x = librosa.effects.trim(x)[0]
    np.save('./data/audio_test_trim/' + filename + '.npy', x)
