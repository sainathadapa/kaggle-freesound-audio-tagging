"""Compute Log Mel-Spectrograms"""

import os
import sys
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Number of cores
num_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 1

sampling_rate = 44100
train = pd.read_csv("./data/train.csv")
samp_subm = pd.read_csv("./data/sample_submission.csv")

# Removing null wavs
samp_subm['toremove'] = 0
samp_subm.loc[samp_subm.fname.isin([
    '0b0427e2.wav', '6ea0099f.wav', 'b39975f5.wav'
]), 'toremove'] = 1


def compute_melspec(filename, indir, outdir):
    wav = np.load(indir + filename + '.npy')
    wav = librosa.resample(wav, 44100, 22050)
    melspec = librosa.feature.melspectrogram(wav,
                                             sr=22050,
                                             n_fft=1764,
                                             hop_length=220,
                                             n_mels=64)
    logmel = librosa.core.power_to_db(melspec)
    np.save(outdir + filename + '.npy', logmel)


print('Train...')
os.makedirs('./data/mel_spec_train', exist_ok=True)
_ = Parallel(n_jobs=num_cores)(
    delayed(lambda x: compute_melspec(x, './data/audio_train_trim/', './data/mel_spec_train/'))(x)
    for x in tqdm(train.fname.values))


print('Test...')
os.makedirs('./data/mel_spec_test', exist_ok=True)
_ = Parallel(n_jobs=num_cores)(
    delayed(lambda x: compute_melspec(x, './data/audio_test_trim/', './data/mel_spec_test/'))(x)
    for x in tqdm(samp_subm.loc[lambda x: x.toremove == 0, :].fname.values))
