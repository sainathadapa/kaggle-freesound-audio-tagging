import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

sr = 44100


def compute_summ_features(x):
    ans = np.hstack((
        np.mean(x, axis=1),
        np.std(x, axis=1),
        skew(x, axis=1),
        np.max(x, axis=1),
        np.min(x, axis=1)))
    return ans


def feat_set_1(x, stft):
    # Features mentioned in
    # - http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
    # - https://www.kaggle.com/amlanpraharaj/xgb-using-mfcc-opanichev-s-features-lb-0-811

    # Mel-scaled power spectrogram
    mels = librosa.feature.melspectrogram(x, sr=sr, S=stft)

    # Mel-frequency cepstral coefficients
    mfccs = librosa.feature.mfcc(y=x, sr=sr, S=stft, n_mfcc=40)

    # chorma-stft: Compute a chromagram from a waveform or power spectrogram
    chromas = librosa.feature.chroma_stft(S=stft, sr=sr)

    # spectral_contrast: Compute spectral contrast
    contrasts = librosa.feature.spectral_contrast(x, S=stft, sr=sr)

    # Compute roll-off frequency
    rolloffs = librosa.feature.spectral_rolloff(x, sr=sr, S=stft)

    # Compute the spectral centroid
    scentroids = librosa.feature.spectral_centroid(x, sr=sr, S=stft)

    # Compute p’th-order spectral bandwidth
    bandwidths = librosa.feature.spectral_bandwidth(x, sr=sr, S=stft)

    # tonnetz: Computes the tonal centroid features (tonnetz)
    tonnetzs = librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sr)

    # zero crossing rate
    zero_crossing_rates = librosa.feature.zero_crossing_rate(x)

    tmp = (mels, mfccs, chromas, contrasts,
           rolloffs, scentroids, bandwidths,
           tonnetzs, zero_crossing_rates)

    ans = np.hstack([
        compute_summ_features(x)
        for x in tmp
    ])

    return ans


# Features from https://www.kaggle.com/opanichev/lightgbm-baseline
def calc_part_features(data, n=2):
    ans = []
    for j, i in enumerate(range(0, len(data), len(data)//n)):
        if j == (n-1):
            i = len(data) - 1
        if j < n:
            ans.append(np.mean(data[i:i + len(data)//n]))
            ans.append(np.std(data[i:i + len(data)//n]))
            ans.append(np.min(data[i:i + len(data)//n]))
            ans.append(np.max(data[i:i + len(data)//n]))
    return ans


def feat_set_4(x):
    abs_data = np.abs(x)
    diff_data = np.diff(x)

    ans = []

    n = 1
    ans += calc_part_features(x, n=n)
    ans += calc_part_features(abs_data, n=n)
    ans += calc_part_features(diff_data, n=n)

    n = 2
    ans += calc_part_features(x, n=n)
    ans += calc_part_features(abs_data, n=n)
    ans += calc_part_features(diff_data, n=n)

    n = 3
    ans += calc_part_features(x, n=n)
    ans += calc_part_features(abs_data, n=n)
    ans += calc_part_features(diff_data, n=n)

    return np.array(ans)


# Features from https://www.kaggle.com/agehsbarg/audio-challenge-cnn-with-concatenated-inputs
def get_spectra_win(y, L, N):
    dft = np.fft.fft(y)
    fl = np.abs(dft)
    xf = np.arange(0.0, N/L, 1/L)
    return (xf, fl)


def get_spectra(signal, fs, M=1000, sM=500):

    N = signal.shape[0]
    ind = np.arange(100, N, M)

    spectra = []
    meanspectrum = np.repeat(0, M)

    for k in range(1, len(ind)):
        n1 = ind[k-1]
        n2 = ind[k]
        y = signal[n1:n2]
        L = (n2-n1)/fs
        N = n2-n1
        (xq, fq) = get_spectra_win(y, L, N)
        spectra.append(fq)

    spectra = pd.DataFrame(spectra)
    meanspectrum = spectra.apply(lambda x: np.log(1+np.mean(x)), axis=0)
    stdspectrum = spectra.apply(lambda x: np.log(1+np.std(x)), axis=0)

    meanspectrum = meanspectrum[0:sM]
    stdspectrum = stdspectrum[0:sM]

    return (meanspectrum, stdspectrum)


def get_width(w):
    if np.sum(w) == 0:
        return [0, 0, 0]
    else:
        z = np.diff(np.where(np.insert(np.append(w, 0), 0, 0) == 0))-1
        z = z[z > 0]
    return [np.log(1+np.mean(z)),
            np.log(1+np.std(z)),
            np.log(1+np.max(z)),
            len(z)]


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# predictors related to peaks
def num_peaks(x):
    x = np.array(x[0:len(x)])
    n10 = np.sum(x > 0.10*np.max(x))
    n20 = np.sum(x > 0.20*np.max(x))
    n50 = np.sum(x > 0.50*np.max(x))
    n90 = np.sum(x > 0.90*np.max(x))
    n99 = np.sum(x > 0.99*np.max(x))
    lead_min = np.min(np.where(x == np.max(x)))
    w10 = get_width(1*(x > 0.10*np.max(x)))
    w20 = get_width(1*(x > 0.20*np.max(x)))
    w50 = get_width(1*(x > 0.50*np.max(x)))
    w90 = get_width(1*(x > 0.90*np.max(x)))
    w99 = get_width(1*(x > 0.99*np.max(x)))
    W = w10+w20+w50+w90+w99

    f_sc = np.sum(np.arange(0, len(x))*(x*x)/np.sum(x*x))

    i1 = np.where(x < 0.10*np.max(x))[0]
    if i1.size == 0:
        lincoef_w = [0, 0, 0]
    else:
        a1 = i1[i1 < lead_min]
        a2 = i1[i1 > lead_min]

        if a1.size == 0:
            i1_left = 0
        else:
            i1_left = np.max(i1[i1 < lead_min])
        if a2.size == 0:
            i1_right = 0
        else:
            i1_right = np.min(i1[i1 > lead_min])

        lead_min_width = i1_right - i1_left
        if (lead_min_width > 2):
            poly_w = PolynomialFeatures(degree=2, include_bias=False)
            f_ind_w = poly_w.fit_transform(
                np.arange(i1_left, i1_right, 1).reshape(-1, 1))
            clf_w = linear_model.LinearRegression()
            linmodel_w = clf_w.fit(f_ind_w, np.array(x[i1_left:i1_right]))
            lincoef_w = list(linmodel_w.coef_)+[linmodel_w.intercept_]
        else:
            lincoef_w = [0, 0, 0]

    S = np.sum(x)
    S_n = np.sum(x)/len(x)
    S2 = np.sqrt(np.sum(x*x))
    S2_n = np.sqrt(np.sum(x*x))/len(x)
    integrals = [S, S_n, S2, S2_n]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    f_ind = poly.fit_transform(np.arange(0, len(x)).reshape(-1, 1))
    clf = linear_model.LinearRegression()
    linmodel = clf.fit(f_ind, x)
    lincoef_spectrum = list(linmodel.coef_)+[linmodel.intercept_]

    high_freq_sum_50 = np.sum(x[0:50] >= 0.5*np.max(x))
    high_freq_sum_90 = np.sum(x[0:50] >= 0.9*np.max(x))

    r = [f_sc, n10, n20, n50, n90, n99,
         lead_min, high_freq_sum_50, high_freq_sum_90] \
        + W + lincoef_spectrum + integrals + lincoef_w
    return r


def runningMeanFast(x, N=20):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def feat_set_2(x):
    rawsignal = x
    rawsignal_sq = rawsignal*rawsignal
    silenced = []
    sound = []
    attack = []
    for wd in [2000]:
        rawsignal_sq_rm = running_mean(rawsignal_sq, wd)
        w1 = 1*(rawsignal_sq_rm < 0.01*np.max(rawsignal_sq_rm))
        silenced = silenced + get_width(w1)
        w2 = 1*(rawsignal_sq_rm < 0.05*np.max(rawsignal_sq_rm))
        silenced = silenced + get_width(w2)
        w3 = 1*(rawsignal_sq_rm > 0.05*np.max(rawsignal_sq_rm))
        sound = sound + get_width(w3)
        w4 = 1*(rawsignal_sq_rm > 0.25*np.max(rawsignal_sq_rm))
        sound = sound + get_width(w4)
        time_to_attack = np.min(np.where(
            rawsignal_sq_rm > 0.99*np.max(rawsignal_sq_rm)))
        time_rel = np.where(rawsignal_sq_rm < 0.2*np.max(rawsignal_sq_rm))[0]
        if (time_rel.size == 0):
            time_to_relax = len(rawsignal_sq_rm)
        elif (time_rel[time_rel > time_to_attack].size == 0):
            time_to_relax = len(rawsignal_sq_rm)
        else:
            time_to_relax = np.min(time_rel[time_rel > time_to_attack])
        attack.append(np.log(1+time_to_attack))
        attack.append(np.log(1+time_to_relax))

    lr = len(rawsignal)
    zerocross_tot = np.log(
        1 + np.sum(
            np.array(
                rawsignal[0:(lr-1)]
            ) * np.array(rawsignal[1:lr]) <= 0))
    zerocross_prop = np.sum(
        np.array(
            rawsignal[0:(lr-1)]) * np.array(rawsignal[1:lr]) <= 0) / lr
    return np.array(sound + attack + [zerocross_tot, zerocross_prop])


def feat_set_3(x):
    (m, sd) = get_spectra(x, sr, 2000, 1000)
    ans1 = np.array(num_peaks(m))
    ans2 = (lambda x: x[np.arange(0, len(x), 40)])(np.array(runningMeanFast(m)))
    return np.concatenate((ans1, ans2))


def all_feats(filename):
    x = np.load(filename)
    stft = np.abs(librosa.stft(x))
    out1 = feat_set_1(x, stft=stft)
    out2 = feat_set_2(x)
    out3 = feat_set_3(x)
    out4 = feat_set_4(x)

    assert out1.shape[0] == 985
    assert out2.shape[0] == 12
    assert out3.shape[0] == 64
    assert out4.shape[0] == 72

    return np.concatenate((
        out1, out2, out3, out4
    ))
