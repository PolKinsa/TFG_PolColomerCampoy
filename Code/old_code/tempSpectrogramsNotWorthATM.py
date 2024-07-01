# for data transformation
import numpy as np
# for visualizing the data
import matplotlib.pyplot as plt
# for opening the media file
import scipy.io.wavfile as wavfile

Fs, aud = wavfile.read('../Data/SimpleElements/BitInvaderDiff.wav')

print(Fs)
print(aud.shape)
print("Max:", np.min(aud))
# select left channel only
aud = aud[:, 0]
print("After", min(aud), max(aud))
# trim the first 125 seconds
first = aud[:int(Fs*125)]

noise = np.random.normal(0, 0.001, first.shape[0])
signalNoise = first + noise
noised = first+32768 * np.random.randn(first.shape[0]) * 0.0000001
# powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(noised, Fs=Fs, NFFT=2**10, noverlap=2**9)
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(signalNoise, Fs=Fs, NFFT=2**10, noverlap=2**9)
plt.show()