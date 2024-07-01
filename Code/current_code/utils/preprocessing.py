# root/utils/preprocessing.py
import numpy as np
import librosa

def preprocess_audio(audio, sr=44100, n_fft=2048, hop_length=512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude)
    return log_spectrogram, phase

def pad_spectrogram(spec, max_len):
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    else:
        spec = np.pad(spec, ((0, 0), (0, max_len - spec.shape[1])), mode='constant')
    return spec

def prepare_data(toy_problems, sr=44100, n_fft=2048, hop_length=512):
    inputs = []
    targets = []

    # Encontrar la longitud máxima de las STFTs
    max_len = 0
    for mixed, gt1, gt2 in toy_problems:
        mixed_spec, _ = preprocess_audio(mixed, sr, n_fft, hop_length)
        max_len = max(max_len, mixed_spec.shape[1])

    for mixed, gt1, gt2 in toy_problems:
        mixed_spec, _ = preprocess_audio(mixed, sr, n_fft, hop_length)
        gt1_spec, _ = preprocess_audio(gt1, sr, n_fft, hop_length)
        gt2_spec, _ = preprocess_audio(gt2, sr, n_fft, hop_length)

        # Rellenar los espectrogramas para que tengan la misma longitud
        mixed_spec = pad_spectrogram(mixed_spec, max_len)
        gt1_spec = pad_spectrogram(gt1_spec, max_len)
        gt2_spec = pad_spectrogram(gt2_spec, max_len)

        inputs.append(mixed_spec)
        targets.append(np.stack([gt1_spec, gt2_spec], axis=-1))

    inputs = np.expand_dims(np.array(inputs), axis=-1)
    targets = np.array(targets)
    return inputs, targets
