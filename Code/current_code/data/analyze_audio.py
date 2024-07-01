import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# Cargar archivos de audio
gt1, sr = librosa.load('output/ica_results/gt1_5.wav', sr=None)
gt2, sr = librosa.load('output/ica_results/gt2_5.wav', sr=None)
separated1, sr = librosa.load('output/ica_results/separated1_5.wav', sr=None)
separated2, sr = librosa.load('output/ica_results/separated2_5.wav', sr=None)

# Imprimir formas de los datos
print(f"GT1 shape: {gt1.shape}")
print(f"GT2 shape: {gt2.shape}")
print(f"Separated1 shape: {separated1.shape}")
print(f"Separated2 shape: {separated2.shape}")

# Calcular espectrogramas
gt1_spec = librosa.amplitude_to_db(np.abs(librosa.stft(gt1)), ref=np.max)
gt2_spec = librosa.amplitude_to_db(np.abs(librosa.stft(gt2)), ref=np.max)
separated1_spec = librosa.amplitude_to_db(np.abs(librosa.stft(separated1)), ref=np.max)
separated2_spec = librosa.amplitude_to_db(np.abs(librosa.stft(separated2)), ref=np.max)

# Imprimir formas de los espectrogramas
print(f"GT1 spectrogram shape: {gt1_spec.shape}")
print(f"GT2 spectrogram shape: {gt2_spec.shape}")
print(f"Separated1 spectrogram shape: {separated1_spec.shape}")
print(f"Separated2 spectrogram shape: {separated2_spec.shape}")

# Visualizar espectrogramas
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
librosa.display.specshow(gt1_spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('GT1 Spectrogram')

plt.subplot(2, 2, 2)
librosa.display.specshow(gt2_spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('GT2 Spectrogram')

plt.subplot(2, 2, 3)
librosa.display.specshow(separated1_spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Separated1 Spectrogram')

plt.subplot(2, 2, 4)
librosa.display.specshow(separated2_spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Separated2 Spectrogram')

plt.tight_layout()
plt.show()
