import os
import numpy as np
import librosa
from sklearn.decomposition import FastICA
from utils.data_generator import generate_toy_problems
import soundfile as sf

def preprocess_audio_for_ica(audio, sr=44100, n_fft=2048, hop_length=512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    log_spectrogram = librosa.amplitude_to_db(magnitude)
    log_spectrogram_normalized = (log_spectrogram - np.mean(log_spectrogram)) / np.std(log_spectrogram)
    print(f"Preprocess - log_spectrogram shape: {log_spectrogram.shape}, phase shape: {phase.shape}")
    return log_spectrogram_normalized, phase, stft.shape

def reconstruct_audio_from_spectrogram(spectrogram, phase, original_shape, sr=44100, n_fft=2048, hop_length=512):
    amplitude = librosa.db_to_amplitude(spectrogram * np.std(spectrogram) + np.mean(spectrogram))
    if amplitude.shape[1] < original_shape[1]:
        amplitude = np.pad(amplitude, ((0, 0), (0, original_shape[1] - amplitude.shape[1])), mode='constant')
    stft_matrix = amplitude * phase
    audio = librosa.istft(stft_matrix, hop_length=hop_length)
    print(f"Reconstruct - amplitude shape: {amplitude.shape}, stft_matrix shape: {stft_matrix.shape}, audio shape: {audio.shape}")
    return audio

def apply_ica(mixed_signals):
    print(f"Apply ICA - mixed_signals shape: {mixed_signals.shape}")
    ica = FastICA(n_components=2, max_iter=1000)
    separated_signals = ica.fit_transform(mixed_signals)
    print(f"Apply ICA - separated_signals shape: {separated_signals.shape}")
    return separated_signals

def ica_decomposition(toy_problems, sr=44100, n_fft=2048, hop_length=512):
    results = []
    for idx, (mixed, gt1, gt2) in enumerate(toy_problems):
        mixed_spec, phase, original_shape = preprocess_audio_for_ica(mixed, sr, n_fft, hop_length)
        mixed_spec = mixed_spec.T  # Transpose to have shape (n_samples, n_features)

        print(f"Toy Problem {idx} - Original mixed_spec shape: {mixed_spec.shape}")

        # Apply ICA
        separated_specs = apply_ica(mixed_spec)

        # Reshape the separated spectrograms to original dimensions
        separated_spec1 = separated_specs[:, 0].reshape(original_shape[1], -1)
        separated_spec2 = separated_specs[:, 1].reshape(original_shape[1], -1)

        print(f"Toy Problem {idx} - separated_spec1 shape after reshape: {separated_spec1.shape}")
        print(f"Toy Problem {idx} - separated_spec2 shape after reshape: {separated_spec2.shape}")

        # Reconstruct audio from spectrograms
        separated_audio1 = reconstruct_audio_from_spectrogram(separated_spec1.T, phase, original_shape, sr, n_fft, hop_length)
        separated_audio2 = reconstruct_audio_from_spectrogram(separated_spec2.T, phase, original_shape, sr, n_fft, hop_length)

        # Normalizar audio separado
        separated_audio1 = librosa.util.normalize(separated_audio1)
        separated_audio2 = librosa.util.normalize(separated_audio2)

        print(f"Toy Problem {idx} - separated_audio1 shape: {separated_audio1.shape}, separated_audio2 shape: {separated_audio2.shape}")

        results.append((separated_audio1, separated_audio2, gt1, gt2))
    return results

def main():
    print("Starting . . .")
    # Generar Toy Problems
    simple_elements_dir = 'data/simple_elements'
    toy_problems = generate_toy_problems(simple_elements_dir, num_problems=15)
    sr = 44100
    print("Toy problems generated, starting ICA . . .")
    # Aplicar ICA
    ica_results = ica_decomposition(toy_problems)

    # Guardar resultados
    output_dir = 'data/output/ica_results'
    os.makedirs(output_dir, exist_ok=True)
    for i, (separated1, separated2, gt1, gt2) in enumerate(ica_results):
        sf.write(os.path.join(output_dir, f'separated1_{i}.wav'), separated1, sr)
        sf.write(os.path.join(output_dir, f'separated2_{i}.wav'), separated2, sr)
        sf.write(os.path.join(output_dir, f'gt1_{i}.wav'), gt1, sr)
        sf.write(os.path.join(output_dir, f'gt2_{i}.wav'), gt2, sr)

if __name__ == '__main__':
    main()
