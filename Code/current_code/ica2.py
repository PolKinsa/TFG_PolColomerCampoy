# root/ica.py
import numpy as np
import librosa
import os
import soundfile as sf
from sklearn.decomposition import FastICA
from utils.data_generator import generate_toy_problems
from utils.preprocessing import preprocess_audio, pad_spectrogram, prepare_data


def apply_ica(mixed_spectrogram, n_components=2):
    """
    Aplica el algoritmo ICA al espectrograma mixto.
    :param mixed_spectrogram: Espectrograma mixto.
    :param n_components: Número de componentes independientes.
    :return: Componentes separadas.
    """
    ica = FastICA(n_components=n_components)
    separated_spectrograms = ica.fit_transform(mixed_spectrogram.T).T
    return separated_spectrograms


def save_audio_signals(signals, sr, output_dir, prefix):
    """
    Guarda las señales de audio en archivos .wav.
    :param signals: Señales de audio separadas.
    :param sr: Tasa de muestreo.
    :param output_dir: Directorio donde se guardarán los archivos.
    :param prefix: Prefijo para los nombres de archivo.
    """
    for i, signal in enumerate(signals):
        sf.write(os.path.join(output_dir, f'{prefix}_{i}.wav'), signal, sr)


def reconstruct_audio_from_spectrogram(spectrogram, phase, hop_length):
    """
    Reconstruye el audio a partir del espectrograma y la fase.
    :param spectrogram: Espectrograma de magnitud.
    :param phase: Información de fase.
    :param hop_length: Longitud de salto para la ISTFT.
    :return: Señal de audio reconstruida.
    """
    stft_matrix = spectrogram * phase
    audio = librosa.istft(stft_matrix, hop_length=hop_length)
    return audio


def main(simple_elements_dir, output_dir, num_problems=10, sr=44100, n_fft=2048, hop_length=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    toy_problems = generate_toy_problems(simple_elements_dir, num_problems, sr)
    inputs, targets = prepare_data(toy_problems, sr, n_fft, hop_length)

    for i, mixed_spec in enumerate(inputs):
        mixed_spec = mixed_spec.squeeze(axis=-1)
        mixed_stft = librosa.stft(mixed_spec.flatten(), n_fft=n_fft, hop_length=hop_length)
        mixed_magnitude, mixed_phase = librosa.magphase(mixed_stft)

        separated_spectrograms = apply_ica(np.abs(mixed_magnitude), n_components=2)

        separated_signals = []
        for j, separated_spec in enumerate(separated_spectrograms):
            separated_audio = reconstruct_audio_from_spectrogram(separated_spec, mixed_phase, hop_length)
            separated_signals.append(separated_audio)

        save_audio_signals(separated_signals, sr, output_dir, f'separated_{i}')

        # Guardar las señales ground truth para referencia
        gt1_spec, gt2_spec = targets[i, :, :, 0], targets[i, :, :, 1]
        gt1_audio = reconstruct_audio_from_spectrogram(librosa.db_to_amplitude(gt1_spec), mixed_phase, hop_length)
        gt2_audio = reconstruct_audio_from_spectrogram(librosa.db_to_amplitude(gt2_spec), mixed_phase, hop_length)

        sf.write(os.path.join(output_dir, f'gt1_{i}.wav'), gt1_audio, sr)
        sf.write(os.path.join(output_dir, f'gt2_{i}.wav'), gt2_audio, sr)
        print(f'Processed toy problem {i + 1}/{num_problems}')


if __name__ == '__main__':
    simple_elements_dir = 'data/simple_elements'
    output_dir = 'data/output/ica_results2'
    main(simple_elements_dir, output_dir)
