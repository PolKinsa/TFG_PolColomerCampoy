# root/utils/data_generator.py
import numpy as np
import librosa
import os
import soundfile as sf

def verify_audio_format(audio, sr=44100):
    if audio.ndim > 1:
        raise ValueError("El audio debe estar en formato mono. Convertir a mono antes de continuar.")
    return audio

def load_audio(file_path, sr=44100, mono=True):
    y, _ = librosa.load(file_path, sr=sr, mono=mono)
    y = verify_audio_format(y, sr)
    return y

def create_toy_problem(element1, element2, alpha, beta, volume1, volume2, sr=44100):
    #element1 = element1 * volume1
    #element2 = element2 * volume2
    element1 = librosa.effects.time_stretch(element1, rate=alpha)
    element2 = librosa.effects.time_stretch(element2, rate=alpha)
    element1 = librosa.effects.pitch_shift(element1, sr=sr, n_steps=beta)
    element2 = librosa.effects.pitch_shift(element2, sr=sr, n_steps=beta)

    min_length = min(len(element1), len(element2))
    element1 = element1[:min_length]
    element2 = element2[:min_length]
    mixed = element1 + element2

    return mixed, element1, element2

def generate_toy_problems(simple_elements_dir, num_problems, sr=44100):
    simple_elements = [os.path.join(simple_elements_dir, f) for f in os.listdir(simple_elements_dir) if f.endswith('.wav')]
    if len(simple_elements) < 2:
        raise ValueError("Debe haber al menos dos elementos simples para generar toy problems.")

    toy_problems = []
    for i in range(num_problems):
        elem1, elem2 = np.random.choice(simple_elements, 2, replace=False)
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-2, 2)
        volume1 = np.random.uniform(0.5, 1.5)
        volume2 = np.random.uniform(0.5, 1.5)
        elem1_audio = load_audio(elem1, sr)
        elem2_audio = load_audio(elem2, sr)
        mixed, gt1, gt2 = create_toy_problem(elem1_audio, elem2_audio, alpha, beta, volume1, volume2, sr)
        toy_problems.append((mixed, gt1, gt2))
        sf.write(os.path.join('data/toy_problems', f'toy_problem_{i}.wav'), gt2, sr)
    return toy_problems
