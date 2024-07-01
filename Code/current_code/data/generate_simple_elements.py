import numpy as np
import soundfile as sf

sr = 44100  # Sample rate
t = np.linspace(0, 5, int(5 * sr), endpoint=False)  # 5 seconds length (similar to my simple_elements (aprox 5-6 sec)

# Sine waves
frequencies = [220, 440, 660, 880, 1100]  # Different frequencies for different elements
for i, freq in enumerate(frequencies):
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(f'simple_elements2/element_{i}.wav', signal, sr)
