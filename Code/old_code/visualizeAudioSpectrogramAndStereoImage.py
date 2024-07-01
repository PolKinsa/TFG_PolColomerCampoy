import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

"""
Source: https://brechtcorbeelsaudioengineering.quora.com/https-www-quora-com-Can-you-provide-a-step-by-step-guide-on-creating-visual-representations-of-audio-data-such-as-spec?ch=10&oid=106691860&share=9cabdd9d&srid=u0uHe&target_type=post
"""
def load_audio(file_path, sample_rate=44100):
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=False)
    return audio, sr


def calculate_spectrogram(audio, sample_rate):
    # Convert stereo to mono
    mono_audio = librosa.to_mono(audio)
    # Compute the spectrogram
    stft = librosa.stft(mono_audio)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return spectrogram


def calculate_stereo_image(audio):
    left_channel = audio[0, :]
    right_channel = audio[1, :]
    stereo_image = left_channel - right_channel
    return stereo_image


def visualize_audio(audio, sample_rate):
    spectrogram = calculate_spectrogram(audio, sample_rate)
    stereo_image = calculate_stereo_image(audio)
    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    # Plot the stereo image
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(stereo_image, sr=sample_rate, axis="time", color="blue")
    plt.title("Stereo Image")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Starting . . . ")
    file_path = "../Data/SimpleElements/kicker.wav"
    audio, sample_rate = load_audio(file_path)
    visualize_audio(audio, sample_rate)
