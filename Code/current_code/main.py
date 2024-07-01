import os
import librosa
import numpy as np
from utils.data_generator import load_audio
from utils.preprocessing import preprocess_audio
from models.unet import unet_model
import tensorflow as tf


def predict_decomposition(model, audio, sr=44100, n_fft=2048, hop_length=512):
    audio_spec, phase = preprocess_audio(audio, sr, n_fft, hop_length)
    audio_spec = np.expand_dims(audio_spec, axis=(0, -1))
    pred_spec = model.predict(audio_spec)[0, :, :, 0]
    pred_audio = librosa.istft(librosa.db_to_amplitude(pred_spec) * phase, hop_length=hop_length)
    return pred_audio


def main():
    # Cargar modelo entrenado
    model = tf.keras.models.load_model('models/unet_model.h5')

    # Predecir descomposición de un nuevo audio
    new_audio = load_audio('data/simple_elements/new_mixed_audio.wav')
    predicted_decomposition = predict_decomposition(model, new_audio)

    # Guardar el resultado
    librosa.output.write_wav('data/output/predicted_decomposition.wav', predicted_decomposition, sr=44100)


if __name__ == '__main__':
    main()
