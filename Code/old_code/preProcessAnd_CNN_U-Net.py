import numpy as np
import os
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from Code.tempDataGenerator import *

def audio_to_spectrogram(audio_path, n_fft=1024, hop_length=256):
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    S_left = np.abs(librosa.stft(y[0], n_fft=n_fft, hop_length=hop_length))
    S_right = np.abs(librosa.stft(y[1], n_fft=n_fft, hop_length=hop_length))
    return S_left, S_right, sr

def load_data(data_dir):
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            S_left, S_right, sr = audio_to_spectrogram(os.path.join(data_dir, filename))
            X.append(S_left)
            y.append(S_right)
    X = np.array(X)
    y = np.array(y)
    return X, y


def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bridge
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # Decoder
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(drop5))
    up6 = layers.Cropping2D(cropping=((0, 0), (1, 0)))(up6)  # Ajustar dimensiones
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    up7 = layers.Cropping2D(cropping=((1, 0), (0, 0)))(up7)  # Ajustar dimensiones
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    up8 = layers.Cropping2D(cropping=((1, 1), (0, 0)))(up8)  # Ajustar dimensiones
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    up9 = layers.Cropping2D(cropping=((1, 1), (1, 0)))(up9)  # Ajustar dimensiones
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Salida
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


# Paso 1: Generar Toy Problems con el generator que hice (faltará actualizarlo para añadir ruido, desfases, etc)
simple_elements_dir = '../Data/SimpleElements'
elementos_simples = obtener_elementos_simples(simple_elements_dir)

for _ in range(10):  # Generar 10 Toy Problems
    generar_toy_problem(elementos_simples)


# Paso 2: Preprocesar los Datos
data_dir = '../Data/ToyProblems'
X, y = load_data(data_dir)
# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Añadir una dimensión adicional para las características de entrada y salida
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]  # Supongo que debería cambiar el método de conseguir los tests con un "generador
# de variaciones de los SimpleElements para adecuarlos a la pista del ToyProblem generado".
y_train = y_train[..., np.newaxis]
y_test = y_test[..., np.newaxis]

# Paso 3: Construir el Modelo U-Net
input_shape = (X_train.shape[1], X_train.shape[2], 1)
model = unet_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 4: Entrenar y Evaluar el Modelo
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss en datos de prueba: {loss}')
print(f'Precisión en datos de prueba: {accuracy}')
