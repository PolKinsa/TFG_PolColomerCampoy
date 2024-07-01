import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import librosa
import os

from Code.tempDataGenerator import obtener_elementos_simples, generar_toy_problem


# Función para cargar y preprocesar los datos
def load_and_preprocess_data(data_dir, sample_rate=16000, n_fft=1024, hop_length=256):
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            # Cargar archivo de audio
            audio, _ = librosa.load(os.path.join(data_dir, filename), sr=sample_rate, mono=True)
            # Calcular el espectrograma del audio de entrada
            spectrogram = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            # Crear la máscara de referencia (pista de audio objetivo)
            mask = np.where(spectrogram > 0, 1, 0)
            X.append(spectrogram.T)
            y.append(mask.T)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Construir el modelo U-Net
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

    # Decoder
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Salida
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model



# Directorio que contiene los archivos de audio
for i in range(0, 10):
    # Directorio que contiene los archivos de SimpleElements
    simple_elements_dir = "../Data/SimpleElements"

    # Obtener la lista de archivos .wav en el directorio SimpleElements
    elementos_simples = obtener_elementos_simples(simple_elements_dir)
    # print("Elementos simples antes de entrar a funcion:", elementos_simples)
    # Generar un Toy Problem
    generar_toy_problem(elementos_simples)



data_dir = '../Data/ToyProblems'
X, y = load_and_preprocess_data(data_dir)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Construir el modelo
model = unet_model(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_val, y_val))

# Evaluar el modelo con datos de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss en datos de prueba: {loss}')
print(f'Precisión en datos de prueba: {accuracy}')
