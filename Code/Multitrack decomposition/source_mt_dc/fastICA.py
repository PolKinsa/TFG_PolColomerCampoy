import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

MIN_PISTES = 2
MAX_PISTES = 10


def cargar_audio(file_path):
    # Carregar l'arxiu d'àudio
    fs, audio_data = wavfile.read(file_path)

    # Si l'àudio és estéreo, convertir-lo a mono fent un promig dels canals. No passar a MONO
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalitzar les dades de l'àudio
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))

    return audio_data


def ica(dades_audio):
    # Convertir las dades d'àudio en un format adequat per a l'algorisme de ICA
    X = np.array(dades_audio)

    # Definir el model de ICA
    model_ica = FastICA()

    # Definir la graella de paràmetres a buscar
    param_grid = {'n_components': range(MIN_PISTES, MAX_PISTES)}  # Ajusta el rang segons calgui

    # Definir la métrica de puntuación personalizada
    def neg_mean_squared_error(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)

    # Utilizamos make_scorer para crear un objeto de puntuación a partir de la métrica personalizada
    neg_mse_scorer = make_scorer(neg_mean_squared_error)

    # Inicialitzar i ajustar el model utilitzant validació creuada
    cerca_graella = GridSearchCV(model_ica, param_grid, cv=5, scoring=neg_mse_scorer)  # Validació creuada amb 5 folds
    cerca_graella.fit(X.T)

    # Recuperar el millor model i aplicar-lo per separar les fonts
    millor_model_ica = cerca_graella.best_estimator_
    fonts_separades = millor_model_ica.fit_transform(X.T)

    return fonts_separades, millor_model_ica


def main():
    # Ruta del toy problem multitrack
    file_path = 'ToyProblem3.wav'

    # Carregar l'arxiu d'àudio mitjançant el mètode creat prèviament
    audio_data = cargar_audio(file_path)

    # Aplicar ICA a l'arxiu d'àudio
    fonts_separades, millor_model_ica = ica(audio_data)

    # Mostrar el número òptim de components estimat
    num_components = millor_model_ica.n_components_
    print("Nombre òptim de components estimat:", num_components)


if __name__ == "__main__":
    main()

"""def cargar_audio(file_path):
    # Carregar l'arxiu d'àudio
    fs, audio_data = wavfile.read(file_path)

    # Si l'àudio és estéreo, convertir-lo a mono fent un promig dels canals
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalitzar les dades de l'àudio
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))

    return audio_data"""

"""import librosa
from librosa.util import fix_length

def cargar_audio(file_path, max_length=None):
    # Cargar el archivo de audio usando librosa
    audio_data, fs = librosa.load(file_path, sr=None, mono=False)

    # Fijar todas las pistas de audio a la misma longitud
    if max_length is not None:
        audio_data = [fix_length(track, size=max_length) for track in audio_data]

    return audio_data, fs




def main():
    # Ruta del toy problem multitrack
    file_path = 'ToyProblem3.wav'

    # Carregar l'arxiu d'àudio mitjançant el mètode creat prèviament
    audio_data = cargar_audio(file_path)

    # Aplicar ICA a l'arxiu d'àudio
    fonts_separades, millor_model_ica = ica(audio_data)

    # Mostrar el número òptim de components estimat
    num_components = millor_model_ica.n_components_
    print("Nombre òptim de components estimat:", num_components)


if __name__ == "__main__":
    main()
"""
"""

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV

MIN_PISTES = 2
MAX_PISTES = 10
def ica(dades_audio):
    # Convertir les dades d'àudio en un format adequat per a l'algorisme de ICA
    X = np.array(dades_audio)

    # Definir el model de ICA
    model_ica = FastICA()

    # Definir la graella de paràmetres a buscar
    param_grid = {'n_components': range(MIN_PISTES, MAX_PISTES)}  # Ajusta el rang segons calgui

    # Inicialitzar i ajustar el model utilitzant cross-validation
    cerca_graella = GridSearchCV(model_ica, param_grid, cv=5)  # Validació creuada amb 5 folds
    cerca_graella.fit(X.T)

    # Recuperar el millor model i aplicar-lo per separar les fonts
    millor_model_ica = cerca_graella.best_estimator_
    fonts_separades = millor_model_ica.fit_transform(X.T)

    return fonts_separades, millor_model_ica


# Exemple d'ús normal. Amb aquesta implementació realitzada, permet a l'usuari introduir un àudio sense haver
# d'especificar el número de components en el que el vol separar, doncs mitjançant aquesta implementació el que
# s'intenta és evitar-ho mitjançant la utilització de cross-validation

# Suposem que 'dades_audio' és una matriu on cada fila representa una mostra d'àudio
# i cada columna representa una pista d'àudio en un fitxer multipista
fonts_separades, millor_model_ica = ica()
num_components = millor_model_ica.n_components_
print("Nombre òptim de components estimat:", num_components)
"""
