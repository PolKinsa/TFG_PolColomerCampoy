import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV

def cargar_audio(file_path):
    # Cargar archivo de audio
    fs, audio_data = wavfile.read(file_path)

    # Si el audio es estéreo, conviértelo a mono promediando los canales
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalizar los datos de audio
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))

    return audio_data

def ica(dades_audio):
    # Convertir las dades d'àudio en un format adequat per a l'algorisme de ICA
    X = np.array(dades_audio)

    # Definir el model de ICA
    model_ica = FastICA()

    # Definir la graella de paràmetres a buscar
    param_grid = {'n_components': range(2, 10)}  # Ajusta el rang segons calgui

    # Inicialitzar i ajustar el model utilitzant validació creuada
    cerca_graella = GridSearchCV(model_ica, param_grid, cv=5)  # Validació creuada amb 5 folds
    cerca_graella.fit(X.T)

    # Recuperar el millor model i aplicar-lo per separar les fonts
    millor_model_ica = cerca_graella.best_estimator_
    fonts_separades = millor_model_ica.fit_transform(X.T)

    return fonts_separades, millor_model_ica

def main():
    # Ruta del archivo de audio
    file_path = 'toyproblem2.wav'

    # Cargar archivo de audio
    audio_data = cargar_audio(file_path)

    # Aplicar ICA al archivo de audio
    fonts_separades, millor_model_ica = ica(audio_data)

    # Mostrar el número óptimo de componentes estimado
    num_components = millor_model_ica.n_components_
    print("Nombre òptim de components estimat:", num_components)

if __name__ == "__main__":
    main()


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