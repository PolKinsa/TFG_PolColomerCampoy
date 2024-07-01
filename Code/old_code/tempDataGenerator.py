import numpy as np
import random
import soundfile as sf
import os

def generar_toy_problem(elementos_simples, output_dir="../Data/ToyProblems"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Seleccionar aleatoriamente dos elementos simples
    elemento_simple1, elemento_simple2 = random.sample(elementos_simples, 2)

    # Generar valores aleatorios para α, β, V1 y V2
    alpha = np.random.uniform(0, 1)
    beta = np.random.uniform(0, 1)
    V1 = np.random.uniform(0.5, 1.5)
    V2 = np.random.uniform(0.5, 1.5)

    # Leer los archivos de audio de los elementos simples
    audio1, sr1 = sf.read(elemento_simple1)
    audio2, sr2 = sf.read(elemento_simple2)

    # Asegurarse de que ambos audios tienen la misma tasa de muestreo
    assert sr1 == sr2, "Sample rates of the two audio files must match."
    sr = sr1

    # Asegurarse de que ambos audios tienen el mismo número de canales (mono)
    if len(audio1.shape) > 1:
        audio1 = audio1[:, 0]
    if len(audio2.shape) > 1:
        audio2 = audio2[:, 0]

    # Asegurarse de que ambos audios tienen la misma longitud
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    # Combinar las pistas simples para generar el Toy Problem
    left_channel = V1 * (alpha * audio1) + V2 * (beta * audio2)
    right_channel = V1 * ((1 - alpha) * audio1) + V2 * ((1 - beta) * audio2)
    toy_problem = np.column_stack((left_channel, right_channel))

    # Crear el nombre del archivo de salida
    nombres_elementos = "_".join(
        [os.path.splitext(os.path.basename(elemento))[0] for elemento in [elemento_simple1, elemento_simple2]])
    output_filename = f"ToyProblem_{nombres_elementos}_alpha{alpha:.2f}_beta{beta:.2f}_V1{V1:.2f}_V2{V2:.2f}.wav"
    output_path = os.path.join(output_dir, output_filename)

    # Guardar el Toy Problem como archivo de audio
    sf.write(output_path, toy_problem, sr)
    print(f"Toy problem saved: {output_path}")

def obtener_elementos_simples(simple_elements_dir):
    # Obtener la lista de archivos .wav en el directorio SimpleElements
    elementos_simples = [os.path.normpath(os.path.join(simple_elements_dir, archivo)) for archivo in
                         os.listdir(simple_elements_dir) if archivo.endswith(".wav")]
    return elementos_simples


"""
# Ejemplo de uso
simple_elements_dir = '../Data/SimpleElements'
elementos_simples = obtener_elementos_simples(simple_elements_dir)

for _ in range(10):  # Generar 10 Toy Problems
    generar_toy_problem(elementos_simples)
"""