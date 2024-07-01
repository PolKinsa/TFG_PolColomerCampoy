import numpy as np
import random
import soundfile as sf
import os


def generar_toy_problem(elementos_simples, output_dir="../Data/ToyProblems"):
    # Seleccionar aleatoriamente dos elementos simples
    elemento_simple1, elemento_simple2 = random.sample(elementos_simples, 2)

    # Generar valores aleatorios para α, β, V1 y V2
    alpha = np.random.uniform(0, 1)
    beta = np.random.uniform(0, 1)
    V1 = np.random.uniform(0.5, 1.5)
    V2 = np.random.uniform(0.5, 1.5)

    # Leer los archivos de audio de los elementos simples
    audio1, sr = sf.read(elemento_simple1)
    audio2, sr = sf.read(elemento_simple2)
    print(audio1.shape)
    print(audio1)
    print(audio2.shape)
    # Combinar las pistas simples para generar el Toy Problem
    left_channel = V1 * (alpha * audio1[:,0]) + V2 * (beta * audio2[:,0])
    print("Left channel", left_channel)
    right_channel = V1 * ((1 - alpha) * audio1[:,0]) + V2 * ((1 - beta) * audio2[:,0])
    toy_problem = np.column_stack((left_channel, right_channel))
    print("Shape Toy Problem:", toy_problem.shape)
    # Crear el nombre del archivo de salida
    nombres_elementos = "_".join(
        [os.path.splitext(os.path.basename(elemento))[0] for elemento in [elemento_simple1, elemento_simple2]])
    output_filename = f"ToyProblem_{nombres_elementos}_alpha{alpha:.2f}_beta{beta:.2f}_V1{V1:.2f}_V2{V2:.2f}.wav"
    output_path = f"{output_dir}/{output_filename}"
    print("Nombres elementos: ", nombres_elementos)

    # Guardar el Toy Problem como archivo de audio
    sf.write(output_path, toy_problem, sr)


def obtener_elementos_simples(simple_elements_dir):
    # Obtener la lista de archivos .wav en el directorio SimpleElements
    elementos_simples = [os.path.normpath(os.path.join(simple_elements_dir, archivo)) for archivo in
                         os.listdir(simple_elements_dir) if archivo.endswith(".wav")]
    return elementos_simples


option = 0
if option == 0:
    # Directorio que contiene los archivos de SimpleElements
    simple_elements_dir = "../Data/SimpleElements"

    # Obtener la lista de archivos .wav en el directorio SimpleElements
    elementos_simples = obtener_elementos_simples(simple_elements_dir)
    print("Elementos simples antes de entrar a funcion:", elementos_simples)
    # Generar un Toy Problem
    generar_toy_problem(elementos_simples)
elif option == 1:
    # Ejemplo de uso
    elementos_simples = ["../Data/SimpleElements/BitInvader.wav", "../Data/SimpleElements/kicker.wav",
                         "../Data/SimpleElements/mallets.wav", "../Data/SimpleElements/organic.wav"]
    generar_toy_problem(elementos_simples)

elif option == 2:
    for i in range(0, 20):
        # Directorio que contiene los archivos de SimpleElements
        simple_elements_dir = "../Data/SimpleElements"

        # Obtener la lista de archivos .wav en el directorio SimpleElements
        elementos_simples = obtener_elementos_simples(simple_elements_dir)
        print("Elementos simples antes de entrar a funcion:", elementos_simples)
        # Generar un Toy Problem
        generar_toy_problem(elementos_simples)
