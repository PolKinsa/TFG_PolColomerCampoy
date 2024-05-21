https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3

https://towardsdatascience.com/understanding-u-net-61276b10f360


ML de una CNN con arquitectura U-Net en Python, necesitaríamos seguir estos pasos:

1. Preprocesamiento de datos: Convertir los archivos de audio .wav en espectrogramas para usarlos como entrada y salida para la red neuronal.
2. Construir el modelo U-Net.
3. Entrenar el modelo con los datos de entrenamiento.
4. Validar el modelo con datos de validación.
5. Evaluar el modelo con datos de prueba.

### Paso 1: Preprocesamiento de datos

**1.1 Cargar los datos de audio**

- Utilizamos la librería `librosa` para cargar los archivos de audio en formato `.wav` desde un directorio dado.
- Especificamos el `sample_rate`, que es la frecuencia de muestreo del audio.
- Convertimos los archivos de audio en espectrogramas utilizando la transformada de Fourier de tiempo corto (`librosa.stft`).

**1.2 Preprocesamiento de los espectrogramas**

- Normalizamos los espectrogramas.
- Creamos una máscara de referencia para cada espectrograma. Esta máscara se utilizará como la salida deseada de nuestro modelo.

### Paso 2: Construir el modelo U-Net

**2.1 Arquitectura del modelo U-Net**

- Utilizamos una arquitectura U-Net, que consta de un codificador (downsampling path) y un decodificador (upsampling path) para capturar características a diferentes escalas.

**2.2 Capas del modelo**

- Utilizamos capas convolucionales para extraer y combinar características.
- Se utiliza dropout para regularización y evitar el sobreajuste.
- La capa de salida utiliza una activación sigmoide para generar una máscara binaria que representa la pista de audio objetivo.

### Paso 3: Compilar el modelo

**3.1 Función de pérdida y optimizador**

- Utilizamos la función de pérdida de entropía cruzada binaria ya que estamos realizando una tarea de clasificación binaria.
- El optimizador `Adam` se utiliza para la optimización de la red.

### Paso 4: Entrenar el modelo

**4.1 Dividir los datos**

- Dividimos los datos en conjuntos de entrenamiento, validación y prueba utilizando `train_test_split`.

**4.2 Entrenamiento**

- Entrenamos el modelo utilizando los datos de entrenamiento y validación.
- Especificamos el tamaño del lote (`batch_size`) y el número de épocas (`epochs`).

### Paso 5: Evaluación del modelo

**5.1 Evaluación**

- Evaluamos el modelo utilizando los datos de prueba para ver cómo se desempeña en datos no vistos.
- Calculamos la pérdida y la precisión del modelo.