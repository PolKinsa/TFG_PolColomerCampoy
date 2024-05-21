
## Pseudocodigo de los pasos para generar datos:
1. **Selección de pistas simples**: Seleccionamos aleatoriamente `K` pistas simples de la lista de elementos simples proporcionados.

2. **Generación de valores aleatorios para α, β, V1 y V2**: Se generan valores aleatorios para cada una de las variables:
   - `alpha`: Factor de mezcla para la pista simple 1.
   - `beta`: Factor de mezcla para la pista simple 2.
   - `V1`: Volumen para la combinación de las pistas simples.
   - `V2`: Otro factor de volumen para la combinación de las pistas simples.

3. **Lectura de archivos de audio de las pistas simples seleccionadas**: Leemos los archivos de audio de las pistas simples seleccionadas.

4. **Combinación de pistas simples para generar el Toy Problem**:
   - Para cada muestra de audio, combinamos las pistas simples seleccionadas aplicando la fórmula dada: $$toy\_problem=\frac{1}{𝐾}\sum_{𝑖=1}^{𝐾} pista_i$$​
- Multiplicamos cada pista simple por los valores aleatorios α, β, V1 y V2 antes de combinarlas: $$ toy\_problem= \frac{1}{𝐾}(\sum_{𝑖=1}^{𝐾}  𝑉1∗(𝛼∗pista1_𝑖+𝛽∗pista2_𝑖))+(\sum_{𝑖=1}^{𝐾} 𝑉2∗((1−𝛼)∗pista1_𝑖+(1−𝛽)∗pista2_𝑖))$$
5. **Normalización del Toy Problem**: Normalizamos el Toy Problem dividiendo cada muestra de audio por `K`.

6. **Creación del nombre del archivo de salida**:
   - El nombre del archivo de salida incluye:
     - El número de pistas simples utilizadas (`K`).
     - Los nombres de las pistas simples seleccionadas.
     - Los valores de α, β, V1 y V2.

7. **Guardado del Toy Problem como archivo de audio**: Guardamos el Toy Problem generado como un archivo de audio .wav en la ruta especificada.

En resumen, el proceso toma `K` pistas simples, las combina según los valores aleatorios de α y β, y luego ajusta el volumen de la combinación resultante según los valores de V1 y V2. El archivo de audio resultante representa un Toy Problem generado a partir de las pistas simples seleccionadas.





# New version

1. **Selección de pistas simples**: Seleccionamos aleatoriamente dos elementos simples de la lista de elementos simples proporcionados.
    
2. **Generación de valores aleatorios para α, β, V1 y V2**:
    
    - `alpha`: Factor de mezcla para las dos pistas simples.
    - `beta`: Factor de mezcla para las dos pistas simples.
    - `V1`: Volumen para el canal izquierdo.
    - `V2`: Volumen para el canal derecho.
3. **Lectura de archivos de audio de las pistas simples seleccionadas**: Leemos los archivos de audio de las dos pistas simples seleccionadas.
    
4. **Combinación de pistas simples para generar el Toy Problem**:
    
    - Para cada muestra de audio, combinamos las dos pistas simples seleccionadas aplicando la fórmula dada: $$𝐿𝑒𝑓𝑡(𝐿)=𝑉1∗(𝛼∗𝐷1) + 𝑉2 * (𝛽∗𝐷2)$$$$ 𝑅𝑖𝑔ℎ𝑡(𝑅)=𝑉1∗((1-𝛼)∗𝐷1)+ 𝑉2 * ((1-𝛽)∗𝐷2)$$
    - Donde 𝐷1 y 𝐷2 representan las dos pistas simples seleccionadas.
5. **Creación del nombre del archivo de salida**:
    
    - El nombre del archivo de salida incluye:
        - Los nombres de las dos pistas simples seleccionadas.
        - Los valores de α, β, V1 y V2.
6. **Guardado del Toy Problem como archivo de audio**: Guardamos el Toy Problem generado como un archivo de audio .wav en la ruta especificada.