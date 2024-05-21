
## Punt 2. Selecció d'algoritmes i tècniques:

ICA multitrack audio decomposition: 
##### Score-Informed Audio Decomposition and Applications
	https://www.researchgate.net/publication/257873782_Score-Informed_Audio_Decomposition_and_Applications/link/57501b2c08ae5c7e547a89a4/download?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJwdWJsaWNhdGlvbiIsInByZXZpb3VzUGFnZSI6bnVsbH19

##### Using Score-Informed Constraints For NMF-Based Source Separation
	https://www.researchgate.net/publication/257873818_Using_Score-Informed_Constraints_For_NMF-Based_Source_Separation

##### Signal Processing for Music Analysis
	https://www.researchgate.net/publication/224217651_Signal_Processing_for_Music_Analysis

##### Algorithms for Non-negative Matrix Factorization
	https://www.researchgate.net/publication/2538030_Algorithms_for_Non-negative_Matrix_Factorization


## Databases:
#### MUSDB18:
Pàgina oficial amb el text per citar-lo: https://sigsep.github.io/datasets/musdb.html#sisec-2018-evaluation-campaign
	Exemple del que es pot fer: https://sisec18.unmix.app/#/unmix/The%20Sunshine%20Garcia%20Band%20-%20For%20I%20Am%20The%20Moon/REF
	Mètodes per separar per pistes utilitzant aquesta DB: https://sisec18.unmix.app/#/methods
Conferència sobre LVA i ICA: https://cvssp.org/events/lva-ica-2018/program/
ICA Papers d'aquesta conferència: 
* https://cvssp.org/events/lva-ica-2018/papers/#p58
* https://cvssp.org/events/lva-ica-2018/papers/#p18
* https://cvssp.org/events/lva-ica-2018/papers/#p40


Toy problem piano audio: https://freesound.org/people/alen_sound/sounds/328747/

## Acotar:

1.  Hacer el Documento en Latex. Usar para la bibliografia un un fichero .bib, que en los papers y tal se encuentra fácil en la parte de citar el código o texto concreto que hay que seguir para citar en LatEX.
2. Mirar i anar pujant coses al Notion pq ho vegi en Felipe també.
3. Anar fragmentant l'estudi del problema podent fer proves amb dos audios de dos instruments i anar afegint poc a poc fins a tenir una cançó gravade en estudi o finalment, la versió HARD que seria un concert amb soroll ambient i tal.
4. Fer el GANTT per posar les tasques i el temps que li dedicaré a cada tasca
5. Dades: Les que vinguin dels papers o amb les que pugui fer Toy problems per ajuntar pistes amb l'audacity. De les pistes d'audio simple es poden modificar els volums d'una i una altra, afegir soroll, etc.
6. Main idea: Ser capaz de separar en pistas un audio.
7. Mirar "Blind surce separation" -> Documentació del 2005, segurament haurà evolucionat, per tant, no mirar massa.
8. Fer notebooks per a cada cas o cada procés per poder fer els estudis. Fer un notebook per separació, un notebook per a obtenir les notes, i finalment un notebook per generar la partitura. Finalment un notebook general que seria el "producte final".


Metodologia:
- Reunions amb el Tutor
- Eines (Github, etc)
- Iteratiu: Toy problems, bla bla bla o més SCRUM type

Development: (Unes 5 pàgs aprox on 2 són resultats i 3 són com ho he fet)
- Tools 
- 
- Resultados















# CANVIS A FER:

1. Objectius tenen la seva secció pròpia. Afegir més nivells d'objectius en aquesta entrega, que formaran les tasques. En següents entregues ja es simplificarà
2. Després d'objectius, afegir estat de l'art
3. El mateix per a les tasques -> Fer diagrama
4. Afegir tasques de reunió i documentació
5. Reestructurar seguint la rubrica
6. Bibliografia i citar coses. Expandir bibliografia i donar context a les coses que expliqui mitjançant cites i bibliografia.









## ML and Music Sheets
https://colab.research.google.com/github/deezer/spleeter/blob/master/spleeter.ipynb#scrollTo=yarMHM64l14k
ML Separator: https://github.com/deezer/spleeter/tree/master?tab=readme-ov-file
https://github.com/dcyoung/pt-spleeter?tab=readme-ov-file

Transcription topic: https://github.com/topics/music-transcription
Omnizart: https://github.com/Music-and-Culture-Technology-Lab/omnizart?tab=readme-ov-file
https://music-and-culture-technology-lab.github.io/omnizart-doc/quick-start.html#colab

MIDI to Music Sheet: https://github.com/CPJKU/partitura
Howl's: https://www.youtube.com/watch?v=QCNVEsk3pcw&ab_channel=PianoMusicBros.
Diff instruments
https://www.mdpi.com/2076-3417/13/21/11882

https://archives.ismir.net/ismir2019/latebreaking/000036.pdf

### Procedural music composition with Python: Conté informació d'una llibreria per a transformar strings representant notes (C5, D4, etc) a partitura:
https://deepnote.com/app/essia/Procedural-music-composition-with-python-9b35ebd7-63e0-47bc-a3d5-c503954a083d
![[Partiture-Example.png]]



Explicació ML for Audio Processing: https://opensource.com/article/19/9/audio-processing-machine-learning-python


## TEST:

Google Colab: https://colab.research.google.com/drive/1W1kDEtN0w8kRLiUZLePwhXBxqFlJAA4x#scrollTo=O-YxojSStkE8

Link de la cançó amb partitura a generar: https://www.youtube.com/watch?v=aCUI6dNECeA&ab_channel=tocapartituras.com
![[screenshot-partitura-correcta.png]]


Para generar una partitura a partir de un archivo MIDI utilizando Python, puedes utilizar la biblioteca `music21`. Esta biblioteca te permite trabajar con archivos MIDI y convertirlos en objetos de partituras que luego puedes manipular y visualizar. Aquí tienes un ejemplo básico de cómo puedes hacerlo:

1. Asegúrate de tener `music21` instalado. Puedes instalarlo usando pip:

```
pip install music21
```

2. Luego, puedes utilizar el siguiente código para convertir un archivo MIDI en una partitura y visualizarla:

```python
from music21 import *

# Ruta al archivo MIDI de entrada
midi_file = 'tu_archivo.mid'

# Convertir el archivo MIDI en un objeto de partitura
score = converter.parse(midi_file)

# Visualizar la partitura
score.show()
```

Este código cargará el archivo MIDI especificado, lo convertirá en un objeto de partitura y luego lo mostrará en una ventana de visualización.

Además de simplemente mostrar la partitura, `music21` te permite realizar diversas operaciones sobre ella, como transponer, agregar dinámicas, realizar análisis, entre otras. Puedes explorar la documentación de `music21` para aprender más sobre estas funcionalidades y cómo aprovecharlas para tus necesidades específicas.


Link fórmula LaTeX - Data Generator: https://latexeditor.lagrida.com/
```Latex
\begin{matrix} \boxed{D1} (Element_{simple1}) \\ +  \\ \boxed{D2} (Element_{simple2}) \end{matrix}  \rightrightarrows \boxed{Toy Problem}=\begin{matrix} Left (L) \longrightarrow  V1 * (\alpha * D1) + V2* (\beta * D2) \\ Right (R) \longrightarrow V1 * ((1-\alpha) * D1) + V2* ((1-\beta) * D2) \end{matrix}
```


## TODO: 9/05/2024

A la visualització de l'espectrograma, canviar el stereo image per una visualització dels dos canals, ja que no aporta massa informació el stereo image proposat.

DATA LOADER AUMENTADOR SINTETIZADOR

Recibe simple elements y sintetiza como el generador hasta ahora pero puede tener una opción para activar el aumentador y que añada desfases, compresiones, ruido, etc.

Procesar como spectrogram, etc

La idea és que a l'hora de fer el Groundtruth s'ha d'anar amb compte pq el ground truth del Toy problem serà cada Simple Element utilitzat per generar el Toy Problem PERÒ TAMBÉ amb les variacions aplicades al Toy Problem, és a dir, amb el canvi de paneo, volums, soroll, i desfase.