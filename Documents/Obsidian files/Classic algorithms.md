## Tipus d'algoritmes clàssics per triar
1. **Independent Component Analysis (ICA)**: És un mètode popular per a la separació de fonts en senyals mixtos. ICA assumeix que les observacions són una combinació lineal d'una sèrie de senyals independents, i intenta descompondre aquesta combinació per recuperar les fonts originals. Aquesta tècnica és útil per a la separació de pistes d'àudio quan les fonts són estadísticament independents.
    
2. **Principal Component Analysis (PCA)**: PCA és un altre algoritme comú utilitzat en l'àmbit de la separació de pistes d'àudio. L'objectiu de PCA és reduir la dimensionalitat d'un conjunt de dades mantenint la major part de la seva informació. En el context de la separació de pistes d'àudio, PCA pot ajudar a identificar les característiques més rellevants del senyal i facilitar-ne la separació.
    
3. **Non-negative Matrix Factorization (NMF)**: Aquest mètode és amplament utilitzat per a la separació de pistes d'àudio, especialment en aplicacions on les representacions negatives no tenen sentit (com ara en el cas de l'audio). NMF intenta descompondre una matriu de dades en dues matrius amb valors no negatius, que es poden interpretar com a components bàsics i coeficients d'activació.
    
4. **Filtres adaptatius**: A més dels mètodes anteriors, també puc considerar l'ús de filtres adaptatius, com ara els filtres de Wiener o els filtres de Kalman, per a la separació de pistes d'àudio. Aquests filtres poden adaptar-se de manera dinàmica al senyal i proporcionar una millora significativa en la qualitat de la separació.


## Dades a triar:

Format: 
- MP3: Hauria de funcionar correctament però he d'anar amb compte, pq mp3 comprimeix l'àudio. Sobretot les freqüències altes i les baixes i això em podria complicar la feina baixant la qualitat de la separació de pistes.
- FLAC o WAV: Els més recomanats doncs l'àudio és de més qualitat i s'eviten els problemes anteriors.
* **MIDI**: Els arxius MIDI ja estan en un format de text simple que pot ser llegit i interpretat fàcilment amb Python. També hi ha biblioteques com `mido` que faciliten la manipulació d'arxius MIDI en Python.


# ICA: Independent Component Analysis

## "Pseudocodi":
1. Inicialització:
    a. Normalitzar les dades d'entrada si és necessari.
    b. Inicialitzar aleatòriament una matriu de descomposició W.
    c. Definir el nombre màxim d'iteracions i un criteri de convergència.

2. Optimització:
    a. Mentre no s'hagi assolit el nombre màxim d'iteracions o el criteri de convergència:
        i. Aplicar la transformació W a les dades observades per obtenir les components estimades.
        ii. Calcular la funció de cost, que mesura la independència entre les components.
        iii. Actualitzar la matriu W mitjançant un algorisme d'optimització (com ara descens del gradient) per minimitzar la funció de cost.
        iv. Verificar si s'ha assolit el criteri de convergència.

3. Avaluació:
    a. Avaluar les components obtingudes mitjançant diverses mètriques d'independència, com ara simetria, curtosi, informació mutua, etc.

4. Retornar les components independents obtingudes.

Implementació de FastICA i exemple de visualització de dades
https://www.geeksforgeeks.org/blind-source-separation-using-fastica-in-scikit-learn/


Llibre on hi ha lo de la ICA en les pàg 270-274: https://theswissbay.ch/pdf/Gentoomen%20Library/Artificial%20Intelligence/ISR.Encyclopedia.Of.Artificial.Intelligence.Aug.2008.eBook-ELOHiM.pdf

Sound Separator: ICA Demonstration: https://github.com/ido90/SoundSeparator



## ICA Demonstration full implementaton (talk + music)

https://gowrishankar.info/blog/blind-source-separation-using-ica-a-practical-guide-to-separate-audio-signals/

https://github.com/ShawhinT/YouTube-Blog/blob/main/ica/pca_ica_livescript.pdf

https://towardsdatascience.com/principal-component-analysis-pca-79d228eb9d24

https://towardsdatascience.com/independent-component-analysis-ica-a3eba0ccec35