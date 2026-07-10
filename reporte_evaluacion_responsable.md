# Evaluación responsable de un prototipo multimodal para patrones textiles andinos generados

## 1. Proyecto personal y tarea multimodal

El proyecto desarrollado para el curso MCC225 consiste en evaluar de manera responsable un prototipo multimodal orientado a la recuperación imagen texto sobre patrones textiles andinos generados. El objetivo no es construir un sistema definitivo de reconocimiento visual, sino analizar si un modelo multimodal puede asociar imágenes generadas con descripciones textuales controladas.

El problema abordado es la dificultad de evaluar sistemas multimodales solo a partir de ejemplos visuales exitosos. Un modelo puede mostrar resultados aparentemente correctos en algunos casos, pero fallar cuando cambia la redacción de la consulta, cuando existen patrones parecidos, cuando la imagen es ambigua o cuando la descripción textual no contiene suficiente información. Por esta razón, la actividad se enfoca en medir resultados, observar errores, revisar confiabilidad y establecer límites de uso.

El dominio de aplicación corresponde al análisis exploratorio de imágenes generadas de inspiración textil andina. Las modalidades consideradas son imagen y texto. La tarea multimodal evaluada será principalmente recuperación imagen texto. En esta tarea, dada una imagen, el sistema debe recuperar la descripción textual más adecuada entre varias opciones candidatas. De manera complementaria, se podrá revisar la recuperación texto imagen, donde una descripción se usa para buscar la imagen correspondiente.

La evaluación se realizará con un conjunto pequeño y controlado de imágenes generadas, acompañadas de captions o descripciones construidas manualmente. Estas descripciones incluirán atributos visuales observables como colores dominantes, presencia de bandas, motivos geométricos, composición, simetría y repetición modular.

El usuario previsto es un estudiante o evaluador académico que desea explorar el comportamiento de modelos multimodales en un contexto controlado. El sistema no está diseñado para realizar identificación cultural, clasificación histórica ni validación patrimonial. Su uso debe entenderse como una evaluación técnica preliminar de alineamiento entre imagen y texto.

El riesgo principal del sistema es interpretar una coincidencia visual como si fuera una identificación confiable. Por ello, la evaluación debe reportar no solo los aciertos, sino también los errores, los casos ambiguos, la sensibilidad del sistema y las limitaciones del conjunto de datos utilizado.

## Ficha mínima del proyecto

| Elemento | Respuesta |
|---|---|
| Nombre del proyecto | Evaluación responsable de un prototipo multimodal para patrones textiles andinos generados |
| Problema que aborda | Evaluar si un modelo multimodal puede alinear imágenes generadas de inspiración textil con descripciones textuales controladas, identificando aciertos, errores y limitaciones |
| Dominio de aplicación | Evaluación multimodal, recuperación imagen texto y análisis visual exploratorio de patrones textiles |
| Modalidades usadas | Imagen y texto |
| Tarea multimodal evaluada | Recuperación imagen texto y, de forma complementaria, recuperación texto imagen |
| Usuario previsto | Estudiante, investigador o evaluador académico en un entorno experimental controlado |
| Riesgo principal del sistema | Confundir similitud visual con identificación confiable o clasificación real del objeto representado |

## Pregunta breve

¿Qué parte del proyecto se puede evaluar con evidencia en esta etapa del curso?

En esta etapa se puede evaluar con evidencia el alineamiento entre imágenes generadas y descripciones textuales controladas. Esto incluye medir si el modelo recupera el texto correcto para una imagen, comparar su desempeño contra un baseline, analizar casos de acierto y error, y observar si el comportamiento se mantiene ante variaciones pequeñas en las consultas o en las imágenes. No se evaluará autenticidad, procedencia histórica ni clasificación patrimonial.

## 2. Adaptación del Cuaderno 14

El Cuaderno 14 se utilizará como base experimental para organizar la evaluación multimodal. Su función principal será permitir la carga del conjunto de imágenes y textos, calcular representaciones multimodales, obtener métricas de recuperación, comparar el sistema contra un baseline y generar evidencia para el análisis posterior.

Para este proyecto, el conjunto de datos original será reemplazado por un manifiesto local compuesto por imágenes generadas de inspiración textil andina y descripciones textuales construidas manualmente. El objetivo de esta adaptación es trabajar con un escenario controlado, donde sea posible observar si el modelo logra asociar atributos visuales de las imágenes con los textos correspondientes.

El modelo previsto para la evaluación será CLIP u OpenCLIP, debido a que permite representar imágenes y textos en un mismo espacio vectorial. A partir de estas representaciones, se calculará la similitud entre cada imagen y cada descripción textual. La tarea principal será recuperación imagen texto, donde el sistema recibe una imagen y debe ubicar la descripción correcta entre un conjunto de textos candidatos. De forma complementaria, se podrá revisar la recuperación texto imagen.

El dataset estará compuesto por un conjunto pequeño de imágenes generadas, acompañado por captions que describen atributos visuales observables. Entre estos atributos se incluirán paleta de colores, composición, presencia de bandas, motivos geométricos, simetría y repetición modular. Esta decisión permite evaluar el alineamiento multimodal sin asumir que el sistema realiza identificación cultural o clasificación real de objetos.

Las métricas principales serán Recall@1, Recall@5 y una métrica complementaria como MRR o CLIPScore promedio. Recall@1 permitirá observar si el texto correcto aparece como primera opción. Recall@5 permitirá evaluar si el texto correcto aparece dentro de los primeros resultados. La métrica complementaria ayudará a revisar la posición relativa o la similitud promedio entre imagen y texto.

También se incluirá un baseline simple. Este baseline podrá construirse mediante captions desplazados o mediante un ranking aleatorio. La comparación contra este baseline es necesaria para saber si el modelo muestra una señal real de alineamiento o si los resultados podrían explicarse por azar o por coincidencias débiles del conjunto evaluado.

La adaptación del cuaderno no se limitará a ejecutar el modelo. También deberá producir archivos de salida en el repositorio, incluyendo una tabla de métricas, una tabla de casos analizados, resultados de pruebas de confiabilidad, una tabla de explicabilidad y una figura con ejemplos evaluados. Esto permitirá que la evaluación sea revisable y reproducible.

## Ficha de adaptación del Cuaderno 14

| Elemento | Respuesta |
|---|---|
| Modelo usado | CLIP u OpenCLIP |
| Dataset o subconjunto | Imágenes generadas de inspiración textil andina con descripciones textuales controladas |
| Número de ejemplos | 40 imágenes generadas, cada una con cinco descripciones textuales |
| Métricas usadas | Recall@1, Recall@5 y MRR o CLIPScore promedio |
| Baseline usado | Captions desplazados o ranking aleatorio |
| Hardware o entorno | Computadora personal o entorno Colab, según disponibilidad |
| Parte del Cuaderno 14 reutilizada | Carga de datos, cálculo de embeddings, matriz de similitud, métricas, baseline, análisis de errores y figuras |
| Parte modificada para el proyecto | Reemplazo del dataset original por un manifiesto local de imágenes generadas y captions descriptivos |

## Pregunta breve

¿Qué cambio fue necesario para adaptar el Cuaderno 14 al proyecto personal?

El cambio principal fue reemplazar el dataset original por un conjunto local de imágenes generadas y descripciones construidas manualmente. Además, se ajustó la interpretación de la evaluación para que los resultados se entiendan como evidencia de alineamiento imagen texto en un escenario controlado, no como reconocimiento confiable ni clasificación real de objetos culturales.

## 3. Construcción del dataset

Para la evaluación se construyó un dataset pequeño y controlado de imágenes generadas. El conjunto está compuesto por 40 imágenes sintéticas inspiradas en patrones textiles andinos. Esta cantidad permite cumplir con el tamaño mínimo esperado para la actividad, manteniendo un volumen manejable para el análisis cuantitativo y cualitativo.

Las imágenes fueron generadas mediante un script local, lo que permite reproducir el conjunto de datos desde el repositorio. Cada imagen se almacena en la carpeta `data/images/` y se registra en el archivo `data/manifest_textiles_generados.csv`.

El manifiesto incluye cinco descripciones textuales por imagen. Estas descripciones no buscan ser interpretaciones culturales, sino captions visuales construidos a partir de atributos observables. Entre los atributos considerados se encuentran la paleta cromática, la composición, el tipo de motivo geométrico, la simetría y el nivel de ambigüedad.

La decisión de usar imágenes generadas responde a la necesidad de trabajar en un entorno controlado. En este escenario, es posible definir de antemano qué atributos visuales deberían ser reconocibles por el modelo y analizar si las descripciones asociadas son recuperadas correctamente. Esto permite evaluar el alineamiento imagen texto sin asumir que el sistema realiza identificación cultural ni clasificación real de objetos.

El dataset incluye variaciones en la composición visual. Se consideran patrones con bandas horizontales, bandas verticales, rombos, grecas escalonadas, retículas geométricas y composiciones mixtas. También se incorporan casos con distintos niveles de ambigüedad para observar si el sistema mantiene un comportamiento estable cuando las imágenes no tienen una lectura visual única.

La generación del dataset queda documentada en el script `scripts/generar_dataset_sintetico.py`. Esta decisión permite que el experimento sea trazable, revisable y reproducible.

## Ficha del dataset

| Elemento | Descripción |
|---|---|
| Nombre del dataset | Textiles generados para evaluación multimodal |
| Tipo de datos | Imágenes generadas y descripciones textuales controladas |
| Número de imágenes | 40 |
| Descripciones por imagen | 5 |
| Total de descripciones | 200 |
| Modalidades | Imagen y texto |
| Carpeta de imágenes | `data/images/` |
| Manifiesto | `data/manifest_textiles_generados.csv` |
| Método de generación | Script local reproducible |
| Uso previsto | Evaluación académica controlada de recuperación imagen texto |
| Limitación principal | No representa piezas reales ni permite identificación cultural |

## 4. Evaluación experimental inicial

La evaluación experimental se realizó a partir del manifiesto de imágenes generadas y captions controlados. Cada imagen fue comparada contra el conjunto completo de textos candidatos. Para ello, se utilizó OpenCLIP, que permite representar imágenes y textos en un mismo espacio vectorial.

El procedimiento consistió en cargar las imágenes, preprocesarlas, calcular sus embeddings visuales y luego calcular embeddings para todos los textos candidatos. A partir de estos vectores se obtuvo una matriz de similitud entre imágenes y textos. Esta matriz permitió ordenar los captions candidatos para cada imagen y evaluar si alguno de los textos asociados a la imagen aparecía entre los primeros resultados.

La tarea principal evaluada fue recuperación imagen texto. En esta tarea, una consulta visual debe recuperar la descripción textual correspondiente. Debido a que cada imagen cuenta con cinco captions válidos, se consideró correcto el resultado cuando cualquiera de esos captions aparecía dentro de las primeras posiciones del ranking.

Las métricas utilizadas fueron Recall@1, Recall@5, Recall@10 y MRR. Recall@1 mide si el sistema ubica un caption correcto en la primera posición. Recall@5 y Recall@10 permiten observar si el sistema recupera algún texto correcto dentro de los primeros resultados. MRR permite revisar la posición promedio del primer resultado correcto.

También se calculó un baseline aleatorio. Este baseline permite estimar qué desempeño se obtendría si los textos se ordenaran sin utilizar información visual ni textual significativa. La comparación contra este baseline ayuda a distinguir si el modelo muestra una señal de alineamiento o si el resultado podría explicarse por azar.

Los resultados se exportaron en archivos CSV y JSON dentro de la carpeta `results/`. Además, se generó una matriz de similitud agrupada por imagen, almacenada en `figures/matriz_similitud_agrupada.png`. Esta figura permite revisar visualmente si las imágenes tienden a asociarse con sus propios grupos de captions o si existen confusiones frecuentes entre imágenes distintas.

## 5. Resultados cuantitativos

La evaluación cuantitativa se realizó sobre 40 imágenes generadas y 200 descripciones textuales candidatas. Cada imagen tuvo cinco captions válidos asociados. Por ello, en la evaluación de recuperación imagen texto se consideró correcto el resultado cuando cualquiera de los cinco captions asociados a la imagen aparecía dentro de las primeras posiciones del ranking.

La tabla siguiente resume los principales resultados obtenidos por el modelo y por el baseline aleatorio.

| Experimento | Modelo | Datos | Métrica | Resultado | Interpretación breve |
|---|---|---:|---|---:|---|
| E1 | OpenCLIP ViT-B-32 | 40 imágenes y 200 textos | Recall@1 | completar | Evalúa si un caption correcto aparece en la primera posición |
| E1 | OpenCLIP ViT-B-32 | 40 imágenes y 200 textos | Recall@5 | completar | Evalúa si un caption correcto aparece entre los cinco primeros resultados |
| E1 | OpenCLIP ViT-B-32 | 40 imágenes y 200 textos | MRR | completar | Evalúa qué tan arriba aparece el primer caption correcto |
| E2 | Baseline aleatorio | 40 imágenes y 200 textos | Recall@1 | completar | Estima el desempeño esperado sin señal multimodal |
| E2 | Baseline aleatorio | 40 imágenes y 200 textos | Recall@5 | completar | Permite comparar si el modelo mejora frente a un ordenamiento aleatorio |
| E2 | Baseline aleatorio | 40 imágenes y 200 textos | MRR | completar | Permite comparar la posición promedio del primer acierto frente al azar |

### Interpretación general

Los resultados permiten revisar si el modelo muestra una señal de alineamiento entre imágenes generadas y descripciones textuales. Si Recall@1 es superior al baseline, se puede afirmar que el modelo logra ubicar captions correctos en la primera posición con mayor frecuencia que un ordenamiento aleatorio. Si Recall@5 también mejora frente al baseline, se observa que el modelo puede recuperar descripciones relacionadas aunque no siempre las ubique en el primer lugar.

Sin embargo, estos resultados deben interpretarse con cautela. Un buen desempeño en este conjunto no significa que el sistema comprenda el significado cultural de los patrones ni que pueda clasificar objetos reales. La evaluación solo mide alineamiento imagen texto en un entorno controlado, con imágenes generadas y captions construidos para el experimento.

### Pregunta 1

¿Qué métrica representa mejor el objetivo de la tarea?

La métrica que representa mejor el objetivo de la tarea es Recall@1, porque la tarea principal consiste en recuperar la descripción textual más adecuada para una imagen. Si el caption correcto aparece en la primera posición, el sistema logra resolver el caso de manera estricta.

### Pregunta 2

¿Qué métrica puede ser insuficiente si se interpreta sola?

Recall@5 puede ser insuficiente si se interpreta sola. Esta métrica permite saber si un caption correcto aparece entre los primeros cinco resultados, pero no indica si el sistema ubicó el texto correcto como primera opción. Un sistema podría tener buen Recall@5 y aun así presentar dudas en la selección final.

### Pregunta 3

¿El sistema supera claramente al baseline o solo muestra funcionamiento parcial?

La respuesta depende de la diferencia observada entre OpenCLIP y el baseline. Si el modelo supera al baseline en Recall@1, Recall@5 y MRR, se puede afirmar que existe una señal de alineamiento multimodal. No obstante, incluso en ese caso, el resultado debe describirse como funcionamiento parcial en condiciones controladas. No sería correcto afirmar que el sistema es confiable para un uso real sin más evidencia.