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

Las métricas principales fueron Recall@1, Recall@5, Recall@10 y MRR. Recall@1 permitirá observar si el texto correcto aparece como primera opción. Recall@5 permitirá evaluar si el texto correcto aparece dentro de los primeros resultados. La métrica complementaria ayudará a revisar la posición relativa o la similitud promedio entre imagen y texto.

También se incluirá un baseline simple. Este baseline podrá construirse mediante captions desplazados o mediante un ranking aleatorio. La comparación contra este baseline es necesaria para saber si el modelo muestra una señal real de alineamiento o si los resultados podrían explicarse por azar o por coincidencias débiles del conjunto evaluado.

La adaptación del cuaderno no se limitará a ejecutar el modelo. También deberá producir archivos de salida en el repositorio, incluyendo una tabla de métricas, una tabla de casos analizados, resultados de pruebas de confiabilidad, una tabla de explicabilidad y una figura con ejemplos evaluados. Esto permitirá que la evaluación sea revisable y reproducible.

## Ficha de adaptación del Cuaderno 14

| Elemento | Respuesta |
|---|---|
| Modelo usado | CLIP u OpenCLIP |
| Dataset o subconjunto | Imágenes generadas de inspiración textil andina con descripciones textuales controladas |
| Número de ejemplos | 40 imágenes generadas, cada una con cinco descripciones textuales |
| Métricas usadas | Recall@1, Recall@5 y MRR |
| Baseline usado | Ranking aleatorio |
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
| E1 | OpenCLIP ViT-B-32 | 40 imágenes y 200 textos | Recall@1 | 0.325 | El modelo ubica un caption correcto en la primera posición en el 32.5% de los casos |
| E1 | OpenCLIP ViT-B-32 | 40 imágenes y 200 textos | Recall@5 | 0.600 | El modelo recupera al menos un caption correcto entre los cinco primeros resultados en el 60.0% de los casos |
| E1 | OpenCLIP ViT-B-32 | 40 imágenes y 200 textos | MRR | 0.481256 | El primer caption correcto aparece relativamente alto en el ranking |
| E2 | Baseline aleatorio | 40 imágenes y 200 textos | Recall@1 | 0.026175 | El desempeño esperado por azar es bajo en la primera posición |
| E2 | Baseline aleatorio | 40 imágenes y 200 textos | Recall@5 | 0.119300 | El baseline recupera captions correctos en el top 5 con mucha menor frecuencia que el modelo |
| E2 | Baseline aleatorio | 40 imágenes y 200 textos | MRR | 0.097035 | El primer acierto aparece mucho más abajo cuando el ordenamiento es aleatorio |

### Interpretación general

Los resultados muestran que OpenCLIP supera claramente al baseline aleatorio en las métricas principales. En Recall@1, el modelo alcanza 0.325 frente a 0.026175 del baseline. Esto indica que el sistema logra ubicar un caption correcto en la primera posición con una frecuencia mucho mayor que la esperada por azar.

En Recall@5, el modelo obtiene 0.600 frente a 0.119300 del baseline. Esto muestra que, aunque el caption correcto no siempre aparece en primer lugar, en una proporción importante de los casos sí se encuentra dentro de los primeros cinco resultados. El valor de MRR también confirma esta tendencia, ya que OpenCLIP alcanza 0.481256, mientras que el baseline obtiene 0.097035.

Estos resultados permiten afirmar que existe una señal de alineamiento imagen texto en el conjunto evaluado. El modelo no está ordenando los captions de manera equivalente al azar, sino que aprovecha información visual y textual para acercar algunas imágenes a sus descripciones correspondientes.

Sin embargo, el resultado debe interpretarse con cautela. Un Recall@1 de 0.325 significa que el sistema solo ubica un caption correcto en la primera posición en aproximadamente un tercio de los casos. Por tanto, aunque el modelo supera al baseline, todavía presenta fallos importantes. El Recall@5 de 0.600 sugiere que el sistema puede recuperar información relacionada, pero no siempre con la precisión suficiente para seleccionar una única descripción correcta.

En consecuencia, el sistema muestra funcionamiento parcial en condiciones controladas. No sería adecuado afirmar que el modelo comprende los patrones visuales ni que puede realizar identificación confiable de objetos. La evidencia obtenida solo permite sostener que OpenCLIP logra cierto alineamiento entre imágenes generadas y captions visuales controlados.

### Pregunta 1

¿Qué métrica representa mejor el objetivo de la tarea?

La métrica que representa mejor el objetivo de la tarea es Recall@1, porque la evaluación busca saber si el sistema puede recuperar la descripción textual más adecuada para una imagen. En este experimento, OpenCLIP obtuvo un Recall@1 de 0.325, lo que indica que en el 32.5% de los casos ubicó un caption correcto en la primera posición.

### Pregunta 2

¿Qué métrica puede ser insuficiente si se interpreta sola?

Recall@5 puede ser insuficiente si se interpreta sola. En este experimento, el modelo obtuvo un Recall@5 de 0.600, lo cual indica que en el 60.0% de los casos algún caption correcto apareció entre los cinco primeros resultados. Sin embargo, esta métrica no indica si el sistema seleccionó correctamente la mejor respuesta en la primera posición. Por ello, debe analizarse junto con Recall@1 y MRR.

### Pregunta 3

¿El sistema supera claramente al baseline o solo muestra funcionamiento parcial?

El sistema supera claramente al baseline aleatorio en Recall@1, Recall@5 y MRR. Esto permite afirmar que existe una señal de alineamiento multimodal. Sin embargo, el desempeño sigue siendo parcial. El modelo mejora frente al azar, pero todavía falla en una proporción considerable de imágenes. Por ello, el resultado debe describirse como funcionamiento parcial en un entorno controlado, no como confiabilidad para uso real.

## 6. Análisis de cinco casos

Además de las métricas cuantitativas, se realizó una revisión cualitativa de cinco casos reales obtenidos del experimento. Esta revisión permite observar en qué situaciones el modelo acierta, dónde falla y qué casos requieren una interpretación más cuidadosa.

Los casos fueron seleccionados a partir de los rankings generados por OpenCLIP. No se construyeron manualmente después de observar el resultado. Se seleccionaron dos aciertos claros, dos errores claros y un caso ambiguo.

La tabla siguiente resume los casos analizados.

| caso_id | image_id | tipo_caso | salida_modelo | resultado_esperado | rank_correcto | tipo_error | explicacion_breve |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C01 | T037 | acierto | Patrón textil generado con composicion horizontal, motivos de bandas geométricas y paleta azul, crema, marrón. | Patrón textil generado con composicion horizontal, motivos de bandas geométricas y paleta azul, crema, marrón. | 1 | no_aplica | El modelo ubicó un caption correcto en la primera posición. La salida coincide con atributos visibles como composición horizontal, motivo bandas geométricas y paleta azul, crema marrón. |
| C02 | T032 | acierto | Patrón textil generado con composicion vertical, motivos de bandas geométricas y paleta azul, crema, marrón. | Patrón textil generado con composicion vertical, motivos de bandas geométricas y paleta azul, crema, marrón. | 1 | no_aplica | El modelo ubicó un caption correcto en la primera posición. La salida coincide con atributos visibles como composición vertical, motivo bandas geométricas y paleta azul, crema marrón. |
| C03 | T010 | error | Patrón textil generado con composicion horizontal, motivos de bandas geométricas y paleta multicolor, tierra. | Diseño geometrico generado con paleta multicolor, tierra, estructura modular y repetición visual. | 26 | error de alineamiento imagen-texto | El modelo ubicó como primera opción un caption asociado a otra imagen. El primer caption correcto apareció en la posición 26. Esto sugiere una confusión entre patrones visualmente parecidos o una asociación insuficiente entre los atributos de la imagen y el texto correspondiente. |
| C04 | T040 | error | Patrón textil generado con composicion horizontal, motivos de bandas geométricas y paleta multicolor, tierra. | Diseño geometrico generado con paleta multicolor, tierra, estructura modular y repetición visual. | 25 | error de alineamiento imagen-texto | El modelo ubicó como primera opción un caption asociado a otra imagen. El primer caption correcto apareció en la posición 25. Esto sugiere una confusión entre patrones visualmente parecidos o una asociación insuficiente entre los atributos de la imagen y el texto correspondiente. |
| C05 | T024 | ambiguo | Patrón textil generado con composicion horizontal, motivos de bandas geométricas y paleta blanco, negro, gris. | Patrón textil generado con composicion mixta, motivos geométricos combinados y paleta blanco, negro, gris. | 4 | error_por_ambiguedad_visual | El caso se considera ambiguo porque la imagen presenta una lectura visual menos directa. El nivel de ambigüedad registrado es alto. La salida del modelo puede estar parcialmente relacionada con la imagen, pero no permite una decisión estricta sin revisión cualitativa. |

### Interpretación de los aciertos

Los aciertos claros corresponden a casos en los que el modelo ubicó un caption correcto en la primera posición. En estos casos, la coincidencia se explica principalmente por atributos visuales observables, como composición, paleta cromática, presencia de bandas, motivos geométricos o repetición modular.

Estos aciertos muestran que el modelo puede capturar señales generales de alineamiento imagen texto. Sin embargo, no deben interpretarse como comprensión completa del patrón ni como capacidad de clasificación real.

### Interpretación de los errores

Los errores claros corresponden a imágenes donde el primer resultado del modelo pertenece a otra imagen o donde el primer caption correcto aparece en una posición baja del ranking. Estos errores indican que el modelo puede confundir patrones que comparten rasgos visuales generales, como geometría repetitiva, contraste cromático o composición modular.

Este tipo de error sería problemático en un uso real porque podría generar una asociación aparentemente plausible, pero incorrecta. Por ello, la salida del modelo requiere revisión humana y no debería utilizarse como decisión final.

### Caso ambiguo

El caso ambiguo muestra una situación en la que la imagen no tiene una lectura visual única o comparte rasgos con varios grupos de captions. En este caso, la salida del modelo puede ser parcialmente razonable, aunque no necesariamente sea la mejor respuesta.

Este caso es importante porque muestra que no todos los desaciertos tienen la misma gravedad. Algunos resultados pueden ser incorrectos de forma clara, mientras que otros reflejan ambigüedad del dato o limitaciones en la forma en que se construyeron los captions.

### Pregunta breve

¿Cuál fue el error más importante y por qué sería problemático en un uso real?

El error más importante es el error de alineamiento imagen texto, especialmente cuando el modelo entrega como primera opción un caption asociado a otra imagen. Este error sería problemático en un uso real porque puede presentar una asociación visual como si fuera correcta, aunque no corresponda al caso evaluado. En contextos donde se requiere interpretación cuidadosa, este tipo de salida puede inducir conclusiones equivocadas si no existe revisión humana.

## 7. Confiabilidad

La confiabilidad del sistema se evaluó mediante dos pruebas breves. La primera prueba revisó la sensibilidad del modelo ante cambios en la redacción del texto. La segunda prueba revisó el comportamiento del modelo ante imágenes degradadas visualmente.

Estas pruebas no buscan demostrar robustez completa. Su objetivo es observar si el sistema mantiene un comportamiento razonable cuando se modifican condiciones pequeñas pero relevantes del experimento.

### Prueba 1: sensibilidad al texto

En esta prueba se tomó un conjunto de captions originales y se construyeron variantes con una redacción distinta, pero con significado visual similar. Luego se evaluó si la imagen esperada seguía apareciendo en una posición cercana dentro del ranking.

Esta prueba permite revisar si el modelo depende excesivamente de una formulación textual específica. Si una reformulación simple produce cambios grandes en el ranking, el sistema sería sensible al prompt y requeriría mayor control en la construcción de consultas.

### Prueba 2: degradación visual

En esta prueba se generaron variantes degradadas de algunas imágenes. La degradación incluyó reducción de resolución, conversión a escala de grises y desenfoque ligero. Luego se comparó el ranking del primer caption correcto antes y después de la degradación.

Esta prueba permite observar si el sistema mantiene la recuperación cuando la imagen pierde información visual. Si el cambio de ranking es alto, el modelo puede ser poco estable ante imágenes de menor calidad.

### Resultados de las pruebas

| prueba | image_id | entrada_original | variante | cambio_observado | interpretacion |
| --- | --- | --- | --- | --- | --- |
| sensibilidad_al_texto | T001 | Patrón textil generado con composicion horizontal, motivos de bandas geométricas y paleta rojo, negro, crema. | Imagen generada con motivo bandas geométricas, composición horizontal y paleta rojo_negro_crema. | bajo | Evalúa si una reformulación textual con significado similar cambia de forma importante la posición de la imagen esperada. |
| sensibilidad_al_texto | T002 | Patrón textil generado con composicion vertical, motivos de bandas geométricas y paleta azul, crema, marrón. | Imagen generada con motivo bandas geométricas, composición vertical y paleta azul, crema marrón. | medio | Evalúa si una reformulación textual con significado similar cambia de forma importante la posición de la imagen esperada. |
| sensibilidad_al_texto | T003 | Patrón textil generado con composicion central, motivos de rombos y paleta ocre, marrón, negro. | Imagen generada con motivo rombos, composición central y paleta ocre_marron_negro. | alto | Evalúa si una reformulación textual con significado similar cambia de forma importante la posición de la imagen esperada. |
| sensibilidad_al_texto | T004 | Patrón textil generado con composicion modular, motivos de grecas escalonadas y paleta blanco, negro, gris. | Imagen generada con motivo grecas escalonadas, composición modular y paleta blanco_negro_gris. | medio | Evalúa si una reformulación textual con significado similar cambia de forma importante la posición de la imagen esperada. |
| sensibilidad_al_texto | T005 | Patrón textil generado con composicion reticular, motivos de cuadrícula geometrica y paleta multicolor, tierra. | Imagen generada con motivo cuadrícula geometrica, composición reticular y paleta multicolor tierra. | medio | Evalúa si una reformulación textual con significado similar cambia de forma importante la posición de la imagen esperada. |
| degradacion_visual | T010 | data/images/T010.png | imagen en baja resolución, escala de grises y desenfoque ligero | medio | Evalúa si la recuperación imagen texto se mantiene cuando la imagen pierde calidad visual. |
| degradacion_visual | T024 | data/images/T024.png | imagen en baja resolución, escala de grises y desenfoque ligero | medio | Evalúa si la recuperación imagen texto se mantiene cuando la imagen pierde calidad visual. |
| degradacion_visual | T032 | data/images/T032.png | imagen en baja resolución, escala de grises y desenfoque ligero | alto | Evalúa si la recuperación imagen texto se mantiene cuando la imagen pierde calidad visual. |
| degradacion_visual | T037 | data/images/T037.png | imagen en baja resolución, escala de grises y desenfoque ligero | alto | Evalúa si la recuperación imagen texto se mantiene cuando la imagen pierde calidad visual. |
| degradacion_visual | T040 | data/images/T040.png | imagen en baja resolución, escala de grises y desenfoque ligero | medio | Evalúa si la recuperación imagen texto se mantiene cuando la imagen pierde calidad visual. |

### Clasificación final de confiabilidad

La clasificación final adoptada es: **confiable solo en condiciones controladas**.

Esta clasificación se justifica porque el sistema supera claramente al baseline aleatorio en las métricas cuantitativas, pero todavía presenta limitaciones importantes. El Recall@1 de 0.325 indica que el caption correcto aparece en la primera posición solo en una parte de los casos. Además, las pruebas de confiabilidad permiten observar que cambios en el texto o en la calidad visual pueden modificar el ranking.

Por ello, el sistema puede utilizarse como herramienta exploratoria dentro de un experimento académico controlado, pero no debería usarse como sistema autónomo ni como mecanismo confiable para tomar decisiones sin revisión humana.

## 8. Explicabilidad

La explicabilidad se trabajó a partir de dos casos reales del experimento. Se seleccionó un caso de acierto y un caso de error o ambigüedad. El objetivo fue revisar si la salida del modelo podía relacionarse con evidencia observable en la imagen y en los textos candidatos.

La explicación propuesta no debe entenderse como una explicación interna completa del modelo. OpenCLIP no entrega directamente una justificación semántica ni una región exacta de atención para cada decisión. Por ello, la explicación se construye como una revisión cualitativa basada en atributos visibles, captions recuperados, resultado esperado y posición del ranking.

La tabla siguiente resume los dos casos explicados.

| caso_id | image_id | tipo_caso | evidencia_visual | explicacion_propuesta | limite_explicacion |
| --- | --- | --- | --- | --- | --- |
| C01 | T037 | acierto | Imagen con composición horizontal, motivo bandas geométricas, paleta azul, crema marrón y simetría repetitiva. | El caso puede explicarse porque la salida del modelo coincide con atributos visuales registrados en el manifiesto, como la composición, el motivo y la paleta. La coincidencia sugiere alineamiento entre la imagen y el texto recuperado. | La explicación se basa en atributos observables y en los resultados del ranking. No permite afirmar qué regiones exactas de la imagen usó el modelo ni demuestra comprensión del significado del patrón. |
| C03 | T010 | error | Imagen con composición modular, motivo grecas escalonadas, paleta multicolor tierra y simetría repetitiva. | El error puede explicarse porque el modelo recuperó primero un caption asociado a otra imagen. Esto indica que algunos rasgos generales, como geometría, color o repetición, pueden haber sido más influyentes que la correspondencia específica con la imagen evaluada. | La explicación se basa en atributos observables y en los resultados del ranking. No permite afirmar qué regiones exactas de la imagen usó el modelo ni demuestra comprensión del significado del patrón. |

### Pregunta 1

¿La explicación se basa en evidencia observable o solo racionaliza la salida después de verla?

La explicación se basa en evidencia observable, porque utiliza atributos registrados en el manifiesto, la imagen evaluada, la salida del modelo y el resultado esperado. No obstante, debe interpretarse con cautela, porque no muestra directamente qué regiones de la imagen fueron usadas por el modelo ni permite reconstruir su proceso interno completo.

### Pregunta 2

¿La explicación ayuda a detectar un error del sistema?

Sí. La explicación ayuda a detectar errores cuando permite observar que el modelo recuperó un caption asociado a otra imagen o confundió atributos visuales generales con una correspondencia específica. En particular, permite distinguir entre un acierto claro, un error de alineamiento imagen texto y un caso ambiguo que requiere revisión cualitativa.

### Interpretación

Los casos explicados muestran que la evidencia observable puede ayudar a interpretar los resultados, pero no elimina la necesidad de supervisión humana. En los aciertos, la explicación permite identificar coincidencias entre composición, color, motivo y texto recuperado. En los errores o casos ambiguos, permite reconocer que el modelo puede responder a similitudes visuales generales sin garantizar una correspondencia correcta.

Por tanto, la explicabilidad en este proyecto debe entenderse como una herramienta de revisión cualitativa. No demuestra comprensión del modelo, pero sí ayuda a documentar por qué una salida puede considerarse aceptable, incorrecta o ambigua.

## 9. Sesgo y uso responsable

La evaluación responsable del sistema requiere reconocer que el experimento se realiza sobre imágenes generadas y descripciones textuales controladas. Por ello, los resultados no deben interpretarse como evidencia de reconocimiento real, clasificación cultural ni validación patrimonial.

El conjunto de evaluación permite revisar alineamiento imagen texto en un escenario controlado, pero no permite demostrar que el sistema comprenda el significado de los patrones visuales. Tampoco permite afirmar que el modelo sea robusto ante datos reales, imágenes de baja calidad, descripciones incompletas o categorías culturales complejas.

La tabla siguiente resume los principales riesgos y condiciones de uso identificadas.

| aspecto | respuesta |
| --- | --- |
| posible_sesgo_visual | Las imágenes generadas pueden reproducir una estética simplificada de patrones textiles andinos. Esto puede hacer que el modelo aprenda o refuerce asociaciones visuales generales, sin distinguir variaciones reales, técnicas específicas ni contexto cultural. |
| posible_sesgo_linguistico | Los captions fueron construidos manualmente y pueden favorecer ciertos términos descriptivos, como bandas, rombos, grecas, simetría o composición. El modelo puede responder mejor a esas palabras que a descripciones alternativas o menos estructuradas. |
| posible_sesgo_cultural_o_dominio | El sistema trabaja con imágenes generadas y no con piezas documentadas. Por ello, puede confundir rasgos visuales inspirados en lo andino con una identificación cultural real. Esta es una limitación importante del experimento. |
| riesgo_principal_si_se_usa_mal | El riesgo principal es interpretar una similitud visual como si fuera una clasificación confiable. Esto podría llevar a conclusiones incorrectas sobre una imagen si se usa el modelo sin revisión humana. |
| supervision_humana_necesaria | Sí. La salida del modelo debe ser revisada por una persona, especialmente cuando se analizan errores, casos ambiguos o posibles interpretaciones del patrón visual. |
| uso_recomendado | Limitado. |
| justificacion_del_uso_recomendado | El sistema puede usarse como herramienta exploratoria en un entorno académico controlado. No debe usarse como mecanismo autónomo de identificación, clasificación cultural o validación patrimonial. |
| afirmacion_irresponsable | Sería irresponsable afirmar que el sistema reconoce correctamente patrones textiles andinos reales o que puede clasificar objetos culturales de manera confiable. |
| afirmacion_prudente | Una afirmación prudente sería indicar que el sistema muestra alineamiento parcial entre imágenes generadas y descripciones visuales controladas en un entorno experimental. |

### Discusión de sesgos posibles

El primer riesgo es visual. Las imágenes generadas pueden simplificar ciertos rasgos de inspiración textil andina y repetir una estética relativamente homogénea. Esto puede favorecer que el modelo aprenda asociaciones generales, como bandas, rombos, grecas o composiciones repetitivas, sin distinguir matices reales.

El segundo riesgo es lingüístico. Los captions fueron construidos manualmente y utilizan un vocabulario controlado. Esto puede favorecer el desempeño del modelo cuando las consultas tienen una forma parecida a las descripciones del manifiesto, pero no garantiza estabilidad ante descripciones más libres o ambiguas.

El tercer riesgo es cultural o de dominio. El sistema no trabaja con piezas reales documentadas, sino con imágenes generadas. Por ello, cualquier interpretación cultural debe evitarse. El modelo solo evalúa similitud entre imágenes y textos dentro del experimento.

### Uso recomendado

El uso recomendado del sistema es limitado. Puede utilizarse como herramienta académica para explorar recuperación imagen texto y analizar errores de modelos multimodales en un conjunto controlado.

No se recomienda usar el sistema para identificación cultural, clasificación real de objetos, validación patrimonial ni toma de decisiones sin supervisión humana.

### Pregunta breve

¿Qué afirmación sería irresponsable hacer sobre este sistema?

Sería irresponsable afirmar que el sistema reconoce correctamente patrones textiles andinos reales o que puede clasificar objetos culturales de manera confiable. Los resultados obtenidos solo permiten afirmar que existe alineamiento parcial entre imágenes generadas y descripciones visuales controladas en un entorno experimental.

Una afirmación más prudente sería la siguiente: el sistema muestra una señal de alineamiento imagen texto superior al baseline aleatorio, pero su desempeño sigue siendo parcial y requiere revisión humana.

## 10. Conclusión técnica

La evaluación realizada permitió analizar de manera responsable un prototipo multimodal aplicado a imágenes generadas de inspiración textil andina. El sistema evaluado no fue planteado como una herramienta final de reconocimiento visual, sino como un ejercicio académico orientado a observar el comportamiento de un modelo imagen texto en condiciones controladas. La tarea principal fue recuperación imagen texto, donde el modelo debía asociar cada imagen generada con una descripción textual adecuada entre un conjunto de captions candidatos.

El experimento se desarrolló con 40 imágenes generadas y 200 descripciones textuales. Cada imagen contó con cinco captions válidos, construidos a partir de atributos visuales observables como paleta cromática, composición, motivo geométrico, simetría y nivel de ambigüedad. Para la evaluación se utilizó OpenCLIP ViT-B-32, comparando sus resultados con un baseline aleatorio. Las métricas principales fueron Recall@1, Recall@5 y MRR.

Los resultados cuantitativos muestran que OpenCLIP supera claramente al baseline aleatorio. El modelo obtuvo un Recall@1 de 0.325 frente a 0.026175 del baseline. También obtuvo un Recall@5 de 0.600 frente a 0.119300 del baseline. En MRR, el modelo alcanzó 0.481256, mientras que el baseline obtuvo 0.097035. Estos valores indican que el modelo sí captura una señal de alineamiento entre imágenes y textos dentro del conjunto evaluado. Sin embargo, el resultado no debe interpretarse como confiabilidad completa. Un Recall@1 de 0.325 significa que el sistema ubica un caption correcto en la primera posición solo en aproximadamente un tercio de los casos.

El análisis cualitativo permitió revisar aciertos, errores y casos ambiguos. Los aciertos se explican principalmente por coincidencias en atributos visuales generales, como bandas, composición, paleta o repetición geométrica. El error más importante identificado es el error de alineamiento imagen texto, especialmente cuando el modelo asigna como primera opción un caption asociado a otra imagen. Este tipo de error es problemático porque puede presentar una salida aparentemente razonable, pero incorrecta, si no existe una revisión posterior.

Las pruebas de confiabilidad mostraron que el sistema debe considerarse confiable solo en condiciones controladas. El modelo puede mejorar frente al azar, pero todavía es sensible a cambios en la formulación textual y a la degradación visual de las imágenes. Por ello, su uso debe limitarse a escenarios exploratorios y académicos.

La limitación más importante que debe comunicarse es que el experimento trabaja con imágenes generadas, no con piezas reales. Por tanto, los resultados no permiten afirmar identificación cultural, procedencia histórica, autenticidad ni clasificación patrimonial. El sistema solo evalúa alineamiento multimodal dentro de un conjunto sintético y controlado.

En conclusión, el prototipo muestra funcionamiento parcial y evidencia inicial de alineamiento imagen texto. Puede ser útil para explorar métricas, errores y límites de modelos multimodales, pero no debe utilizarse como herramienta autónoma de interpretación, clasificación o validación. Cualquier uso fuera del entorno académico controlado requeriría nuevos datos, mayor evaluación, revisión humana especializada y una discusión más amplia sobre sesgos y responsabilidad.