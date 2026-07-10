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