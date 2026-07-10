# MCC225 Textiles Generados Evaluacion Multimodal

Este repositorio contiene el desarrollo de una actividad complementaria del curso MCC225. El trabajo se centra en la evaluacion responsable de un prototipo multimodal aplicado a imagenes generadas de inspiracion textil andina.

El objetivo principal es analizar si un modelo multimodal puede relacionar imagenes y textos descriptivos en un escenario controlado. Para ello, se evaluara una tarea de recuperacion imagen texto, donde el sistema debe asociar cada imagen con la descripcion mas adecuada dentro de un conjunto de textos candidatos.

El proyecto no busca demostrar que el sistema funciona de manera perfecta. Se busca observar su comportamiento, identificar aciertos, revisar errores, analizar casos ambiguos y discutir limites de uso. La evaluacion se plantea como un ejercicio tecnico y academico, con evidencia trazable y resultados reproducibles.

## Tarea multimodal

La tarea principal es recuperación imagen texto. A partir de una imagen generada, el sistema debe recuperar la descripcion textual que mejor corresponde a sus atributos visuales.

De forma complementaria, tambien se podra revisar la recuperación texto imagen, donde una descripcion se utiliza para buscar la imagen mas cercana dentro del conjunto evaluado.

## Modalidades utilizadas

Las modalidades consideradas son:

* Imagen.
* Texto.

## Evidencia esperada

El repositorio organizará los siguientes elementos:

* Notebook adaptado al proyecto.
* Reporte breve de evaluación responsable.
* Tabla de métricas.
* Tabla de cinco casos analizados.
* Pruebas de confiabilidad.
* Tabla de explicabilidad.
* Ficha de uso responsable.
* Figura o collage con ejemplos evaluados.

## Estructura inicial del repositorio

```text
mcc225-textiles-generados-evaluacion-multimodal/
├── README.md
├── notebooks/
│   └── Cuaderno14_MCC225_resuelto.ipynb
├── data/
│   ├── images/
│   └── manifest_textiles_generados.csv
├── results/
│   ├── metricas.csv
│   ├── casos_analizados.csv
│   ├── pruebas_confiabilidad.csv
│   ├── explicabilidad.csv
│   └── ficha_uso_responsable.csv
├── figures/
│   └── ejemplos_evaluados.png
├── reporte_evaluacion_responsable.md
└── requirements.txt
```

## Alcance

El sistema se evaluará en condiciones controladas usando imágenes generadas y descripciones textuales construidas para el experimento. Los resultados deben interpretarse como evidencia preliminar sobre alineamiento multimodal, no como reconocimiento confiable ni clasificacion real de objetos culturales.

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

## Construcción del dataset

El dataset de trabajo se construye de forma reproducible mediante el script `scripts/generar_dataset_sintetico.py`.

Este script genera 40 imágenes sintéticas inspiradas en patrones textiles andinos. Las imágenes se almacenan en `data/images/` y el manifiesto se guarda en `data/manifest_textiles_generados.csv`.

Cada registro del manifiesto contiene:

- identificador de imagen;
- ruta local de la imagen;
- cinco descripciones textuales;
- paleta cromática;
- composición;
- tipo de motivo;
- simetría;
- nivel de ambigüedad;
- observación de uso.

El objetivo de este dataset no es representar piezas reales ni clasificar objetos culturales. Su finalidad es crear un escenario controlado para evaluar alineamiento imagen texto, errores, confiabilidad y límites de uso del sistema multimodal.

Para generar el dataset se ejecuta:

```bash
python scripts/generar_dataset_sintetico.py

## Evaluación experimental

El cuaderno `notebooks/Cuaderno14_MCC225_resuelto.ipynb` ejecuta la evaluación principal del prototipo multimodal.

El procedimiento general es el siguiente:

1. Cargar el manifiesto de imágenes y captions.
2. Validar la existencia de las imágenes.
3. Construir la tabla de textos candidatos.
4. Calcular embeddings de imágenes y textos con OpenCLIP.
5. Calcular una matriz de similitud imagen texto.
6. Evaluar recuperación imagen texto mediante Recall@1, Recall@5, Recall@10 y MRR.
7. Comparar los resultados contra un baseline aleatorio.
8. Exportar rankings, métricas y figuras.

Los resultados principales se almacenan en:

- `results/metricas.csv`
- `results/rankings_imagen_texto_top10.csv`
- `results/primer_acierto_por_imagen.csv`
- `results/resumen_evaluacion.json`
- `figures/matriz_similitud_agrupada.png`

## Resultados cuantitativos

La evaluación experimental genera una tabla de métricas en `results/metricas.csv`. También se guarda una comparación entre el modelo y el baseline en `results/comparacion_modelo_baseline.csv`.

Las métricas principales son:

- Recall@1.
- Recall@5.
- Recall@10.
- MRR.

Estas métricas permiten revisar si el modelo recupera captions correctos para cada imagen y si supera a un baseline aleatorio. La interpretación de los resultados se desarrolla en `reporte_evaluacion_responsable.md`.