# MCC225 Textiles Generados Evaluación Multimodal

Este repositorio contiene el desarrollo de una actividad complementaria del curso MCC225. El trabajo se centra en la evaluación responsable de un prototipo multimodal aplicado a imágenes generadas de inspiración textil andina.

El objetivo es analizar si un modelo multimodal puede relacionar imágenes y textos descriptivos en un escenario controlado. Para ello, se evalúa una tarea de recuperación imagen texto, donde el sistema debe asociar cada imagen con la descripción más adecuada dentro de un conjunto de textos candidatos.

El proyecto no busca demostrar que el sistema funciona de manera perfecta. Se busca observar su comportamiento, identificar aciertos, revisar errores, analizar casos ambiguos y discutir límites de uso. La evaluación se plantea como un ejercicio técnico y académico, con evidencia trazable y resultados reproducibles.

## 1. Tarea multimodal

La tarea principal es recuperación imagen texto. A partir de una imagen generada, el sistema debe recuperar la descripción textual que mejor corresponde a sus atributos visuales.

De forma complementaria, también se puede revisar la recuperación texto imagen, donde una descripción se utiliza para buscar la imagen más cercana dentro del conjunto evaluado.

## 2. Modalidades utilizadas

Las modalidades consideradas son:

- Imagen.
- Texto.

## 3. Dataset

El dataset de trabajo se construye de forma reproducible mediante el script `scripts/generar_dataset_sintetico.py`.

El conjunto contiene 40 imágenes generadas y 200 descripciones textuales candidatas. Cada imagen tiene cinco captions asociados. Las imágenes se almacenan en `data/images/` y el manifiesto se registra en `data/manifest_textiles_generados.csv`.

El dataset no representa piezas reales ni permite identificación cultural. Su finalidad es crear un escenario controlado para evaluar alineamiento imagen texto, errores, confiabilidad y límites de uso.

## 4. Modelo utilizado

El modelo utilizado para la evaluación es OpenCLIP ViT-B-32 con pesos preentrenados `laion2b_s34b_b79k`.

OpenCLIP permite representar imágenes y textos en un mismo espacio vectorial. A partir de estas representaciones se calcula una matriz de similitud entre cada imagen y cada descripción textual.

## 5. Evaluación experimental

El cuaderno `notebooks/Cuaderno14_MCC225_resuelto.ipynb` ejecuta la evaluación principal del prototipo multimodal.

El procedimiento general es el siguiente:

1. Cargar el manifiesto de imágenes y captions.
2. Validar la existencia de las imágenes.
3. Construir la tabla de textos candidatos.
4. Calcular embeddings de imágenes y textos con OpenCLIP.
5. Calcular la matriz de similitud imagen texto.
6. Evaluar recuperación imagen texto mediante Recall@1, Recall@5, Recall@10 y MRR.
7. Comparar los resultados contra un baseline aleatorio.
8. Exportar métricas, rankings y figuras.

## 6. Métricas utilizadas

Las métricas principales son:

- Recall@1.
- Recall@5.
- Recall@10.
- MRR.

Recall@1 mide si un caption correcto aparece en la primera posición del ranking.

Recall@5 y Recall@10 permiten revisar si un caption correcto aparece dentro de los primeros resultados.

MRR permite evaluar qué tan arriba aparece el primer caption correcto en el ranking.

## 7. Resultados cuantitativos preliminares

La evaluación se realizó sobre 40 imágenes generadas y 200 textos candidatos.

| Modelo | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| OpenCLIP ViT-B-32 | 0.325 | 0.600 | 0.481256 |
| Baseline aleatorio | 0.026175 | 0.119300 | 0.097035 |

Los resultados muestran que OpenCLIP supera al baseline aleatorio en las métricas principales. Sin embargo, el desempeño debe interpretarse como funcionamiento parcial en condiciones controladas, no como confiabilidad para uso real.

## 8. Interpretación breve de resultados

El modelo obtiene un Recall@1 de 0.325. Esto significa que logra ubicar un caption correcto en la primera posición en el 32.5 por ciento de los casos.

El Recall@5 es 0.600. Esto indica que en el 60.0 por ciento de los casos el modelo recupera al menos un caption correcto entre los cinco primeros resultados.

El MRR obtenido por el modelo es 0.481256, valor superior al baseline aleatorio. Esto sugiere que el primer caption correcto suele aparecer bastante más arriba en el ranking que cuando los textos se ordenan al azar.

A pesar de esta mejora frente al baseline, el resultado no debe interpretarse como un sistema confiable para uso real. El modelo muestra una señal de alineamiento imagen texto, pero todavía falla en una proporción importante de imágenes.

## 9. Estructura del repositorio

La estructura principal del repositorio es la siguiente:

README.md

reporte_evaluacion_responsable.md

requirements.txt

notebooks/
- Cuaderno14_MCC225_resuelto.ipynb

data/
- manifest_textiles_generados.csv

data/images/
- T001.png
- T002.png
- ...
- T040.png

results/
- metricas.csv
- comparacion_modelo_baseline.csv
- casos_analizados.csv
- pruebas_confiabilidad.csv
- explicabilidad.csv
- ficha_uso_responsable.csv

figures/
- comparacion_metricas.png

scripts/
- generar_dataset_sintetico.py

## 10. Archivos principales

`reporte_evaluacion_responsable.md`

Contiene el reporte técnico de la actividad, incluyendo descripción del proyecto, adaptación del cuaderno, construcción del dataset, resultados cuantitativos, análisis de casos, confiabilidad, explicabilidad, sesgos y conclusiones.

`notebooks/Cuaderno14_MCC225_resuelto.ipynb`

Contiene el cuaderno experimental adaptado para evaluar recuperación imagen texto con OpenCLIP.

`scripts/generar_dataset_sintetico.py`

Genera de forma reproducible las imágenes sintéticas y el manifiesto del dataset.

`data/manifest_textiles_generados.csv`

Contiene la relación entre imágenes, captions y atributos visuales controlados.

`results/metricas.csv`

Contiene las métricas principales del modelo y del baseline.

`results/comparacion_modelo_baseline.csv`

Contiene la comparación entre OpenCLIP y el baseline aleatorio.

`figures/comparacion_metricas.png`

Contiene una figura comparativa de las métricas principales.

## 11. Reproducción del experimento

Primero se recomienda crear y activar un entorno virtual.

En Windows, por ejemplo:

python -m venv D:\venvs\mcc225_eval

D:\venvs\mcc225_eval\Scripts\activate

Luego, instalar las dependencias:

pip install -r requirements.txt

Generar el dataset:

python scripts\generar_dataset_sintetico.py

Después, ejecutar el cuaderno:

notebooks/Cuaderno14_MCC225_resuelto.ipynb

## 12. Dependencias principales

Las dependencias principales del proyecto son:

- pandas.
- numpy.
- matplotlib.
- Pillow.
- torch.
- torchvision.
- open_clip_torch.
- scikit-learn.
- tqdm.
- ipykernel.

## 13. Alcance

El sistema se evalúa en condiciones controladas usando imágenes generadas y descripciones textuales construidas para el experimento. Los resultados deben interpretarse como evidencia preliminar sobre alineamiento multimodal.

El experimento no permite afirmar que el sistema reconozca objetos culturales reales. Tampoco permite afirmar autenticidad, procedencia histórica ni clasificación patrimonial.

## 14. Limitaciones

Las principales limitaciones son:

- Las imágenes son generadas, no corresponden a piezas reales.
- Los captions fueron construidos manualmente para el experimento.
- El conjunto de evaluación es pequeño.
- El modelo puede capturar similitud visual sin comprender el significado del patrón.
- El buen desempeño frente al baseline no implica confiabilidad para uso real.
- La evaluación requiere revisión cualitativa de aciertos, errores y casos ambiguos.

## 15. Uso recomendado

El uso recomendado del sistema es limitado. Puede utilizarse como ejercicio académico para explorar alineamiento imagen texto en un entorno controlado.

No se recomienda usarlo para identificación cultural, clasificación real de objetos, validación patrimonial ni toma de decisiones sin supervisión humana.

## 16. Estado del proyecto

El repositorio contiene el dataset generado, el cuaderno experimental, los resultados cuantitativos, el análisis cualitativo de cinco casos, las pruebas de confiabilidad, la tabla de explicabilidad, la ficha de uso responsable y la conclusión técnica.

El trabajo se encuentra en estado de cierre para revisión académica.

## Análisis cualitativo

Además de las métricas, el proyecto incluye una revisión de cinco casos reales del experimento:

- dos aciertos claros;
- dos errores claros;
- un caso ambiguo.

La tabla se almacena en `results/casos_analizados.csv` y la figura resumen en `figures/ejemplos_evaluados.png`.

Este análisis permite revisar errores de alineamiento imagen texto, casos de ambigüedad y límites de interpretación del modelo.

## Pruebas de confiabilidad

El proyecto incluye dos pruebas breves de confiabilidad:

- sensibilidad al texto;
- degradación visual.

La primera prueba revisa si cambios en la redacción del caption modifican de forma importante el ranking. La segunda prueba revisa si el sistema mantiene su comportamiento cuando la imagen pierde calidad visual.

Los resultados se almacenan en:

- `results/pruebas_confiabilidad.csv`
- `results/resumen_confiabilidad.csv`
- `results/clasificacion_confiabilidad.json`

La clasificación final adoptada es: confiable solo en condiciones controladas.

## Explicabilidad

El proyecto incluye una revisión de explicabilidad sobre dos casos reales del experimento. Se analiza un caso de acierto y un caso de error o ambigüedad.

La explicación se basa en evidencia observable:

- atributos visuales de la imagen;
- caption recuperado por el modelo;
- resultado esperado;
- tipo de caso;
- límite de la explicación.

Los resultados se almacenan en:

- `results/explicabilidad.csv`
- `results/respuestas_explicabilidad.json`
- `figures/explicabilidad_casos.png`

La explicación no pretende describir el funcionamiento interno completo del modelo. Su finalidad es apoyar la revisión cualitativa de aciertos, errores y casos ambiguos.

## Sesgo y uso responsable

El proyecto incluye una ficha de uso responsable del sistema. Esta ficha identifica riesgos visuales, lingüísticos, culturales y de uso.

Los resultados se almacenan en:

- `results/ficha_uso_responsable.csv`
- `results/uso_responsable_resumen.json`

El uso recomendado del sistema es limitado. Puede utilizarse como ejercicio académico para explorar alineamiento imagen texto en condiciones controladas.

No se recomienda usarlo para identificación cultural, clasificación real de objetos, validación patrimonial ni toma de decisiones sin supervisión humana.

## Conclusión técnica

La conclusión técnica del trabajo se encuentra en:

- `results/conclusion_tecnica.md`
- `reporte_evaluacion_responsable.md`

El resultado principal es que OpenCLIP muestra una señal de alineamiento imagen texto superior al baseline aleatorio. Sin embargo, el sistema solo debe considerarse confiable en condiciones controladas. No se recomienda su uso para identificación cultural, clasificación real de objetos ni validación patrimonial.

## Cumplimiento de entregables

| Entregable | Archivo |
|---|---|
| Cuaderno adaptado | `notebooks/Cuaderno14_MCC225_resuelto.ipynb` |
| Reporte breve | `reporte_evaluacion_responsable.md` |
| Tabla de métricas | `results/metricas.csv` |
| Tabla de cinco casos | `results/casos_analizados.csv` |
| Ficha de uso responsable | `results/ficha_uso_responsable.csv` |
| Figura de ejemplos | `figures/ejemplos_evaluados.png` |
| Repositorio actualizado | `README.md` y archivos del proyecto |