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
