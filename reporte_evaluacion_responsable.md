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