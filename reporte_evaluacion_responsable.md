# Evaluacion responsable de un prototipo multimodal para patrones textiles andinos generados

## 1. Proyecto personal y tarea multimodal

El proyecto desarrollado para el curso MCC225 consiste en evaluar de manera responsable un prototipo multimodal orientado a la recuperacion imagen texto sobre patrones textiles andinos generados. El objetivo no es construir un sistema definitivo de reconocimiento visual, sino analizar si un modelo multimodal puede asociar imagenes generadas con descripciones textuales controladas.

El problema abordado es la dificultad de evaluar sistemas multimodales solo a partir de ejemplos visuales exitosos. Un modelo puede mostrar resultados aparentemente correctos en algunos casos, pero fallar cuando cambia la redaccion de la consulta, cuando existen patrones parecidos, cuando la imagen es ambigua o cuando la descripcion textual no contiene suficiente informacion. Por esta razon, la actividad se enfoca en medir resultados, observar errores, revisar confiabilidad y establecer limites de uso.

El dominio de aplicacion corresponde al analisis exploratorio de imagenes generadas de inspiracion textil andina. Las modalidades consideradas son imagen y texto. La tarea multimodal evaluada sera principalmente recuperacion imagen texto. En esta tarea, dada una imagen, el sistema debe recuperar la descripcion textual mas adecuada entre varias opciones candidatas. De manera complementaria, se podra revisar la recuperacion texto imagen, donde una descripcion se usa para buscar la imagen correspondiente.

La evaluacion se realizara con un conjunto pequeño y controlado de imagenes generadas, acompañadas de captions o descripciones construidas manualmente. Estas descripciones incluiran atributos visuales observables como colores dominantes, presencia de bandas, motivos geometricos, composicion, simetria y repeticion modular.

El usuario previsto es un estudiante o evaluador academico que desea explorar el comportamiento de modelos multimodales en un contexto controlado. El sistema no esta diseñado para realizar identificacion cultural, clasificacion historica ni validacion patrimonial. Su uso debe entenderse como una evaluacion tecnica preliminar de alineamiento entre imagen y texto.

El riesgo principal del sistema es interpretar una coincidencia visual como si fuera una identificacion confiable. Por ello, la evaluacion debe reportar no solo los aciertos, sino tambien los errores, los casos ambiguos, la sensibilidad del sistema y las limitaciones del conjunto de datos utilizado.

## Ficha minima del proyecto

| Elemento                     | Respuesta                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Nombre del proyecto          | Evaluacion responsable de un prototipo multimodal para patrones textiles andinos generados                                                                                     |
| Problema que aborda          | Evaluar si un modelo multimodal puede alinear imagenes generadas de inspiracion textil con descripciones textuales controladas, identificando aciertos, errores y limitaciones |
| Dominio de aplicacion        | Evaluacion multimodal, recuperacion imagen texto y analisis visual exploratorio de patrones textiles                                                                           |
| Modalidades usadas           | Imagen y texto                                                                                                                                                                 |
| Tarea multimodal evaluada    | Recuperacion imagen texto y, de forma complementaria, recuperacion texto imagen                                                                                                |
| Usuario previsto             | Estudiante, investigador o evaluador academico en un entorno experimental controlado                                                                                           |
| Riesgo principal del sistema | Confundir similitud visual con identificacion confiable o clasificacion real del objeto representado                                                                           |

## Pregunta breve

¿Que parte del proyecto se puede evaluar con evidencia en esta etapa del curso?

En esta etapa se puede evaluar con evidencia el alineamiento entre imagenes generadas y descripciones textuales controladas. Esto incluye medir si el modelo recupera el texto correcto para una imagen, comparar su desempeño contra un baseline, analizar casos de acierto y error, y observar si el comportamiento se mantiene ante variaciones pequeñas en las consultas o en las imagenes. No se evaluara autenticidad, procedencia historica ni clasificacion patrimonial.
