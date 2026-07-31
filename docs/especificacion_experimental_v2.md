# Especificación experimental del benchmark v2

## 1. Identificación

**Proyecto:** Evaluación controlada de OpenCLIP para recuperación entre imágenes y textos con patrones textiles sintéticos.

**Estudiante:** Henry Sánchez Alvarado.

**Curso:** MCC225: IA Generativa y Aprendizaje Multimodal.

**Versión del protocolo:** 2.0.

**Estado:** protocolo definido antes de generar el dataset v2 y antes de observar sus resultados.

---

## 2. Alcance

El proyecto evalúa recuperación cruzada entre imágenes y textos sobre patrones geométricos sintéticos de inspiración textil andina.

El objetivo no es reconocer textiles andinos reales, identificar culturas, determinar procedencias, autenticar piezas, interpretar símbolos ni realizar clasificación patrimonial.

La expresión «inspiración textil andina» describe solamente la motivación visual del generador. Las etiquetas del benchmark representan atributos geométricos y cromáticos observables.

---

## 3. Problema multimodal

El problema es multimodal porque relaciona dos representaciones diferentes:

* **Imagen:** contiene paleta, motivo, orientación, composición y simetría observables.
* **Texto:** expresa lingüísticamente esos atributos mediante captions controlados.

El sistema debe alinear ambas modalidades dentro de un espacio compartido de embeddings y ordenar los captions según su similitud con una imagen de consulta.

La tarea principal es **retrieval cruzado de imagen a texto**. No es clasificación cultural, captioning generativo, VQA ni reconocimiento de objetos patrimoniales.

---

## 4. Pregunta experimental

> ¿En qué medida OpenCLIP ViT-B-32 recupera captions correctos de patrones geométricos sintéticos cuando debe distinguirlos de descripciones que cambian un solo atributo visual, y cuánto de su rendimiento puede explicarse únicamente por la paleta de color?

La pregunta es falsable porque el rendimiento puede compararse con:

1. un ranking aleatorio;
2. una línea base basada únicamente en color;
3. condiciones donde se elimina la información cromática;
4. negativos que modifican un solo atributo;
5. patrones y paletas no presentes en el subconjunto ID.

---

## 5. Hipótesis

### H1. Alineamiento multimodal

OpenCLIP obtendrá mejores resultados que el baseline aleatorio y que el baseline basado únicamente en color en Recall@1, MRR y nDCG@10.

### H2. Discriminación composicional mínima

En una prueba con un caption positivo y cuatro negativos difíciles, OpenCLIP obtendrá una exactitud superior al nivel de azar de 0.20.

### H3. Información estructural más allá del color

Al retirar la información cromática de la imagen, del caption o de ambas modalidades, el rendimiento disminuirá, pero permanecerá por encima del baseline aleatorio si el modelo utiliza motivo, orientación, composición o simetría.

### H4. Generalización controlada

El rendimiento será menor en condiciones OOD que en ID, pero deberá permanecer por encima del baseline aleatorio para sostener que existe generalización parcial.

---

## 6. Criterios de refutación

La conclusión principal no estará respaldada cuando ocurra alguno de los siguientes resultados:

1. OpenCLIP no supera al baseline de color en las métricas principales.
2. El intervalo bootstrap de la diferencia entre OpenCLIP y el baseline de color incluye ampliamente cero.
3. La exactitud con negativos difíciles no supera el azar de 0.20.
4. El rendimiento cae hasta niveles cercanos al azar al eliminar el color.
5. Los resultados favorables se concentran únicamente en captions que contienen plantillas o palabras repetidas.
6. El rendimiento OOD no supera al baseline aleatorio.
7. La mejora global desaparece al reportar resultados por motivo o nivel de ambigüedad.

Un resultado desfavorable no se ocultará: se interpretará como evidencia sobre los límites de OpenCLIP en este benchmark.

---

## 7. Modelo principal

El modelo principal será:

* arquitectura: OpenCLIP ViT-B-32;
* pesos: `laion2b_s34b_b79k`;
* régimen: zero-shot;
* encoder visual y encoder textual congelados;
* embeddings normalizados mediante norma L2;
* similitud calculada mediante producto punto entre embeddings normalizados.

OpenCLIP es un dual encoder. La imagen y el texto se codifican por separado y solo interactúan al calcular su similitud. Esta arquitectura es eficiente para retrieval, pero puede perder interacciones finas entre atributos.

No se realizará fine-tuning, LoRA ni QLoRA porque el objetivo es evaluar la representación preentrenada, no adaptar el modelo al generador sintético.

---

## 8. Unidad de análisis y relevancia

### 8.1. Unidad de consulta

Una imagen sintética individual.

### 8.2. Unidad candidata

Un caption textual individual.

### 8.3. Unidad semántica

Cada caso tendrá un `semantic_id` construido a partir de:

* paleta;
* motivo;
* orientación;
* composición;
* simetría.

El nivel de ambigüedad será una variable derivada para estratificar resultados, pero no formará parte del identificador semántico.

### 8.4. Relevancia

Un caption será positivo cuando sus cinco atributos coincidan con el `semantic_id` de la imagen.

Cada imagen tendrá cinco captions positivos. Recall@K deberá considerar correcto el ranking cuando al menos uno de esos captions aparezca dentro del top K.

No se utilizará únicamente `image_id` para definir relevancia.

---

## 9. Diseño del dataset v2

El dataset tendrá 56 imágenes distribuidas en cuatro subconjuntos.

| Split         | Descripción                      | Número esperado |
| ------------- | -------------------------------- | --------------: |
| `id`          | Patrones base con paletas base   |              30 |
| `ood_palette` | Patrones base con paletas nuevas |              12 |
| `ood_pattern` | Patrones nuevos con paletas base |              10 |
| `ood_both`    | Patrones y paletas nuevos        |               4 |
| **Total**     |                                  |          **56** |

La condición OOD se refiere únicamente a la construcción del benchmark. No implica que OpenCLIP nunca haya observado conceptos similares durante su preentrenamiento.

No se generarán combinaciones mediante dos ciclos modulares independientes. Se construirá explícitamente el producto cartesiano de patrones y paletas correspondiente a cada split.

Cada combinación de patrón y paleta aparecerá una sola vez.

---

## 10. Atributos controlados

Los captions y negativos utilizarán cinco atributos:

1. `palette_id`;
2. `motif`;
3. `orientation`;
4. `composition`;
5. `symmetry`.

El manifiesto también incluirá:

* `image_id`;
* `semantic_id`;
* `pattern_id`;
* `image_path`;
* `split`;
* `ambiguity_level`;
* `seed`;
* cinco captions positivos;
* procedencia sintética;
* restricciones de uso.

---

## 11. Captions positivos

Cada imagen tendrá cinco captions positivos.

Todos los captions positivos deberán incluir los cinco atributos discriminativos. Se utilizarán variaciones sintácticas, pero no captions genéricos que puedan describir simultáneamente muchas imágenes.

Los cinco captions deberán:

* ser no vacíos;
* ser diferentes entre sí;
* describir una única firma semántica;
* utilizar vocabulario observable;
* evitar afirmaciones culturales;
* permanecer por debajo del límite de tokens del encoder textual.

---

## 12. Negativos difíciles

Para cada consulta se construirá:

* un caption canónico positivo;
* cuatro captions negativos.

Cada negativo modificará exactamente uno de estos atributos:

* paleta;
* motivo;
* orientación;
* composición;
* simetría.

Los otros cuatro atributos permanecerán sin cambios.

Como existen cinco tipos de modificación y se utilizarán cuatro negativos por consulta, el atributo omitido rotará de manera determinista. En el conjunto completo, los cinco tipos deberán aparecer de manera balanceada.

Todos los candidatos de una prueba utilizarán la misma plantilla sintáctica. De este modo, una elección correcta no podrá explicarse solamente por diferencias de estilo o longitud.

---

## 13. Líneas base

### 13.1. Baseline aleatorio

Ordenará candidatos aleatoriamente con semilla fija. Representa el funcionamiento esperado sin información de imagen ni texto.

### 13.2. Baseline de color

Representará la imagen mediante un descriptor cromático HSV y cada caption mediante la paleta indicada en sus metadatos.

La similitud dependerá exclusivamente del color. Motivo, orientación, composición y simetría serán ignorados.

Este baseline permitirá comprobar si OpenCLIP aporta información estructural adicional o si su rendimiento se explica principalmente por la paleta.

---

## 14. Experimentos

### E0. Reproducción del benchmark v1

Reproducir las métricas heredadas de la Actividad 5 y conservarlas como referencia histórica.

### E1. Retrieval global v2

Evaluar OpenCLIP sobre las 56 imágenes y los 280 captions positivos.

### E2. Negativos difíciles

Evaluar una selección forzada entre un positivo y cuatro negativos de un solo atributo.

### E3. Comparación de baselines

Comparar OpenCLIP, ranking aleatorio y descriptor de color bajo el mismo manifiesto y protocolo.

### E4. Ablaciones cromáticas

Evaluar:

1. imagen original y caption completo;
2. imagen en escala de grises y caption completo;
3. imagen original y caption sin términos de color;
4. imagen en escala de grises y caption sin términos de color.

### E5. Evaluación ID/OOD

Reportar por separado:

* ID;
* OOD-paleta;
* OOD-patrón;
* OOD-combinado.

### E6. Estabilidad y estratificación

Calcular intervalos bootstrap y reportar métricas por:

* patrón;
* nivel de ambigüedad;
* split;
* tipo de negativo;
* condición de ablación.

---

## 15. Métricas

### Métricas principales

1. **Recall@1:** indica si al menos un caption positivo ocupa la primera posición.
2. **MRR:** resume la posición del primer caption positivo.
3. **nDCG@10:** evalúa la calidad del ranking cuando existen varios captions relevantes.

### Métricas complementarias

* Recall@5;
* exactitud en negativos difíciles;
* margen entre el score positivo y el mejor negativo;
* diferencia frente al baseline;
* intervalo bootstrap del 95 %.

Recall@1 no deberá interpretarse de forma aislada porque no utiliza toda la información del ranking y puede ocultar la posición de otros captions positivos.

---

## 16. Bootstrap

Se utilizarán 2 000 remuestreos de consultas con semilla 225.

Los intervalos serán descriptivos debido al tamaño limitado del benchmark. No se afirmará generalización estadística hacia textiles reales.

Cuando se comparen dos condiciones sobre las mismas consultas, se utilizará bootstrap pareado.

---

## 17. Control de fuga y duplicados

El dataset v2 deberá cumplir:

* `image_id` único;
* `semantic_id` único;
* rutas existentes;
* hashes de imagen únicos;
* cinco captions positivos no vacíos;
* ausencia de captions positivos compartidos entre firmas diferentes;
* separación explícita entre patrones base y patrones OOD;
* separación explícita entre paletas base y paletas OOD;
* ninguna combinación repetida;
* semillas registradas por imagen.

No se utilizará una división aleatoria posterior. Los splits se definirán antes de generar las imágenes.

---

## 18. Grounding

En este proyecto, grounding significa que un caption relevante debe estar respaldado por atributos observables en la imagen:

* colores;
* formas;
* orientación;
* distribución;
* simetría.

El benchmark no evalúa grounding cultural, histórico, simbólico ni geográfico.

---

## 19. Amenazas a la validez

1. Las imágenes y captions provienen de reglas conocidas.
2. El vocabulario visual es limitado.
3. Las imágenes sintéticas son más regulares que las piezas reales.
4. El generador puede favorecer atributos fáciles de reconocer.
5. OpenCLIP pudo observar conceptos relacionados durante su preentrenamiento.
6. OOD describe el benchmark y no el conjunto de entrenamiento original de OpenCLIP.
7. El tamaño de 56 imágenes limita la estabilidad de comparaciones pequeñas.
8. La línea base de color no representa todos los descriptores visuales clásicos posibles.
9. El resultado no permite realizar afirmaciones culturales ni patrimoniales.

---

## 20. Afirmaciones permitidas

Podrá afirmarse que:

* OpenCLIP muestra o no muestra alineamiento parcial entre imágenes sintéticas y captions controlados;
* el modelo supera o no supera las líneas base evaluadas;
* el color explica una parte determinada del rendimiento;
* ciertos atributos o niveles de ambigüedad presentan más errores;
* existe o no existe generalización parcial dentro de las condiciones OOD definidas.

---

## 21. Afirmaciones no permitidas

No podrá afirmarse que:

* OpenCLIP reconoce textiles andinos reales;
* identifica una cultura, comunidad, periodo o procedencia;
* interpreta significados simbólicos;
* autentica objetos patrimoniales;
* el rendimiento sintético se transfiere automáticamente a colecciones reales;
* el sistema es confiable para decisiones culturales o curatoriales.

---

## 22. Evidencia esperada en el repositorio

| Decisión                 | Evidencia prevista                           |
| ------------------------ | -------------------------------------------- |
| Alcance y pregunta       | `docs/especificacion_experimental_v2.md`     |
| Configuración ejecutable | `config/experimento_v2.json`                 |
| Dataset                  | `data/v2/manifest_v2.csv`                    |
| Positivos                | `data/v2/captions_positivos_v2.csv`          |
| Negativos                | `data/v2/hard_negatives_v2.csv`              |
| Integridad               | `tests/test_dataset_v2.py`                   |
| Métricas                 | `results/v2/metricas_globales.csv`           |
| Patrón y ambigüedad      | `results/v2/metricas_estratificadas.csv`     |
| Baselines                | `results/v2/comparacion_baselines.csv`       |
| Ablaciones               | `results/v2/metricas_ablaciones.csv`         |
| ID/OOD                   | `results/v2/metricas_splits.csv`             |
| Casos                    | `results/v2/casos_analizados.csv`            |
| Entorno                  | `results/v2/configuracion_experimental.json` |
| Informe                  | `report/Sanchez_Henry_MCC225_Final.pdf`      |

---

## 23. Regla de congelamiento

Este protocolo se registra antes de generar el dataset v2.

Cualquier cambio posterior en:

* pregunta experimental;
* hipótesis;
* número de ejemplos;
* splits;
* métricas principales;
* definición de relevancia;
* baselines;
* criterios de refutación;

deberá documentarse mediante un commit explícito y justificarse en el informe final.
