# Diseño de los captions positivos del dataset v2

## 1. Objetivo

El dataset v2 utilizará captions controlados para evaluar recuperación imagen–texto sobre atributos geométricos y cromáticos observables.

Los captions no describen culturas, procedencias, periodos históricos, significados simbólicos ni autenticidad patrimonial.

---

## 2. Atributos expresados

Cada caption positivo completo expresa cinco atributos:

1. paleta;
2. motivo;
3. orientación;
4. composición;
5. simetría.

El nivel de ambigüedad no aparece en el caption. Se conserva exclusivamente como variable de estratificación experimental.

---

## 3. Cardinalidad

El corpus tiene 56 imágenes y cinco captions positivos por imagen:

$$
56 \times 5 = 280
$$

Cada `semantic_id` tendrá exactamente cinco captions.

Las cinco plantillas se aplicarán a todas las firmas semánticas. Por tanto, cada plantilla aparecerá exactamente 56 veces.

---

## 4. Control de atajos léxicos

La identidad de la plantilla no permite determinar por sí sola qué imagen es relevante, porque todas las firmas semánticas utilizan las mismas cinco plantillas.

Los captions completos deben ser globalmente únicos. La diferencia semántica depende de los valores de los cinco atributos y no de una plantilla exclusiva para un patrón o split.

Este control no elimina todos los posibles atajos. Los nombres explícitos de los atributos pueden facilitar la tarea. Por ello, la evaluación posterior incluirá:

- negativos con una única modificación;
- candidatos escritos con la misma plantilla;
- ablaciones cromáticas;
- métricas por atributo y patrón.

---

## 5. Caption canónico

`TPL_01` será la plantilla canónica.

Se utilizará posteriormente para construir la prueba de negativos difíciles. El caption positivo y sus cuatro negativos tendrán exactamente la misma estructura sintáctica y solo cambiarán sus valores de atributos.

---

## 6. Versión sin color

Cada caption tendrá una representación derivada sin términos cromáticos.

La versión sin color conserva:

- motivo;
- orientación;
- composición;
- simetría.

Al retirar la paleta, las siete imágenes que comparten un patrón producen la misma descripción estructural para una plantilla determinada.

Por ello, las versiones sin color no se evaluarán como 280 candidatos independientes. Se deduplicarán utilizando un `structure_id`.

El corpus estructural tendrá:

$$
8\ \text{patrones} \times 5\ \text{plantillas}
= 40\ \text{captions sin color}
$$

En esa condición, la relevancia se definirá por estructura y no por el `semantic_id` completo.

---

## 7. Idioma

Los captions se redactan en español porque el experimento original, el informe y la defensa utilizan este idioma.

Esta decisión limita la interpretación: el resultado también depende de la capacidad del encoder textual de OpenCLIP para representar captions en español.

No se afirmará que el comportamiento sería idéntico con captions en inglés u otros idiomas.

---

## 8. Longitud

Los captions tendrán entre 12 y 40 palabras según una tokenización simple basada en espacios.

La comprobación definitiva con el tokenizer de OpenCLIP se realizará antes de calcular embeddings. Ningún caption podrá exceder la longitud admitida por el encoder textual.

---

## 9. Restricciones de contenido

Se excluyen expresiones que atribuyan:

- identidad cultural;
- procedencia histórica;
- autenticidad;
- significado simbólico;
- validación patrimonial.

Los textos utilizarán únicamente vocabulario visual y geométrico observable.

---

## 10. Artefactos previstos

La siguiente etapa generará:

- `data/v2/captions_positivos_v2.csv`;
- `data/v2/captions_sin_color_v2.csv`;
- `data/v2/manifest_multimodal_v2.csv`;
- `results/v2/resumen_captions_positivos_v2.json`.

La generación se realizará únicamente después de validar y congelar el vocabulario actual.