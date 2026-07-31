# Diseño de negativos difíciles del dataset v2

## 1. Objetivo

La prueba de negativos difíciles evalúa si el modelo distingue una descripción correcta de otras descripciones que difieren en un único atributo visual.

No sustituye la evaluación global de recuperación. Es una prueba local de sensibilidad semántica.

---

## 2. Unidad de evaluación

Cada una de las 56 imágenes define una consulta.

La consulta contiene cinco captions candidatos:

- un caption positivo;
- cuatro negativos difíciles.

Por tanto, el conjunto tendrá:

$$
56 \times 5 = 280
$$

relaciones candidatas entre imagen y caption.

El nivel de azar de la selección entre cinco candidatos es:

$$
\frac{1}{5}=0.20.
$$

---

## 3. Plantilla controlada

Todos los candidatos de una consulta utilizan `TPL_01`.

El positivo y los negativos comparten la misma estructura sintáctica. La diferencia se limita a los valores de los atributos.

Esto evita que el modelo identifique la respuesta correcta mediante una plantilla exclusiva.

---

## 4. Atributos controlados

Los atributos son:

1. `palette_id`;
2. `motif`;
3. `orientation`;
4. `composition`;
5. `symmetry`.

Cada negativo cambia exactamente uno de ellos y conserva los otros cuatro.

---

## 5. Omisión balanceada

Existen cinco atributos, pero solo cuatro negativos por consulta. En consecuencia, una categoría debe omitirse en cada consulta.

La categoría omitida rota de acuerdo con el orden de `image_id`.

Como 56 no es divisible entre 5, el balance óptimo es:

- un atributo omitido 12 veces;
- cuatro atributos omitidos 11 veces.

Así, los atributos serán modificados entre 44 y 45 veces.

---

## 6. Valor sustituto

El valor alternativo se elige mediante sucesor cíclico dentro del dominio congelado de cada atributo.

Si el valor original ocupa la posición $j$, se selecciona:

$$
j'=(j+1)\bmod |\mathcal D|.
$$

Este procedimiento garantiza que:

- el valor sustituto pertenece al vocabulario;
- el valor sustituto difiere del original;
- la generación es determinista;
- no se requiere muestreo aleatorio.

---

## 7. Posición del positivo

La respuesta correcta no ocupa una posición fija.

Para la consulta de índice cero-based $i$, la posición se calcula como:

$$
p_i =
\left(
2i+
\left\lfloor\frac{i}{5}\right\rfloor+
1
\right)
\bmod 5 + 1.
$$

La distribución resultante es 11, 11, 12, 11 y 11 apariciones en las posiciones 1 a 5.

La combinación entre atributo omitido y posición positiva cubre los 25 pares posibles.

---

## 8. Naturaleza contrafactual

La modificación de un único atributo puede producir una descripción que no corresponda a una combinación generada por el renderer.

Por ejemplo, podría conservarse una composición en bandas mientras se cambia únicamente el motivo a círculos concéntricos.

Este caso no se interpreta como una nueva clase válida. Se utiliza como intervención textual controlada para medir sensibilidad a un atributo.

---

## 9. Relevancia local

La etiqueta negativa se define respecto de la imagen de la consulta.

Un caption negativo puede describir correctamente otra imagen del corpus. Esto ocurrirá especialmente cuando se modifique la paleta y exista otra imagen con la misma estructura.

Por tanto:

- estos candidatos no se utilizarán como negativos globales;
- el solapamiento con positivos de otras imágenes se registrará;
- la evaluación será de elección forzada por consulta;
- no se alterará el ground truth multipositivo de recuperación global.

---

## 10. Métricas

La medida principal será `hard_negative_accuracy`.

También se reportarán:

- margen entre el positivo y el negativo mejor puntuado;
- exactitud por atributo cambiado;
- exactitud por patrón;
- exactitud por ambigüedad;
- exactitud por split;
- frecuencia de negativos que son positivos para otra consulta.

---

## 11. Límites interpretativos

La prueba permite estudiar sensibilidad a atributos visuales controlados.

No demuestra:

- reconocimiento de textiles andinos reales;
- identificación cultural;
- procedencia histórica;
- autenticidad;
- interpretación simbólica;
- robustez fuera del generador sintético.
---

## 12. Solapamientos globales previstos

La simulación previa a la generación detectó 50 negativos que coinciden con captions positivos canónicos de otras imágenes:

- 44 cambios de `palette_id`;
- 6 cambios de `orientation`;
- 0 cambios de motivo, composición o simetría.

Los 44 casos cromáticos son consecuencia directa del diseño factorial: cada uno de los ocho patrones existe con las siete paletas.

Los seis casos de orientación aparecen cuando una descripción de bandas horizontales cambia únicamente su orientación a vertical. El resultado coincide con la descripción positiva del patrón de bandas verticales con la misma paleta. En una de las siete imágenes de bandas horizontales, la orientación es el atributo omitido por el calendario rotativo, por lo que quedan seis casos.

Estos solapamientos no se consideran errores ni fuga de etiquetas dentro de la prueba de elección forzada. Constituyen negativos válidos respecto de la imagen consultada, pero positivos para otra imagen.

Por esta razón:

- la relevancia de los candidatos es local a cada consulta;
- los candidatos no deben incorporarse como negativos globales;
- la evaluación global de recuperación conserva su ground truth multipositivo independiente;
- el informe debe distinguir negativos contrafactuales de negativos que describen otra imagen real.

La generación también cubre las 29 transiciones cíclicas posibles en los cinco dominios de atributos:

$$
7 + 7 + 6 + 5 + 4 = 29.
$$
