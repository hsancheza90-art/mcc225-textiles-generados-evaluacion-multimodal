# Auditoría visual de los patrones del dataset v2

## 1. Objetivo

Esta auditoría verifica, antes de generar las 56 imágenes finales, que las ocho funciones de renderizado representen atributos visuales distinguibles y coherentes con la especificación experimental.

La inspección se realizó utilizando una única paleta acromática, `blanco_negro_gris`, para reducir el efecto del color y concentrar la revisión en la estructura geométrica.

La lámina evaluada se encuentra en:

`figures/v2/previsualizacion_patrones_v2.png`

---

## 2. Resultado general

La previsualización fue aprobada.

Los ocho patrones:

- tienen archivos y hashes de píxeles diferentes;
- pueden distinguirse visualmente;
- corresponden a los ocho `pattern_id` de la configuración;
- son reproducibles con la misma semilla;
- mantienen una resolución de 512 × 512 píxeles;
- no requieren modificaciones antes de generar el dataset completo.

---

## 3. Evaluación por patrón

| Patrón | Evidencia observada | Estado |
|---|---|---|
| `bands_horizontal` | Bandas paralelas y motivos repetidos sobre el eje horizontal. | Aprobado |
| `bands_vertical` | Bandas paralelas y motivos repetidos sobre el eje vertical. | Aprobado |
| `diamonds_central` | Rombos anidados, centro visual dominante y simetría axial. | Aprobado |
| `grecas_modular` | Grecas escalonadas repetidas dentro de una organización modular. | Aprobado |
| `grid_reticular` | Retícula regular con repetición en filas y columnas. | Aprobado |
| `mixed_asymmetric` | Formas, bloques y líneas distribuidos sin simetría global. | Aprobado |
| `chevrons_diagonal` | Trazos diagonales que forman unidades repetidas en forma de V. | Aprobado |
| `circles_concentric` | Anillos concéntricos con centro y organización radial visibles. | Aprobado |

---

## 4. Precisiones semánticas

### 4.1. Orientación de los chevrones

En `chevrons_diagonal`, el atributo `orientation=diagonal` describe la orientación local dominante de los trazos que forman cada chevrón.

Las unidades aparecen organizadas en filas, pero sus segmentos visuales principales son diagonales. Esta definición debe conservarse al construir captions positivos y negativos.

### 4.2. Nivel de ambigüedad

El campo `ambiguity_level` es una etiqueta de diseño asignada por el generador.

No debe interpretarse directamente como dificultad empírica. Por ejemplo, `mixed_asymmetric` puede resultar visualmente muy distintivo aunque su nivel configurado sea `alto`.

La dificultad real se evaluará posteriormente mediante:

- Recall@K;
- MRR;
- nDCG@10;
- exactitud con negativos difíciles;
- margen entre el positivo y el mejor negativo;
- errores por patrón.

---

## 5. Observaciones sobre posibles confusores

Los patrones presentan diferencias estructurales marcadas. Esto favorece una evaluación inicial controlada, pero también puede facilitar la recuperación mediante señales globales simples.

En particular:

- `mixed_asymmetric` posee bloques amplios muy distintivos;
- `circles_concentric` es el único patrón con organización radial;
- `grid_reticular` presenta una densidad geométrica superior;
- los patrones de bandas pueden distinguirse por orientación global.

Por ello, un buen rendimiento global no demostrará por sí mismo comprensión composicional fina. La evaluación deberá incluir negativos que cambien un único atributo y ablaciones cromáticas.

---

## 6. Decisión

Se aprueban las ocho funciones de renderizado para generar las 56 imágenes del dataset v2.

No se modificará el diseño visual después de observar las métricas principales, salvo que se detecte un error técnico documentado.

La previsualización se conserva como evidencia del control realizado antes de la generación final.