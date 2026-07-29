# Auditoría del dataset sintético v1

## Propósito

Esta auditoría revisa la integridad, unicidad y unidad de relevancia
del dataset heredado de la Actividad 5. No modifica imágenes ni
captions.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Registros del manifiesto | 40 |
| Identificadores únicos | 40 |
| Captions almacenados | 200 |
| Textos de caption únicos | 76 |
| Firmas semánticas únicas | 30 |
| Firmas semánticas repetidas | 10 |
| Imágenes únicas por SHA-256 | 31 |
| Grupos de imágenes exactas | 9 |
| Consultas afectadas por duplicación semántica | 20 |
| Longitud esperada del ciclo conjunto | 30 |

## Origen del ciclo

El generador combina 6 configuraciones
y 5 paletas mediante índices modulares
independientes. La combinación completa se repite cada
30 registros.

Como el dataset contiene 40 registros,
los últimos 10 vuelven
a utilizar combinaciones ya observadas.

## Duplicados exactos

- `T001, T031`
- `T002, T032`
- `T003, T033`
- `T004, T034`
- `T005, T035`
- `T007, T037`
- `T008, T038`
- `T009, T039`
- `T010, T040`

## Problema de relevancia

La evaluación original asigna cinco captions positivos a cada
`image_id`. Sin embargo, 20
consultas pertenecen a firmas repetidas. En esos casos, otros cinco
captions corresponden a la misma firma semántica y pueden ser
contabilizados como negativos por el protocolo basado solamente en
`image_id`.

## Repetición textual

De 200 slots de captions, únicamente
76 textos son únicos. Además,
16 textos compartidos
describen más de una firma semántica.

Esto muestra que algunas plantillas son demasiado generales para
evaluar discriminación visual fina.

## Decisión metodológica

El dataset v1 se conserva como línea base histórica. La evaluación
final utilizará un dataset v2 con:

- unidad semántica explícita;
- control de duplicados;
- captions positivos discriminativos;
- negativos que modifiquen un único atributo;
- métricas por patrón y nivel de ambigüedad;
- prueba con configuraciones no vistas.

## Artefactos

- `results/auditoria_dataset_v1.json`
- `results/duplicados_semanticos_v1.csv`
- `results/captions_compartidos_v1.csv`
