"""Genera el informe final v2 a partir de artefactos auditados."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "informe_final_v2.json"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "tablas_maestras"
    / "metricas_maestras_v2.csv"
)

CASES_PATH = (
    PROJECT_ROOT
    / "results"
    / "casos_analizados.csv"
)

RELIABILITY_PATH = (
    PROJECT_ROOT
    / "results"
    / "pruebas_confiabilidad.csv"
)

EXPLAINABILITY_PATH = (
    PROJECT_ROOT
    / "results"
    / "explicabilidad.csv"
)

RESPONSIBLE_USE_PATH = (
    PROJECT_ROOT
    / "results"
    / "ficha_uso_responsable.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reporte_evaluacion_responsable.md"
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene CRLF."
    )

    return json.loads(
        raw.decode("utf-8")
    )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def metric_index(
    rows: list[dict[str, str]],
) -> dict[
    tuple[str, str, str, str],
    float,
]:
    result = {}

    for row in rows:
        key = (
            row["experiment_id"],
            row["source_section"],
            row["condition"],
            row["metric_name"],
        )

        assert key not in result, (
            f"Métrica duplicada: {key}"
        )

        result[key] = float(
            row["metric_value"]
        )

    return result


def metric(
    index: dict[
        tuple[str, str, str, str],
        float,
    ],
    experiment_id: str,
    source_section: str,
    condition: str,
    metric_name: str,
) -> float:
    key = (
        experiment_id,
        source_section,
        condition,
        metric_name,
    )

    assert key in index, (
        f"No existe la métrica {key}."
    )

    return index[key]


def format_metric(
    value: float,
) -> str:
    return f"{value:.3f}"


def format_percentage(
    value: float,
) -> str:
    return f"{100.0 * value:.1f} %"


def clean_text(
    value: str,
) -> str:
    replacements = {
        "comoprimera": "como primera",
        "visualmenteparcidos": (
            "visualmente parecidos"
        ),
        "visualmenteparecidos": (
            "visualmente parecidos"
        ),
        "puede estarparcialmente": (
            "puede estar parcialmente"
        ),
        "sebasa": "se basa",
        "delranking": "del ranking",
        "motivosde": "motivos de",
        "cuadriculageometrica": (
            "cuadrícula geométrica"
        ),
        "textualcon": "textual con",
        "primeraposición": (
            "primera posición"
        ),
    }

    cleaned = value.strip()

    for old, new in replacements.items():
        cleaned = cleaned.replace(
            old,
            new,
        )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    )

    return cleaned


def label_from_identifier(
    value: str,
) -> str:
    return (
        value
        .replace("_", " ")
        .strip()
        .capitalize()
    )


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    assert headers
    assert rows

    column_count = len(headers)

    assert all(
        len(row) == column_count
        for row in rows
    )

    if alignments is None:
        alignments = [
            "left"
            for _ in headers
        ]

    assert len(alignments) == column_count

    alignment_cells = []

    for alignment in alignments:
        if alignment == "right":
            alignment_cells.append(
                "---:"
            )
        elif alignment == "center":
            alignment_cells.append(
                ":---:"
            )
        else:
            alignment_cells.append(
                "---"
            )

    lines = [
        "| "
        + " | ".join(headers)
        + " |",
        "| "
        + " | ".join(
            alignment_cells
        )
        + " |",
    ]

    for row in rows:
        normalized = [
            str(cell)
            .replace("|", r"\|")
            .replace("\n", " ")
            for cell in row
        ]

        lines.append(
            "| "
            + " | ".join(normalized)
            + " |"
        )

    return "\n".join(lines)


def reliability_summary(
    rows: list[dict[str, str]],
) -> list[list[str]]:
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["prueba"]].append(
            row
        )

    reliability_labels = {
        "sensibilidad_al_texto": (
            "Sensibilidad al texto"
        ),
        "degradacion_visual": (
            "Degradación visual"
        ),
    }

    assert set(grouped) == set(
        reliability_labels
    ), (
        "Pruebas de confiabilidad "
        f"inesperadas: {sorted(grouped)}"
    )

    output = []

    for test_name in sorted(grouped):
        group = grouped[test_name]

        rank_changes = [
            abs(
                float(
                    row["cambio_rank"]
                )
            )
            for row in group
        ]

        categories = Counter(
            row["cambio_observado"]
            for row in group
        )

        mean_change = (
            sum(rank_changes)
            / len(rank_changes)
        )

        category_text = ", ".join(
            f"{category}: "
            f"{categories.get(category, 0)}"
            for category in (
                "bajo",
                "medio",
                "alto",
            )
        )

        output.append(
            [
                reliability_labels[
                    test_name
                ],
                str(len(group)),
                f"{mean_change:.2f}",
                category_text,
                clean_text(
                    group[0][
                        "interpretacion"
                    ]
                ),
            ]
        )

    return output


def responsible_use_index(
    rows: list[dict[str, str]],
) -> dict[str, str]:
    result = {}

    for row in rows:
        key = row["aspecto"]

        assert key not in result

        result[key] = clean_text(
            row["respuesta"]
        )

    return result


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
    )

    metrics = metric_index(
        load_csv(
            METRICS_PATH
        )
    )

    cases = load_csv(
        CASES_PATH
    )

    reliability = load_csv(
        RELIABILITY_PATH
    )

    explainability = load_csv(
        EXPLAINABILITY_PATH
    )

    responsible_use = (
        responsible_use_index(
            load_csv(
                RESPONSIBLE_USE_PATH
            )
        )
    )

    assert len(cases) == 5
    assert len(reliability) == 10
    assert len(explainability) == 2

    # E1 y E3: recuperación exacta global.
    openclip_r1 = metric(
        metrics,
        "E3",
        "overall_metrics",
        "openclip",
        "recall_at_1",
    )

    openclip_r5 = metric(
        metrics,
        "E3",
        "overall_metrics",
        "openclip",
        "recall_at_5",
    )

    openclip_mrr = metric(
        metrics,
        "E3",
        "overall_metrics",
        "openclip",
        "mrr",
    )

    openclip_ndcg = metric(
        metrics,
        "E3",
        "overall_metrics",
        "openclip",
        "ndcg_at_10",
    )

    random_r1 = metric(
        metrics,
        "E3",
        "overall_metrics",
        "random",
        "recall_at_1",
    )

    random_r5 = metric(
        metrics,
        "E3",
        "overall_metrics",
        "random",
        "recall_at_5",
    )

    random_mrr = metric(
        metrics,
        "E3",
        "overall_metrics",
        "random",
        "mrr",
    )

    random_ndcg = metric(
        metrics,
        "E3",
        "overall_metrics",
        "random",
        "ndcg_at_10",
    )

    hsv_r1 = metric(
        metrics,
        "E3",
        "overall_metrics",
        "color_histogram",
        "recall_at_1",
    )

    hsv_r5 = metric(
        metrics,
        "E3",
        "overall_metrics",
        "color_histogram",
        "recall_at_5",
    )

    hsv_mrr = metric(
        metrics,
        "E3",
        "overall_metrics",
        "color_histogram",
        "mrr",
    )

    hsv_ndcg = metric(
        metrics,
        "E3",
        "overall_metrics",
        "color_histogram",
        "ndcg_at_10",
    )

    # E2: negativos difíciles.
    hard_accuracy = metric(
        metrics,
        "E2",
        "overall_metrics",
        "openclip_hard_negatives",
        "hard_negative_accuracy",
    )

    hard_mrr = metric(
        metrics,
        "E2",
        "overall_metrics",
        "openclip_hard_negatives",
        "mrr",
    )

    hard_ndcg = metric(
        metrics,
        "E2",
        "overall_metrics",
        "openclip_hard_negatives",
        "ndcg_at_10",
    )

    pairwise_win_rate = metric(
        metrics,
        "E2",
        "overall_metrics",
        "openclip_hard_negatives",
        "pairwise_win_rate",
    )

    mean_paired_difference = metric(
        metrics,
        "E2",
        "overall_metrics",
        "openclip_hard_negatives",
        "mean_paired_difference",
    )

    # E4: recuperación exacta en gris.
    gray_r1 = metric(
        metrics,
        "E4",
        "exact_overall_metrics",
        "grayscale_image_full_caption",
        "recall_at_1",
    )

    gray_r5 = metric(
        metrics,
        "E4",
        "exact_overall_metrics",
        "grayscale_image_full_caption",
        "recall_at_5",
    )

    gray_mrr = metric(
        metrics,
        "E4",
        "exact_overall_metrics",
        "grayscale_image_full_caption",
        "mrr",
    )

    gray_ndcg = metric(
        metrics,
        "E4",
        "exact_overall_metrics",
        "grayscale_image_full_caption",
        "ndcg_at_10",
    )

    structural_conditions = (
        (
            "original_image_full_caption",
            "Imagen original + caption completo",
        ),
        (
            "grayscale_image_full_caption",
            "Imagen gris + caption completo",
        ),
        (
            "original_image_caption_without_color",
            "Imagen original + caption sin color",
        ),
        (
            "grayscale_image_caption_without_color",
            "Imagen gris + caption sin color",
        ),
    )

    structural_rows = []

    for condition, label in (
        structural_conditions
    ):
        structural_rows.append(
            [
                label,
                format_metric(
                    metric(
                        metrics,
                        "E4",
                        (
                            "structural_"
                            "overall_metrics"
                        ),
                        condition,
                        "structural_hit_at_1",
                    )
                ),
                format_metric(
                    metric(
                        metrics,
                        "E4",
                        (
                            "structural_"
                            "overall_metrics"
                        ),
                        condition,
                        "structural_hit_at_5",
                    )
                ),
                format_metric(
                    metric(
                        metrics,
                        "E4",
                        (
                            "structural_"
                            "overall_metrics"
                        ),
                        condition,
                        (
                            "structural_"
                            "fractional_recall_at_5"
                        ),
                    )
                ),
                format_metric(
                    metric(
                        metrics,
                        "E4",
                        (
                            "structural_"
                            "overall_metrics"
                        ),
                        condition,
                        "structural_mrr",
                    )
                ),
                format_metric(
                    metric(
                        metrics,
                        "E4",
                        (
                            "structural_"
                            "overall_metrics"
                        ),
                        condition,
                        "structural_ndcg_at_10",
                    )
                ),
            ]
        )

    case_rows = []

    for row in cases:
        case_rows.append(
            [
                row["caso_id"],
                row["image_id"],
                label_from_identifier(
                    row["tipo_caso"]
                ),
                row["rank_correcto"],
                label_from_identifier(
                    row["tipo_error"]
                ),
                clean_text(
                    row[
                        "explicacion_breve"
                    ]
                ),
            ]
        )

    explainability_rows = []

    for row in explainability:
        explainability_rows.append(
            [
                row["caso_id"],
                row["image_id"],
                label_from_identifier(
                    row["tipo_caso"]
                ),
                clean_text(
                    row[
                        "explicacion_propuesta"
                    ]
                ),
                clean_text(
                    row[
                        "limite_explicacion"
                    ]
                ),
            ]
        )

    exact_comparison_table = (
        markdown_table(
            [
                "Método",
                "R@1",
                "R@5",
                "MRR",
                "nDCG@10",
            ],
            [
                [
                    "Aleatorio",
                    format_metric(
                        random_r1
                    ),
                    format_metric(
                        random_r5
                    ),
                    format_metric(
                        random_mrr
                    ),
                    format_metric(
                        random_ndcg
                    ),
                ],
                [
                    "Histograma HSV",
                    format_metric(
                        hsv_r1
                    ),
                    format_metric(
                        hsv_r5
                    ),
                    format_metric(
                        hsv_mrr
                    ),
                    format_metric(
                        hsv_ndcg
                    ),
                ],
                [
                    "OpenCLIP",
                    format_metric(
                        openclip_r1
                    ),
                    format_metric(
                        openclip_r5
                    ),
                    format_metric(
                        openclip_mrr
                    ),
                    format_metric(
                        openclip_ndcg
                    ),
                ],
                [
                    (
                        "OpenCLIP con "
                        "imágenes grises"
                    ),
                    format_metric(
                        gray_r1
                    ),
                    format_metric(
                        gray_r5
                    ),
                    format_metric(
                        gray_mrr
                    ),
                    format_metric(
                        gray_ndcg
                    ),
                ],
            ],
            [
                "left",
                "right",
                "right",
                "right",
                "right",
            ],
        )
    )

    hard_negative_table = (
        markdown_table(
            [
                "Métrica",
                "Resultado",
                "Lectura",
            ],
            [
                [
                    (
                        "Exactitud ante "
                        "negativos difíciles"
                    ),
                    format_metric(
                        hard_accuracy
                    ),
                    (
                        "El caption positivo "
                        "queda primero en "
                        f"{format_percentage(hard_accuracy)} "
                        "de las unidades evaluadas."
                    ),
                ],
                [
                    "MRR",
                    format_metric(
                        hard_mrr
                    ),
                    (
                        "El positivo suele "
                        "aparecer en posiciones "
                        "altas dentro de las "
                        "cinco alternativas."
                    ),
                ],
                [
                    "nDCG@10",
                    format_metric(
                        hard_ndcg
                    ),
                    (
                        "Resume la calidad del "
                        "ordenamiento local."
                    ),
                ],
                [
                    "Victorias pareadas",
                    format_metric(
                        pairwise_win_rate
                    ),
                    (
                        "El positivo supera "
                        "individualmente a un "
                        "negativo controlado en "
                        f"{format_percentage(pairwise_win_rate)} "
                        "de las comparaciones."
                    ),
                ],
                [
                    (
                        "Diferencia pareada "
                        "media"
                    ),
                    format_metric(
                        mean_paired_difference
                    ),
                    (
                        "La diferencia media "
                        "de similitud es positiva, "
                        "aunque de magnitud pequeña."
                    ),
                ],
            ],
            [
                "left",
                "right",
                "left",
            ],
        )
    )

    exact_gray_table = markdown_table(
        [
            "Condición exacta",
            "R@1",
            "R@5",
            "MRR",
            "nDCG@10",
        ],
        [
            [
                (
                    "Imagen original + "
                    "caption completo"
                ),
                format_metric(
                    openclip_r1
                ),
                format_metric(
                    openclip_r5
                ),
                format_metric(
                    openclip_mrr
                ),
                format_metric(
                    openclip_ndcg
                ),
            ],
            [
                (
                    "Imagen gris + "
                    "caption completo"
                ),
                format_metric(
                    gray_r1
                ),
                format_metric(
                    gray_r5
                ),
                format_metric(
                    gray_mrr
                ),
                format_metric(
                    gray_ndcg
                ),
            ],
            [
                "Diferencia gris − original",
                format_metric(
                    gray_r1
                    - openclip_r1
                ),
                format_metric(
                    gray_r5
                    - openclip_r5
                ),
                format_metric(
                    gray_mrr
                    - openclip_mrr
                ),
                format_metric(
                    gray_ndcg
                    - openclip_ndcg
                ),
            ],
        ],
        [
            "left",
            "right",
            "right",
            "right",
            "right",
        ],
    )

    structural_table = markdown_table(
        [
            "Condición estructural",
            "Hit@1",
            "Hit@5",
            "Recall fracc.@5",
            "MRR",
            "nDCG@10",
        ],
        structural_rows,
        [
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
        ],
    )

    cases_table = markdown_table(
        [
            "Caso",
            "Imagen",
            "Tipo",
            "Rango correcto",
            "Diagnóstico",
            "Interpretación",
        ],
        case_rows,
        [
            "center",
            "center",
            "center",
            "right",
            "left",
            "left",
        ],
    )

    reliability_table = markdown_table(
        [
            "Prueba v1",
            "Casos",
            (
                "Cambio absoluto "
                "medio de rango"
            ),
            "Clasificación observada",
            "Propósito",
        ],
        reliability_summary(
            reliability
        ),
        [
            "left",
            "right",
            "right",
            "left",
            "left",
        ],
    )

    explainability_table = (
        markdown_table(
            [
                "Caso",
                "Imagen",
                "Tipo",
                "Explicación propuesta",
                "Límite",
            ],
            explainability_rows,
            [
                "center",
                "center",
                "center",
                "left",
                "left",
            ],
        )
    )

    responsible_rows = []

    responsible_keys = (
        (
            "posible_sesgo_visual",
            "Sesgo visual",
        ),
        (
            "posible_sesgo_linguistico",
            "Sesgo lingüístico",
        ),
        (
            (
                "posible_sesgo_"
                "cultural_o_dominio"
            ),
            "Sesgo cultural o de dominio",
        ),
        (
            "riesgo_principal_si_se_usa_mal",
            "Riesgo principal",
        ),
        (
            "supervision_humana_necesaria",
            "Supervisión humana",
        ),
        (
            "uso_recomendado",
            "Uso recomendado",
        ),
        (
            (
                "justificacion_del_"
                "uso_recomendado"
            ),
            "Justificación",
        ),
    )

    for key, label in responsible_keys:
        if key in responsible_use:
            responsible_rows.append(
                [
                    label,
                    responsible_use[key],
                ]
            )

    responsible_table = markdown_table(
        [
            "Aspecto",
            "Evaluación",
        ],
        responsible_rows,
        [
            "left",
            "left",
        ],
    )

    limitations = "\n".join(
        f"- {item}"
        for item in contract[
            "required_limitations"
        ]
    )

    report = f"""# Evaluación responsable de recuperación de texto a imagen sobre patrones textiles generados

> **Versión final v2.** La evidencia cuantitativa principal procede del benchmark v2. Las secciones cualitativas heredadas del protocolo v1 se mantienen como evidencia histórica y están identificadas explícitamente; no deben interpretarse como resultados cuantitativos de v2.

## 1. Proyecto personal y alcance

Este trabajo evalúa un prototipo multimodal para relacionar descripciones textuales controladas con imágenes sintéticas de patrones geométricos inspirados en una estética textil. La tarea principal de v2 es **recuperación de texto a imagen**: cada consulta textual debe ordenar una galería de 56 imágenes y situar la imagen correspondiente en una posición alta.

El objetivo no es construir un sistema de identificación cultural ni demostrar reconocimiento de textiles reales. El propósito es medir, en un entorno controlado, hasta qué punto OpenCLIP conserva una señal conjunta de composición, patrón y paleta, y dónde aparecen sus principales errores.

| Elemento | Definición |
| --- | --- |
| Modalidades | Texto e imagen |
| Tarea principal | Recuperación de texto a imagen |
| Modelo | OpenCLIP ViT-B-32 |
| Pesos | `laion2b_s34b_b79k` |
| Régimen | Zero-shot, modelo congelado |
| Galería | 56 imágenes sintéticas |
| Consultas positivas | 280 usos de captions, cinco por imagen |
| Usuario previsto | Estudiante, investigador o evaluador académico |
| Riesgo principal | Confundir similitud visual con identificación real o interpretación cultural |

La parte que puede evaluarse con evidencia es el **alineamiento multimodal dentro del benchmark construido**. La evaluación permite comparar rankings, líneas base, negativos difíciles y ablaciones; no permite atribuir origen, técnica, periodo, autenticidad o significado cultural.

## 2. Evolución del prototipo y adaptación experimental

La primera versión del proyecto adaptó el Cuaderno 14 para una tarea de imagen a texto sobre 40 imágenes y 200 descripciones. Esa etapa permitió comprobar el flujo general, producir cinco casos cualitativos, ejecutar pruebas breves de confiabilidad y documentar riesgos de uso.

La auditoría posterior encontró duplicación exacta y repetición semántica en v1. Por ello se diseñó v2 como un benchmark separado, con combinaciones visuales únicas, protocolo formal y artefactos verificables.

| Aspecto | Protocolo inicial v1 | Benchmark v2 |
| --- | --- | --- |
| Función principal | Adaptación académica inicial | Evaluación cuantitativa controlada |
| Dirección principal | Imagen a texto | Texto a imagen |
| Imágenes | 40 | 56 |
| Consultas | 200 descripciones candidatas | 280 usos de captions positivos |
| Baselines | Aleatorio | Aleatorio y color HSV |
| Controles | Casos y pruebas breves | Negativos difíciles, ablaciones y particiones sintéticas |
| Estado en este informe | Evidencia cualitativa heredada | Evidencia cuantitativa principal |

La adaptación más importante consistió en pasar de una demostración funcional a una evaluación con contratos explícitos. El dataset, los captions, los negativos, los embeddings, las métricas, las tablas y las figuras se validaron mediante scripts independientes e idempotencia.

El cuaderno heredado permanece como evidencia de v1. Los resultados oficiales de v2 proceden de los artefactos de `results/v2/` y no requieren regenerar los embeddings del modelo.

## 3. Construcción del dataset sintético v2

El dataset v2 contiene 56 imágenes sintéticas distribuidas en cuatro grupos controlados:

| Partición sintética | Imágenes |
| --- | ---: |
| ID | 30 |
| OOD por paleta | 12 |
| OOD por patrón | 10 |
| OOD por patrón y paleta | 4 |
| **Total** | **56** |

Cada imagen posee cinco captions positivos, lo que produce 280 usos de consulta en la recuperación exacta global. Los textos describen atributos observables y controlados: patrón, organización visual, orientación y paleta.

Para la evaluación estructural se agruparon descripciones que comparten estructura, aunque difieran en color. Se definieron 40 grupos de captions sin color, cada uno asociado a siete imágenes estructuralmente relevantes.

También se construyó una evaluación de negativos difíciles. Cada unidad contiene un caption positivo y cuatro alternativas contrafactuales controladas. Los negativos alteran u omiten atributos de forma sistemática, evitando que la evaluación se limite a distinguir textos completamente ajenos.

Las imágenes son generadas. No representan piezas documentadas, no constituyen un corpus etnográfico y no pueden usarse para autenticar ni clasificar patrimonio.

## 4. Modelo, líneas base y reproducibilidad

El modelo evaluado es OpenCLIP ViT-B-32 con pesos `laion2b_s34b_b79k`. Se utilizó sin ajuste fino. Las imágenes y textos se transformaron en embeddings de 512 dimensiones, normalizados con norma L2, y se ordenaron mediante similitud coseno.

El entorno canónico utilizó Python 3.11.9, PyTorch CPU, OpenCLIP 3.3.0 y caché externa de Hugging Face. La GPU disponible no fue necesaria para reproducir las evaluaciones a partir de los embeddings congelados.

Se compararon tres familias de métodos:

1. **Baseline aleatorio:** ordenamiento reproducible sin información visual o textual.
2. **Baseline HSV:** comparación basada en distribución de color.
3. **OpenCLIP:** representación conjunta de visión y lenguaje.

Las métricas exactas son R@1, R@5, MRR y nDCG@10. R@1 comprueba si la imagen exacta queda primera; R@5 revisa si aparece en las cinco primeras posiciones; MRR considera la posición recíproca de la primera respuesta correcta; y nDCG@10 resume la calidad del ordenamiento temprano.

En la recuperación estructural existen siete imágenes relevantes por consulta. Por ello se distinguen Hit@K y Recall fraccional@5. Hit@K indica si aparece al menos un relevante; el recall fraccional mide qué proporción de los siete relevantes entra en los cinco primeros resultados. Estas métricas no son intercambiables con la recuperación exacta.

## 5. Resultados cuantitativos v2

La tabla siguiente resume la recuperación exacta en la galería global.

{exact_comparison_table}

![Recuperación exacta en la galería global](results/v2/figuras/f1_recuperacion_exacta_v2.png)

*Figura F1. Comparación de OpenCLIP con las líneas base y con la ablación visual en escala de grises.*

OpenCLIP obtuvo R@1 = **{format_metric(openclip_r1)}**, R@5 = **{format_metric(openclip_r5)}**, MRR = **{format_metric(openclip_mrr)}** y nDCG@10 = **{format_metric(openclip_ndcg)}**. El modelo situó la imagen exacta en primer lugar en {format_percentage(openclip_r1)} de las consultas y dentro del top 5 en {format_percentage(openclip_r5)}.

El resultado supera tanto al baseline aleatorio como al histograma HSV. En R@1, OpenCLIP alcanzó {format_metric(openclip_r1)}, frente a {format_metric(hsv_r1)} del baseline de color y {format_metric(random_r1)} del aleatorio. Esto demuestra que la recuperación no depende únicamente de azar o paleta.

Sin embargo, el desempeño sigue siendo parcial. Aproximadamente tres de cada cuatro consultas no colocan la imagen exacta en la primera posición. R@5 es alto, pero no debe interpretarse aisladamente: una imagen puede aparecer dentro de cinco resultados sin quedar correctamente priorizada.

MRR y nDCG@10 complementan esta lectura. Ambos muestran que OpenCLIP ordena la galería mejor que las líneas base, pero dejan margen para mejorar la discriminación fina entre imágenes que comparten estructura o color.

## 6. Negativos difíciles, ablaciones y generalización

### 6.1. Negativos difíciles

La evaluación E2 restringe cada decisión a un caption positivo y cuatro negativos contrafactuales.

{hard_negative_table}

![Desempeño ante negativos difíciles](results/v2/figuras/f2_negativos_dificiles_v2.png)

*Figura F2. Desempeño de OpenCLIP ante cuatro negativos controlados por unidad.*

La exactitud de {format_metric(hard_accuracy)} indica que el positivo queda primero en algo más de la mitad de los casos. Aun así, la tasa pareada de {format_metric(pairwise_win_rate)} muestra que el positivo suele superar individualmente a cada negativo. La diferencia entre ambas métricas es importante: ganar varias comparaciones pareadas no garantiza ocupar la primera posición frente a las cuatro alternativas simultáneamente.

Estos resultados describen discriminación local dentro de un conjunto diseñado. No equivalen a precisión global ni a una prueba de comprensión semántica.

### 6.2. Ablación visual en recuperación exacta

{exact_gray_table}

![Efecto de la escala de grises en la recuperación exacta](results/v2/figuras/f4_efecto_grises_exacto_v2.png)

*Figura F4. Diferencia entre imágenes en escala de grises e imágenes originales para la tarea exacta.*

Eliminar el color visual reduce R@1 de {format_metric(openclip_r1)} a {format_metric(gray_r1)} y R@5 de {format_metric(openclip_r5)} a {format_metric(gray_r5)}. También disminuyen MRR y nDCG@10. Por tanto, la información cromática contribuye de manera importante a identificar la imagen exacta.

Esta observación no implica que el modelo dependa exclusivamente del color. OpenCLIP también supera al baseline HSV, lo que indica que utiliza información adicional. La lectura correcta es que color y estructura contribuyen de forma conjunta.

### 6.3. Recuperación estructural

{structural_table}

![Ablaciones cromáticas en la recuperación estructural](results/v2/figuras/f3_ablaciones_estructurales_v2.png)

*Figura F3. Comparación de cuatro condiciones en la tarea con múltiples imágenes estructuralmente relevantes.*

![Compromiso entre Hit@1 y Hit@5 estructurales](results/v2/figuras/f5_compromiso_hit1_hit5_v2.png)

*Figura F5. Relación entre éxito inmediato y cobertura temprana para las cuatro condiciones.*

En la tarea estructural, la condición **imagen gris + caption sin color** obtiene el mayor Hit@1, MRR, nDCG@10 y recall fraccional@5. No obstante, la condición original con caption completo conserva el mayor Hit@5.

Por ello, no es válido afirmar que retirar el color sea universalmente mejor. Las métricas responden preguntas diferentes. La supresión cromática puede favorecer la selección inmediata de una imagen estructuralmente compatible, mientras que la condición completa recupera al menos un relevante en el top 5 con mayor frecuencia.

Las particiones ID y OOD del proyecto son controles sintéticos construidos mediante combinaciones de patrones y paletas. Permiten estudiar cambios dentro de este generador, pero no demuestran generalización a textiles reales, colecciones museales ni categorías culturales.

## 7. Evaluación cualitativa heredada del protocolo v1

> **Alcance de esta sección:** los cinco casos siguientes provienen del experimento inicial v1 de recuperación de imagen a texto sobre 40 imágenes. Se conservan porque responden al requisito cualitativo de la actividad, pero no constituyen evidencia directa de E1 a E4 ni deben mezclarse con las métricas v2.

{cases_table}

![Cinco casos evaluados en el protocolo inicial](figures/ejemplos_evaluados.png)

*Figura cualitativa v1. Dos aciertos, dos errores y un caso ambiguo del experimento inicial.*

Los dos aciertos muestran que el modelo puede recuperar captions compatibles cuando composición, patrón y paleta tienen una correspondencia clara. Los dos errores revelan confusión entre imágenes visualmente próximas y descripciones de alto solapamiento. El caso ambiguo recuerda que una única etiqueta esperada puede ser insuficiente cuando la lectura visual admite más de una descripción razonable.

Esta evidencia es útil para discutir comportamiento y límites, no para recalcular el desempeño de v2.

## 8. Confiabilidad y explicabilidad

Las pruebas heredadas de v1 estudiaron dos perturbaciones: reformulación textual y degradación visual. Se realizaron diez observaciones en total.

{reliability_table}

La prueba de sensibilidad al texto compara una descripción original con una reformulación de significado similar. La degradación visual examina si cambios de calidad alteran el ranking. Estas pruebas son breves y no constituyen una certificación de robustez.

La clasificación prudente sigue siendo **confiable únicamente en condiciones controladas**. El sistema produce una señal útil, pero sus posiciones pueden variar por redacción, calidad visual y similitud entre candidatos.

La explicabilidad se documentó sobre un acierto y un error:

{explainability_table}

![Explicabilidad de un acierto y un error](figures/explicabilidad_casos.png)

*Figura cualitativa v1. Explicación basada en atributos observables y resultados del ranking.*

Las explicaciones relacionan composición, patrón, paleta, caption recuperado y posición. No identifican qué regiones internas utilizó el modelo ni demuestran comprensión del significado de los motivos.

## 9. Sesgo, uso responsable y amenazas a la validez

{responsible_table}

Los principales límites del trabajo son:

{limitations}

Además, las descripciones usan vocabulario controlado. El modelo puede responder mejor a esas formulaciones que a lenguaje libre. Los patrones generados simplifican la variabilidad de objetos textiles reales y pueden inducir asociaciones visuales excesivamente homogéneas.

El uso recomendado es académico y exploratorio, con supervisión humana. El sistema no debe utilizarse para identificación cultural, atribución histórica, autenticación, clasificación patrimonial o decisiones sobre objetos reales.

## 10. Conclusiones

El benchmark v2 permitió pasar de una demostración inicial a una evaluación reproducible y auditable de recuperación de texto a imagen. OpenCLIP superó al baseline aleatorio y al baseline HSV en la galería global. Su resultado principal fue R@1 = **{format_metric(openclip_r1)}**, R@5 = **{format_metric(openclip_r5)}**, MRR = **{format_metric(openclip_mrr)}** y nDCG@10 = **{format_metric(openclip_ndcg)}**.

La evaluación de negativos difíciles mostró discriminación parcial: la exactitud fue **{format_metric(hard_accuracy)}**, mientras que la tasa de victorias pareadas alcanzó **{format_metric(pairwise_win_rate)}**. Esto indica que el modelo suele preferir el caption positivo frente a un negativo individual, pero no siempre logra situarlo primero frente a cuatro alternativas simultáneas.

Las ablaciones muestran dos comportamientos complementarios. La escala de grises perjudica claramente la recuperación de la imagen exacta. En cambio, retirar el color de imagen y texto puede favorecer algunas métricas estructurales. No existe una condición universalmente superior: el resultado depende de si la pregunta exige identidad exacta o semejanza estructural.

Los cinco casos, las pruebas de confiabilidad y las explicaciones heredadas de v1 conservan valor como evidencia cualitativa. Su función es ilustrar aciertos, errores y ambigüedad; no reemplazan ni amplían artificialmente los resultados cuantitativos de v2.

En conjunto, el sistema muestra alineamiento multimodal útil dentro de un benchmark sintético controlado. La evidencia no permite afirmar reconocimiento cultural, autenticación de textiles, comprensión del significado de los motivos ni generalización a colecciones reales.

### Trazabilidad principal

- Protocolo: `docs/especificacion_experimental_v2.md`
- Dataset y captions: `docs/auditoria_visual_patrones_v2.md` y `docs/diseno_captions_positivos_v2.md`
- Negativos difíciles: `docs/diseno_negativos_dificiles_v2.md`
- Entorno: `docs/entorno_reproducible_v2.md`
- Métricas maestras: `results/v2/tablas_maestras/metricas_maestras_v2.csv`
- Comparaciones maestras: `results/v2/tablas_maestras/comparaciones_maestras_v2.csv`
- Figuras auditadas: `results/v2/figuras/`
- Evidencia cualitativa v1: `results/casos_analizados.csv`, `results/pruebas_confiabilidad.csv` y `results/explicabilidad.csv`
"""

    report = (
        report
        .replace("\r\n", "\n")
        .rstrip()
        + "\n"
    )

    forbidden_legacy_metrics = (
        "0.325",
        "0.600",
        "0.481256",
        "0.026175",
        "0.119300",
    )

    for marker in (
        forbidden_legacy_metrics
    ):
        assert marker not in report, (
            "El informe conserva una "
            f"métrica cuantitativa v1: {marker}"
        )

    for figure in contract[
        "required_v2_figures"
    ]:
        assert figure["path"] in report

    section_numbers = [
        int(match.group(1))
        for match in re.finditer(
            r"^## (\d+)\. ",
            report,
            flags=re.MULTILINE,
        )
    ]

    assert section_numbers == list(
        range(1, 11)
    )

    for claim in contract[
        "forbidden_unqualified_claims"
    ]:
        assert (
            claim.casefold()
            not in report.casefold()
        ), (
            "El informe contiene una "
            "afirmación no permitida: "
            f"{claim}"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(report)

    output_raw = (
        OUTPUT_PATH.read_bytes()
    )

    assert not output_raw.startswith(
        b"\xef\xbb\xbf"
    )

    assert b"\r\n" not in output_raw

    print("=" * 92)
    print(
        "GENERACIÓN DEL INFORME FINAL "
        "V2 COMPLETADA"
    )
    print("=" * 92)

    print(
        "Archivo:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
    )

    print(
        "Secciones numeradas:",
        len(section_numbers),
    )

    print(
        "Figuras v2 embebidas:",
        sum(
            figure["path"] in report
            for figure in contract[
                "required_v2_figures"
            ]
        ),
    )

    print(
        "Figuras cualitativas v1:",
        sum(
            path in report
            for path in (
                (
                    "figures/"
                    "ejemplos_evaluados.png"
                ),
                (
                    "figures/"
                    "explicabilidad_casos.png"
                ),
            )
        ),
    )

    print(
        "Casos cualitativos:",
        len(cases),
    )

    print(
        "Pruebas de confiabilidad:",
        len(reliability),
    )

    print(
        "Casos de explicabilidad:",
        len(explainability),
    )

    print(
        "Caracteres:",
        len(report),
    )

    print(
        "Palabras aproximadas:",
        len(
            re.findall(
                r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]+\b",
                report,
            )
        ),
    )

    print(
        "UTF-8 sin BOM:",
        True,
    )

    print(
        "Saltos LF:",
        True,
    )

    print(
        "Informe generado:",
        True,
    )


if __name__ == "__main__":
    main()
