"""Evalúa las cuatro ablaciones cromáticas de E4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from metricas_retrieval_v2 import evaluate_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "ablaciones_v2.json"
)

FULL_CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
)

COLORLESS_CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_sin_color_v2.csv"
)

EMBEDDINGS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "embeddings"
)

IMAGE_INDEX_PATH = (
    EMBEDDINGS_DIRECTORY
    / "index_images_v2.csv"
)

TEXT_USAGE_PATH = (
    EMBEDDINGS_DIRECTORY
    / "usos_textos_v2.csv"
)

ORIGINAL_IMAGE_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIRECTORY
    / "embeddings_imagen_original_v2.npy"
)

GRAYSCALE_IMAGE_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIRECTORY
    / "embeddings_imagen_grayscale_v2.npy"
)

TEXT_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIRECTORY
    / "embeddings_textos_unicos_v2.npy"
)

E1_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "global_openclip_v2"
    / "resultados_consulta_global_openclip_v2.csv"
)

E1_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "global_openclip_v2"
    / "resumen_global_openclip_v2.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "ablaciones_v2"
)

TEMPORARY_DIRECTORY = (
    OUTPUT_DIRECTORY.with_name(
        OUTPUT_DIRECTORY.name + ".tmp"
    )
)

RAW_RESULTS_FILENAME = (
    "resultados_consulta_estructural_"
    "ablaciones_v2.csv"
)

STRUCTURAL_RANKING_FILENAME = (
    "ranking_estructural_ablaciones_v2.csv"
)

GROUP_RESULTS_FILENAME = (
    "resultados_grupo_estructural_"
    "ablaciones_v2.csv"
)

STRUCTURAL_AGGREGATES_FILENAME = (
    "agregados_estructurales_"
    "ablaciones_v2.csv"
)

PAIRED_RESULTS_FILENAME = (
    "comparaciones_pareadas_"
    "estructurales_v2.csv"
)

PAIRED_AGGREGATES_FILENAME = (
    "agregados_comparaciones_"
    "estructurales_v2.csv"
)

EXACT_RESULTS_FILENAME = (
    "resultados_exactos_ablaciones_v2.csv"
)

EXACT_AGGREGATES_FILENAME = (
    "agregados_exactos_ablaciones_v2.csv"
)

SUMMARY_FILENAME = (
    "resumen_ablaciones_v2.json"
)

CONDITION_ORDER = (
    "original_image_full_caption",
    "grayscale_image_full_caption",
    "original_image_caption_without_color",
    "grayscale_image_caption_without_color",
)

STRUCTURAL_METRIC_KEYS = (
    "structural_hit_at_1",
    "structural_hit_at_5",
    "structural_fractional_recall_at_5",
    "structural_mrr",
    "structural_ndcg_at_10",
    "best_relevant_margin",
)

EXACT_METRIC_KEYS = (
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
    "positive_margin",
)

RAW_FIELDS = (
    "raw_row_index",
    "condition",
    "image_variant",
    "text_variant",
    "condition_query_index",
    "group_index",
    "group_query_index",
    "query_id",
    "source_id",
    "text_row_index",
    "colorless_caption_id",
    "structure_id",
    "template_id",
    "is_canonical",
    "regime",
    "pattern_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "relevant_count",
    "first_relevant_rank",
    "top1_image_id",
    "top1_score",
    "best_relevant_score",
    "best_nonrelevant_score",
    "best_relevant_margin",
    "structural_hit_at_1",
    "structural_hit_at_5",
    "structural_fractional_recall_at_5",
    "structural_mrr",
    "structural_ndcg_at_10",
)

RANKING_FIELDS = (
    "condition",
    "raw_row_index",
    "condition_query_index",
    "group_index",
    "query_id",
    "rank",
    "image_row_index",
    "image_id",
    "score",
    "is_relevant",
    "image_split",
    "image_pattern_id",
    "image_palette_id",
)

GROUP_FIELDS = (
    "condition",
    "image_variant",
    "text_variant",
    "group_index",
    "colorless_caption_id",
    "structure_id",
    "template_id",
    "is_canonical",
    "regime",
    "pattern_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "raw_query_count",
    "structural_hit_at_1",
    "structural_hit_at_5",
    "structural_fractional_recall_at_5",
    "structural_mrr",
    "structural_ndcg_at_10",
    "best_relevant_margin",
)

STRUCTURAL_AGGREGATE_FIELDS = (
    "condition",
    "group_dimension",
    "group_value",
    "group_count",
    "structural_hit_at_1",
    "structural_hit_at_5",
    "structural_fractional_recall_at_5",
    "structural_mrr",
    "structural_ndcg_at_10",
    "best_relevant_margin",
)

PAIRED_FIELDS = (
    "comparison_id",
    "minuend",
    "subtrahend",
    "group_index",
    "colorless_caption_id",
    "structure_id",
    "template_id",
    "is_canonical",
    "regime",
    "delta_structural_hit_at_1",
    "delta_structural_hit_at_5",
    "delta_structural_fractional_recall_at_5",
    "delta_structural_mrr",
    "delta_structural_ndcg_at_10",
    "delta_best_relevant_margin",
)

PAIRED_AGGREGATE_FIELDS = (
    "comparison_id",
    "minuend",
    "subtrahend",
    "group_dimension",
    "group_value",
    "group_count",
    "delta_structural_hit_at_1",
    "delta_structural_hit_at_5",
    "delta_structural_fractional_recall_at_5",
    "delta_structural_mrr",
    "delta_structural_ndcg_at_10",
    "delta_best_relevant_margin",
)

EXACT_FIELDS = (
    "condition",
    "source",
    "query_index",
    "caption_id",
    "image_id",
    "split",
    "template_id",
    "is_canonical",
    "text_row_index",
    "relevant_image_row_index",
    "relevant_rank",
    "top1_image_id",
    "top1_score",
    "relevant_score",
    "positive_margin",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
)

EXACT_AGGREGATE_FIELDS = (
    "condition",
    "group_dimension",
    "group_value",
    "query_count",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
    "positive_margin",
)


def load_json(path: Path) -> dict[str, Any]:
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
) -> tuple[list[str], list[dict[str, str]]]:
    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene CRLF."
    )

    raw.decode("utf-8")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return list(reader.fieldnames or []), rows


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def format_float(value: float) -> str:
    return format(
        float(value),
        ".12f",
    )


def assert_close(
    actual: float,
    expected: float,
    context: str,
    tolerance: float = 1e-7,
) -> None:
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ), (
        f"{context}: actual={actual}, "
        f"esperado={expected}."
    )


def split_ids(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def mean_metric(
    rows: Iterable[dict[str, Any]],
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
    ]

    assert values
    assert np.isfinite(values).all()

    return float(
        np.mean(values)
    )


def validate_norms(
    matrix: np.ndarray,
    name: str,
) -> None:
    assert matrix.ndim == 2
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    maximum_error = float(
        np.max(
            np.abs(norms - 1.0)
        )
    )

    assert maximum_error < 1e-5, (
        f"{name}: error de norma "
        f"{maximum_error}."
    )


def determine_regime(
    relevant_image_ids: set[str],
    image_by_id: dict[str, dict[str, str]],
) -> str:
    split_composition = Counter(
        image_by_id[image_id]["split"]
        for image_id
        in relevant_image_ids
    )

    if split_composition == {
        "id": 5,
        "ood_palette": 2,
    }:
        return "base_pattern"

    if split_composition == {
        "ood_pattern": 5,
        "ood_both": 2,
    }:
        return "heldout_pattern"

    raise AssertionError(
        "Composición de splits inesperada: "
        f"{dict(split_composition)}"
    )


def structural_metrics(
    scores: np.ndarray,
    relevant_indices: set[int],
    image_ids: list[str],
) -> dict[str, Any]:
    result = evaluate_query(
        scores=scores,
        relevant_indices=relevant_indices,
        candidate_keys=image_ids,
    )

    ranking = [
        int(index)
        for index in result[
            "ranking_indices"
        ]
    ]

    relevant_in_top_5 = sum(
        index in relevant_indices
        for index in ranking[:5]
    )

    fractional_recall_at_5 = (
        relevant_in_top_5
        / len(relevant_indices)
    )

    relevant_scores = np.asarray(
        [
            scores[index]
            for index in relevant_indices
        ],
        dtype=np.float64,
    )

    nonrelevant_scores = np.asarray(
        [
            scores[index]
            for index in range(
                scores.size
            )
            if index not in relevant_indices
        ],
        dtype=np.float64,
    )

    assert relevant_scores.size == 7
    assert nonrelevant_scores.size == 49

    best_relevant_score = float(
        np.max(relevant_scores)
    )

    best_nonrelevant_score = float(
        np.max(nonrelevant_scores)
    )

    return {
        "ranking_indices": ranking,
        "first_relevant_rank": int(
            result["first_relevant_rank"]
        ),
        "structural_hit_at_1": float(
            result["recall_at_1"]
        ),
        "structural_hit_at_5": float(
            result["recall_at_5"]
        ),
        "structural_fractional_recall_at_5": (
            float(
                fractional_recall_at_5
            )
        ),
        "structural_mrr": float(
            result["mrr"]
        ),
        "structural_ndcg_at_10": float(
            result["ndcg_at_10"]
        ),
        "best_relevant_score": (
            best_relevant_score
        ),
        "best_nonrelevant_score": (
            best_nonrelevant_score
        ),
        "best_relevant_margin": (
            best_relevant_score
            - best_nonrelevant_score
        ),
    }


def aggregate_structural_group(
    condition: str,
    image_variant: str,
    text_variant: str,
    group_index: int,
    colorless: dict[str, str],
    regime: str,
    raw_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_count = (
        7
        if text_variant == "full_caption"
        else 1
    )

    assert len(raw_rows) == expected_count

    row = {
        "condition": condition,
        "image_variant": image_variant,
        "text_variant": text_variant,
        "group_index": group_index,
        "colorless_caption_id": (
            colorless[
                "colorless_caption_id"
            ]
        ),
        "structure_id": (
            colorless["structure_id"]
        ),
        "template_id": (
            colorless["template_id"]
        ),
        "is_canonical": (
            colorless[
                "is_canonical"
            ].strip().lower()
        ),
        "regime": regime,
        "pattern_id": (
            colorless["pattern_id"]
        ),
        "motif": colorless["motif"],
        "orientation": (
            colorless["orientation"]
        ),
        "composition": (
            colorless["composition"]
        ),
        "symmetry": (
            colorless["symmetry"]
        ),
        "raw_query_count": len(raw_rows),
    }

    for metric in STRUCTURAL_METRIC_KEYS:
        row[metric] = format_float(
            mean_metric(
                raw_rows,
                metric,
            )
        )

    return row


def structural_aggregate_row(
    condition: str,
    dimension: str,
    value: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "condition": condition,
        "group_dimension": dimension,
        "group_value": value,
        "group_count": len(rows),
    }

    for metric in STRUCTURAL_METRIC_KEYS:
        result[metric] = format_float(
            mean_metric(rows, metric)
        )

    return result


def exact_aggregate_row(
    condition: str,
    dimension: str,
    value: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "condition": condition,
        "group_dimension": dimension,
        "group_value": value,
        "query_count": len(rows),
    }

    for metric in EXACT_METRIC_KEYS:
        result[metric] = format_float(
            mean_metric(rows, metric)
        )

    return result


def publish_directory(
    temporary_directory: Path,
    output_directory: Path,
) -> None:
    backup_directory = (
        output_directory.with_name(
            output_directory.name
            + ".previous"
        )
    )

    if backup_directory.exists():
        shutil.rmtree(
            backup_directory
        )

    if output_directory.exists():
        output_directory.replace(
            backup_directory
        )

    try:
        temporary_directory.replace(
            output_directory
        )
    except Exception:
        if (
            backup_directory.exists()
            and not output_directory.exists()
        ):
            backup_directory.replace(
                output_directory
            )

        raise

    if backup_directory.exists():
        shutil.rmtree(
            backup_directory
        )


def main() -> None:
    contract = load_json(
        CONFIG_PATH
    )

    _, full_rows = load_csv(
        FULL_CAPTIONS_PATH
    )

    _, colorless_rows = load_csv(
        COLORLESS_CAPTIONS_PATH
    )

    _, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    _, usage_rows = load_csv(
        TEXT_USAGE_PATH
    )

    _, e1_rows = load_csv(
        E1_RESULTS_PATH
    )

    e1_summary = load_json(
        E1_SUMMARY_PATH
    )

    assert contract["experiment_id"] == "E4"

    assert tuple(
        record["condition_id"]
        for record in contract[
            "conditions"
        ]
    ) == CONDITION_ORDER

    assert len(full_rows) == 280
    assert len(colorless_rows) == 40
    assert len(image_rows) == 56
    assert len(usage_rows) == 600
    assert len(e1_rows) == 280

    ordered_images = sorted(
        image_rows,
        key=lambda row: int(
            row["image_row_index"]
        ),
    )

    assert [
        int(row["image_row_index"])
        for row in ordered_images
    ] == list(range(56))

    image_ids = [
        row["image_id"]
        for row in ordered_images
    ]

    image_row_by_id = {
        image_id: index
        for index, image_id
        in enumerate(image_ids)
    }

    image_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    assert len(image_row_by_id) == 56
    assert len(image_by_id) == 56

    usage_map = {
        (
            row["source_id"],
            row["record_id"],
        ): int(
            row["text_row_index"]
        )
        for row in usage_rows
    }

    assert len(usage_map) == 600

    original_images = np.load(
        ORIGINAL_IMAGE_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    grayscale_images = np.load(
        GRAYSCALE_IMAGE_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    text_embeddings = np.load(
        TEXT_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    assert original_images.shape == (
        56,
        512,
    )

    assert grayscale_images.shape == (
        56,
        512,
    )

    assert text_embeddings.shape == (
        494,
        512,
    )

    validate_norms(
        original_images,
        "original",
    )

    validate_norms(
        grayscale_images,
        "grayscale",
    )

    validate_norms(
        text_embeddings,
        "textos",
    )

    image_matrix_by_variant = {
        "original": original_images,
        "grayscale": grayscale_images,
    }

    colorless_by_id = {
        row[
            "colorless_caption_id"
        ]: row
        for row in colorless_rows
    }

    assert len(colorless_by_id) == 40

    group_ids = [
        row["colorless_caption_id"]
        for row in colorless_rows
    ]

    group_index_by_id = {
        group_id: group_index
        for group_index, group_id
        in enumerate(group_ids)
    }

    full_by_group: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in full_rows:
        full_by_group[
            row["colorless_caption_id"]
        ].append(row)

    assert len(full_by_group) == 40

    assert Counter(
        len(rows)
        for rows in full_by_group.values()
    ) == {
        7: 40
    }

    group_query_index_by_caption = {}

    for group_id in group_ids:
        for group_query_index, row in enumerate(
            full_by_group[group_id]
        ):
            group_query_index_by_caption[
                row["caption_id"]
            ] = group_query_index

    e1_by_caption = {
        row["caption_id"]: row
        for row in e1_rows
    }

    assert len(e1_by_caption) == 280

    assert e1_summary[
        "evaluation_valid"
    ] is True

    condition_specs = {
        "original_image_full_caption": {
            "image_variant": "original",
            "text_variant": "full_caption",
            "source_id": "positivos",
            "queries": full_rows,
        },
        "grayscale_image_full_caption": {
            "image_variant": "grayscale",
            "text_variant": "full_caption",
            "source_id": "positivos",
            "queries": full_rows,
        },
        "original_image_caption_without_color": {
            "image_variant": "original",
            "text_variant": (
                "caption_without_color"
            ),
            "source_id": "sin_color",
            "queries": colorless_rows,
        },
        "grayscale_image_caption_without_color": {
            "image_variant": "grayscale",
            "text_variant": (
                "caption_without_color"
            ),
            "source_id": "sin_color",
            "queries": colorless_rows,
        },
    }

    raw_result_rows: list[
        dict[str, Any]
    ] = []

    ranking_rows: list[
        dict[str, Any]
    ] = []

    raw_rows_by_condition_group: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    raw_row_index = 0

    print("=" * 88)
    print("EVALUACIÓN DE ABLACIONES CROMÁTICAS E4")
    print("=" * 88)
    print("Grupos estructurales:", len(group_ids))
    print("Galería:", len(image_ids))
    print("Condiciones:", len(CONDITION_ORDER))

    for condition in CONDITION_ORDER:
        specification = condition_specs[
            condition
        ]

        image_variant = specification[
            "image_variant"
        ]

        text_variant = specification[
            "text_variant"
        ]

        source_id = specification[
            "source_id"
        ]

        queries = specification[
            "queries"
        ]

        image_matrix = (
            image_matrix_by_variant[
                image_variant
            ]
        )

        for condition_query_index, query in enumerate(
            queries
        ):
            if text_variant == "full_caption":
                query_id = query["caption_id"]

                group_id = query[
                    "colorless_caption_id"
                ]

                group_query_index = (
                    group_query_index_by_caption[
                        query_id
                    ]
                )

            else:
                query_id = query[
                    "colorless_caption_id"
                ]

                group_id = query_id
                group_query_index = 0

            colorless = colorless_by_id[
                group_id
            ]

            group_index = (
                group_index_by_id[
                    group_id
                ]
            )

            usage_key = (
                source_id,
                query_id,
            )

            assert usage_key in usage_map

            text_row_index = usage_map[
                usage_key
            ]

            relevant_image_ids = set(
                split_ids(
                    colorless[
                        "relevant_image_ids"
                    ]
                )
            )

            assert len(
                relevant_image_ids
            ) == 7

            relevant_indices = {
                image_row_by_id[
                    image_id
                ]
                for image_id
                in relevant_image_ids
            }

            regime = determine_regime(
                relevant_image_ids,
                image_by_id,
            )

            scores = np.asarray(
                text_embeddings[
                    text_row_index
                ]
                @ image_matrix.T,
                dtype=np.float64,
            )

            assert scores.shape == (56,)
            assert np.isfinite(scores).all()

            metrics = structural_metrics(
                scores,
                relevant_indices,
                image_ids,
            )

            ranking = metrics[
                "ranking_indices"
            ]

            top1_index = int(
                ranking[0]
            )

            raw_row = {
                "raw_row_index": (
                    raw_row_index
                ),
                "condition": condition,
                "image_variant": (
                    image_variant
                ),
                "text_variant": (
                    text_variant
                ),
                "condition_query_index": (
                    condition_query_index
                ),
                "group_index": group_index,
                "group_query_index": (
                    group_query_index
                ),
                "query_id": query_id,
                "source_id": source_id,
                "text_row_index": (
                    text_row_index
                ),
                "colorless_caption_id": (
                    group_id
                ),
                "structure_id": (
                    colorless["structure_id"]
                ),
                "template_id": (
                    colorless["template_id"]
                ),
                "is_canonical": (
                    colorless[
                        "is_canonical"
                    ].strip().lower()
                ),
                "regime": regime,
                "pattern_id": (
                    colorless["pattern_id"]
                ),
                "motif": (
                    colorless["motif"]
                ),
                "orientation": (
                    colorless["orientation"]
                ),
                "composition": (
                    colorless["composition"]
                ),
                "symmetry": (
                    colorless["symmetry"]
                ),
                "relevant_count": 7,
                "first_relevant_rank": (
                    metrics[
                        "first_relevant_rank"
                    ]
                ),
                "top1_image_id": (
                    image_ids[top1_index]
                ),
                "top1_score": format_float(
                    scores[top1_index]
                ),
                "best_relevant_score": (
                    format_float(
                        metrics[
                            "best_relevant_score"
                        ]
                    )
                ),
                "best_nonrelevant_score": (
                    format_float(
                        metrics[
                            "best_nonrelevant_score"
                        ]
                    )
                ),
                "best_relevant_margin": (
                    format_float(
                        metrics[
                            "best_relevant_margin"
                        ]
                    )
                ),
                "structural_hit_at_1": (
                    format_float(
                        metrics[
                            "structural_hit_at_1"
                        ]
                    )
                ),
                "structural_hit_at_5": (
                    format_float(
                        metrics[
                            "structural_hit_at_5"
                        ]
                    )
                ),
                "structural_fractional_recall_at_5": (
                    format_float(
                        metrics[
                            "structural_fractional_recall_at_5"
                        ]
                    )
                ),
                "structural_mrr": (
                    format_float(
                        metrics[
                            "structural_mrr"
                        ]
                    )
                ),
                "structural_ndcg_at_10": (
                    format_float(
                        metrics[
                            "structural_ndcg_at_10"
                        ]
                    )
                ),
            }

            raw_result_rows.append(
                raw_row
            )

            raw_rows_by_condition_group[
                (
                    condition,
                    group_id,
                )
            ].append(raw_row)

            for rank, image_row_index in enumerate(
                ranking,
                start=1,
            ):
                image_row_index = int(
                    image_row_index
                )

                image_record = (
                    ordered_images[
                        image_row_index
                    ]
                )

                ranking_rows.append(
                    {
                        "condition": (
                            condition
                        ),
                        "raw_row_index": (
                            raw_row_index
                        ),
                        "condition_query_index": (
                            condition_query_index
                        ),
                        "group_index": (
                            group_index
                        ),
                        "query_id": query_id,
                        "rank": rank,
                        "image_row_index": (
                            image_row_index
                        ),
                        "image_id": (
                            image_record[
                                "image_id"
                            ]
                        ),
                        "score": format_float(
                            scores[
                                image_row_index
                            ]
                        ),
                        "is_relevant": str(
                            image_row_index
                            in relevant_indices
                        ).lower(),
                        "image_split": (
                            image_record["split"]
                        ),
                        "image_pattern_id": (
                            image_record[
                                "pattern_id"
                            ]
                        ),
                        "image_palette_id": (
                            image_record[
                                "palette_id"
                            ]
                        ),
                    }
                )

            raw_row_index += 1

    assert len(raw_result_rows) == 640
    assert len(ranking_rows) == 35840
    assert raw_row_index == 640

    group_result_rows: list[
        dict[str, Any]
    ] = []

    for condition in CONDITION_ORDER:
        specification = condition_specs[
            condition
        ]

        for group_index, group_id in enumerate(
            group_ids
        ):
            colorless = colorless_by_id[
                group_id
            ]

            relevant_image_ids = set(
                split_ids(
                    colorless[
                        "relevant_image_ids"
                    ]
                )
            )

            regime = determine_regime(
                relevant_image_ids,
                image_by_id,
            )

            rows = (
                raw_rows_by_condition_group[
                    (
                        condition,
                        group_id,
                    )
                ]
            )

            group_result_rows.append(
                aggregate_structural_group(
                    condition=condition,
                    image_variant=(
                        specification[
                            "image_variant"
                        ]
                    ),
                    text_variant=(
                        specification[
                            "text_variant"
                        ]
                    ),
                    group_index=group_index,
                    colorless=colorless,
                    regime=regime,
                    raw_rows=rows,
                )
            )

    assert len(group_result_rows) == 160

    groups_by_condition: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in group_result_rows:
        groups_by_condition[
            row["condition"]
        ].append(row)

    structural_aggregate_rows = []

    for condition in CONDITION_ORDER:
        condition_rows = (
            groups_by_condition[
                condition
            ]
        )

        assert len(condition_rows) == 40

        structural_aggregate_rows.append(
            structural_aggregate_row(
                condition,
                "overall",
                "all",
                condition_rows,
            )
        )

        for dimension in (
            "regime",
            "template_id",
            "is_canonical",
        ):
            grouped: dict[
                str,
                list[dict[str, Any]],
            ] = defaultdict(list)

            for row in condition_rows:
                grouped[
                    str(row[dimension])
                ].append(row)

            for value in sorted(grouped):
                structural_aggregate_rows.append(
                    structural_aggregate_row(
                        condition,
                        dimension,
                        value,
                        grouped[value],
                    )
                )

    assert len(
        structural_aggregate_rows
    ) == 40

    group_row_map = {
        (
            row["condition"],
            row["colorless_caption_id"],
        ): row
        for row in group_result_rows
    }

    assert len(group_row_map) == 160

    paired_rows = []

    for comparison in contract[
        "paired_comparisons"
    ]:
        comparison_id = comparison[
            "comparison_id"
        ]

        minuend = comparison[
            "minuend"
        ]

        subtrahend = comparison[
            "subtrahend"
        ]

        for group_id in group_ids:
            minuend_row = group_row_map[
                (
                    minuend,
                    group_id,
                )
            ]

            subtrahend_row = group_row_map[
                (
                    subtrahend,
                    group_id,
                )
            ]

            paired_row = {
                "comparison_id": (
                    comparison_id
                ),
                "minuend": minuend,
                "subtrahend": (
                    subtrahend
                ),
                "group_index": (
                    minuend_row[
                        "group_index"
                    ]
                ),
                "colorless_caption_id": (
                    group_id
                ),
                "structure_id": (
                    minuend_row[
                        "structure_id"
                    ]
                ),
                "template_id": (
                    minuend_row[
                        "template_id"
                    ]
                ),
                "is_canonical": (
                    minuend_row[
                        "is_canonical"
                    ]
                ),
                "regime": (
                    minuend_row["regime"]
                ),
            }

            for metric in (
                STRUCTURAL_METRIC_KEYS
            ):
                paired_row[
                    f"delta_{metric}"
                ] = format_float(
                    float(
                        minuend_row[metric]
                    )
                    - float(
                        subtrahend_row[
                            metric
                        ]
                    )
                )

            paired_rows.append(
                paired_row
            )

    assert len(paired_rows) == 160

    paired_aggregate_rows = []

    for comparison in contract[
        "paired_comparisons"
    ]:
        comparison_id = comparison[
            "comparison_id"
        ]

        comparison_rows = [
            row
            for row in paired_rows
            if row["comparison_id"]
            == comparison_id
        ]

        assert len(comparison_rows) == 40

        for dimension, value, rows in (
            (
                "overall",
                "all",
                comparison_rows,
            ),
        ):
            aggregate_row = {
                "comparison_id": (
                    comparison_id
                ),
                "minuend": (
                    comparison["minuend"]
                ),
                "subtrahend": (
                    comparison[
                        "subtrahend"
                    ]
                ),
                "group_dimension": (
                    dimension
                ),
                "group_value": value,
                "group_count": len(rows),
            }

            for metric in (
                STRUCTURAL_METRIC_KEYS
            ):
                aggregate_row[
                    f"delta_{metric}"
                ] = format_float(
                    mean_metric(
                        rows,
                        f"delta_{metric}",
                    )
                )

            paired_aggregate_rows.append(
                aggregate_row
            )

        rows_by_regime: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for row in comparison_rows:
            rows_by_regime[
                row["regime"]
            ].append(row)

        for regime in sorted(
            rows_by_regime
        ):
            rows = rows_by_regime[
                regime
            ]

            aggregate_row = {
                "comparison_id": (
                    comparison_id
                ),
                "minuend": (
                    comparison["minuend"]
                ),
                "subtrahend": (
                    comparison[
                        "subtrahend"
                    ]
                ),
                "group_dimension": (
                    "regime"
                ),
                "group_value": regime,
                "group_count": len(rows),
            }

            for metric in (
                STRUCTURAL_METRIC_KEYS
            ):
                aggregate_row[
                    f"delta_{metric}"
                ] = format_float(
                    mean_metric(
                        rows,
                        f"delta_{metric}",
                    )
                )

            paired_aggregate_rows.append(
                aggregate_row
            )

    assert len(
        paired_aggregate_rows
    ) == 12

    exact_result_rows = []

    exact_condition_specs = (
        (
            "original_image_full_caption",
            original_images,
            "recomputed_and_matched_E1",
        ),
        (
            "grayscale_image_full_caption",
            grayscale_images,
            "computed_from_existing_embeddings",
        ),
    )

    for (
        condition,
        image_matrix,
        source,
    ) in exact_condition_specs:
        for query_index, caption in enumerate(
            full_rows
        ):
            caption_id = caption[
                "caption_id"
            ]

            text_row_index = usage_map[
                (
                    "positivos",
                    caption_id,
                )
            ]

            relevant_image_row_index = (
                image_row_by_id[
                    caption["image_id"]
                ]
            )

            scores = np.asarray(
                text_embeddings[
                    text_row_index
                ]
                @ image_matrix.T,
                dtype=np.float64,
            )

            result = evaluate_query(
                scores=scores,
                relevant_indices={
                    relevant_image_row_index
                },
                candidate_keys=image_ids,
            )

            ranking = [
                int(index)
                for index in result[
                    "ranking_indices"
                ]
            ]

            top1_index = ranking[0]

            if condition == (
                "original_image_full_caption"
            ):
                e1 = e1_by_caption[
                    caption_id
                ]

                assert int(
                    e1["query_index"]
                ) == query_index

                assert (
                    e1["image_id"]
                    == caption["image_id"]
                )

                assert int(
                    e1["relevant_rank"]
                ) == int(
                    result[
                        "first_relevant_rank"
                    ]
                )

                assert (
                    e1["top1_image_id"]
                    == image_ids[top1_index]
                )

                for metric in (
                    "recall_at_1",
                    "recall_at_5",
                    "mrr",
                    "ndcg_at_10",
                    "positive_margin",
                ):
                    assert_close(
                        float(result[metric]),
                        float(e1[metric]),
                        (
                            f"E1.{caption_id}."
                            f"{metric}"
                        ),
                    )

            exact_result_rows.append(
                {
                    "condition": condition,
                    "source": source,
                    "query_index": (
                        query_index
                    ),
                    "caption_id": (
                        caption_id
                    ),
                    "image_id": (
                        caption["image_id"]
                    ),
                    "split": (
                        caption["split"]
                    ),
                    "template_id": (
                        caption[
                            "template_id"
                        ]
                    ),
                    "is_canonical": (
                        caption[
                            "is_canonical"
                        ].strip().lower()
                    ),
                    "text_row_index": (
                        text_row_index
                    ),
                    "relevant_image_row_index": (
                        relevant_image_row_index
                    ),
                    "relevant_rank": int(
                        result[
                            "first_relevant_rank"
                        ]
                    ),
                    "top1_image_id": (
                        image_ids[top1_index]
                    ),
                    "top1_score": (
                        format_float(
                            scores[top1_index]
                        )
                    ),
                    "relevant_score": (
                        format_float(
                            scores[
                                relevant_image_row_index
                            ]
                        )
                    ),
                    "positive_margin": (
                        format_float(
                            result[
                                "positive_margin"
                            ]
                        )
                    ),
                    "recall_at_1": (
                        format_float(
                            result[
                                "recall_at_1"
                            ]
                        )
                    ),
                    "recall_at_5": (
                        format_float(
                            result[
                                "recall_at_5"
                            ]
                        )
                    ),
                    "mrr": format_float(
                        result["mrr"]
                    ),
                    "ndcg_at_10": (
                        format_float(
                            result[
                                "ndcg_at_10"
                            ]
                        )
                    ),
                }
            )

    assert len(exact_result_rows) == 560

    exact_rows_by_condition: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in exact_result_rows:
        exact_rows_by_condition[
            row["condition"]
        ].append(row)

    exact_aggregate_rows = []

    for condition in (
        "original_image_full_caption",
        "grayscale_image_full_caption",
    ):
        rows = exact_rows_by_condition[
            condition
        ]

        assert len(rows) == 280

        exact_aggregate_rows.append(
            exact_aggregate_row(
                condition,
                "overall",
                "all",
                rows,
            )
        )

        for dimension in (
            "split",
            "is_canonical",
        ):
            grouped: dict[
                str,
                list[dict[str, Any]],
            ] = defaultdict(list)

            for row in rows:
                grouped[
                    str(row[dimension])
                ].append(row)

            for value in sorted(grouped):
                exact_aggregate_rows.append(
                    exact_aggregate_row(
                        condition,
                        dimension,
                        value,
                        grouped[value],
                    )
                )

    assert len(exact_aggregate_rows) == 14

    if TEMPORARY_DIRECTORY.exists():
        shutil.rmtree(
            TEMPORARY_DIRECTORY
        )

    TEMPORARY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    paths = {
        "raw_results": (
            TEMPORARY_DIRECTORY
            / RAW_RESULTS_FILENAME
        ),
        "ranking": (
            TEMPORARY_DIRECTORY
            / STRUCTURAL_RANKING_FILENAME
        ),
        "group_results": (
            TEMPORARY_DIRECTORY
            / GROUP_RESULTS_FILENAME
        ),
        "structural_aggregates": (
            TEMPORARY_DIRECTORY
            / STRUCTURAL_AGGREGATES_FILENAME
        ),
        "paired_results": (
            TEMPORARY_DIRECTORY
            / PAIRED_RESULTS_FILENAME
        ),
        "paired_aggregates": (
            TEMPORARY_DIRECTORY
            / PAIRED_AGGREGATES_FILENAME
        ),
        "exact_results": (
            TEMPORARY_DIRECTORY
            / EXACT_RESULTS_FILENAME
        ),
        "exact_aggregates": (
            TEMPORARY_DIRECTORY
            / EXACT_AGGREGATES_FILENAME
        ),
        "summary": (
            TEMPORARY_DIRECTORY
            / SUMMARY_FILENAME
        ),
    }

    try:
        write_csv(
            paths["raw_results"],
            RAW_FIELDS,
            raw_result_rows,
        )

        write_csv(
            paths["ranking"],
            RANKING_FIELDS,
            ranking_rows,
        )

        write_csv(
            paths["group_results"],
            GROUP_FIELDS,
            group_result_rows,
        )

        write_csv(
            paths[
                "structural_aggregates"
            ],
            STRUCTURAL_AGGREGATE_FIELDS,
            structural_aggregate_rows,
        )

        write_csv(
            paths["paired_results"],
            PAIRED_FIELDS,
            paired_rows,
        )

        write_csv(
            paths["paired_aggregates"],
            PAIRED_AGGREGATE_FIELDS,
            paired_aggregate_rows,
        )

        write_csv(
            paths["exact_results"],
            EXACT_FIELDS,
            exact_result_rows,
        )

        write_csv(
            paths["exact_aggregates"],
            EXACT_AGGREGATE_FIELDS,
            exact_aggregate_rows,
        )

        output_rows = {
            "raw_results": 640,
            "ranking": 35840,
            "group_results": 160,
            "structural_aggregates": 40,
            "paired_results": 160,
            "paired_aggregates": 12,
            "exact_results": 560,
            "exact_aggregates": 14,
        }

        output_artifacts = {}

        for key, row_count in (
            output_rows.items()
        ):
            output_path = paths[key]

            output_artifacts[key] = {
                "path": (
                    OUTPUT_DIRECTORY
                    / output_path.name
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": row_count,
                "sha256": sha256_file(
                    output_path
                ),
            }

        structural_overall = {
            row["condition"]: {
                metric: float(row[metric])
                for metric
                in STRUCTURAL_METRIC_KEYS
            }
            | {
                "group_count": int(
                    row["group_count"]
                )
            }
            for row
            in structural_aggregate_rows
            if (
                row["group_dimension"]
                == "overall"
            )
        }

        exact_overall = {
            row["condition"]: {
                metric: float(row[metric])
                for metric
                in EXACT_METRIC_KEYS
            }
            | {
                "query_count": int(
                    row["query_count"]
                )
            }
            for row
            in exact_aggregate_rows
            if (
                row["group_dimension"]
                == "overall"
            )
        }

        paired_overall = {
            row["comparison_id"]: {
                key: float(value)
                for key, value
                in row.items()
                if key.startswith(
                    "delta_"
                )
            }
            | {
                "group_count": int(
                    row["group_count"]
                ),
                "minuend": (
                    row["minuend"]
                ),
                "subtrahend": (
                    row["subtrahend"]
                ),
            }
            for row
            in paired_aggregate_rows
            if (
                row["group_dimension"]
                == "overall"
            )
        }

        summary = {
            "schema_version": "1.0",
            "dataset_version": "v2",
            "experiment_id": "E4",
            "experiment_name": (
                "color_ablations"
            ),
            "conditions": list(
                CONDITION_ORDER
            ),
            "protocol": {
                "gallery_count": 56,
                "structural_group_count": 40,
                "structural_relevant_count": 7,
                "raw_structural_query_rows": 640,
                "group_condition_rows": 160,
                "exact_secondary_rows": 560,
                "similarity": (
                    "dot_product_of_l2_"
                    "normalized_embeddings"
                ),
                "tie_breaker": (
                    "image_id_ascending"
                ),
                "legacy_recall_mapping": {
                    "recall_at_1": (
                        "structural_hit_at_1"
                    ),
                    "recall_at_5": (
                        "structural_hit_at_5"
                    ),
                },
            },
            "counts": {
                "raw_structural_results": 640,
                "structural_ranking_rows": 35840,
                "structural_group_results": 160,
                "structural_aggregate_rows": 40,
                "paired_group_rows": 160,
                "paired_aggregate_rows": 12,
                "exact_results": 560,
                "exact_aggregate_rows": 14,
            },
            "structural_overall_metrics": (
                structural_overall
            ),
            "paired_overall_differences": (
                paired_overall
            ),
            "exact_overall_metrics": (
                exact_overall
            ),
            "input_artifacts": {
                "contract": {
                    "path": (
                        CONFIG_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        CONFIG_PATH
                    ),
                },
                "full_captions": {
                    "path": (
                        FULL_CAPTIONS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 280,
                    "sha256": sha256_file(
                        FULL_CAPTIONS_PATH
                    ),
                },
                "colorless_captions": {
                    "path": (
                        COLORLESS_CAPTIONS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 40,
                    "sha256": sha256_file(
                        COLORLESS_CAPTIONS_PATH
                    ),
                },
                "image_index": {
                    "path": (
                        IMAGE_INDEX_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 56,
                    "sha256": sha256_file(
                        IMAGE_INDEX_PATH
                    ),
                },
                "text_usage": {
                    "path": (
                        TEXT_USAGE_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 600,
                    "sha256": sha256_file(
                        TEXT_USAGE_PATH
                    ),
                },
                "original_image_embeddings": {
                    "path": (
                        ORIGINAL_IMAGE_EMBEDDINGS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        ORIGINAL_IMAGE_EMBEDDINGS_PATH
                    ),
                },
                "grayscale_image_embeddings": {
                    "path": (
                        GRAYSCALE_IMAGE_EMBEDDINGS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        GRAYSCALE_IMAGE_EMBEDDINGS_PATH
                    ),
                },
                "text_embeddings": {
                    "path": (
                        TEXT_EMBEDDINGS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        TEXT_EMBEDDINGS_PATH
                    ),
                },
                "e1_results": {
                    "path": (
                        E1_RESULTS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 280,
                    "sha256": sha256_file(
                        E1_RESULTS_PATH
                    ),
                },
            },
            "output_artifacts": (
                output_artifacts
            ),
            "runtime": {
                "python": (
                    platform.python_version()
                ),
                "numpy": np.__version__,
            },
            "evaluation_valid": True,
        }

        write_json(
            paths["summary"],
            summary,
        )

        expected_files = {
            RAW_RESULTS_FILENAME,
            STRUCTURAL_RANKING_FILENAME,
            GROUP_RESULTS_FILENAME,
            STRUCTURAL_AGGREGATES_FILENAME,
            PAIRED_RESULTS_FILENAME,
            PAIRED_AGGREGATES_FILENAME,
            EXACT_RESULTS_FILENAME,
            EXACT_AGGREGATES_FILENAME,
            SUMMARY_FILENAME,
        }

        actual_files = {
            path.name
            for path
            in TEMPORARY_DIRECTORY.iterdir()
            if path.is_file()
        }

        assert actual_files == expected_files

        publish_directory(
            TEMPORARY_DIRECTORY,
            OUTPUT_DIRECTORY,
        )

    except Exception:
        if TEMPORARY_DIRECTORY.exists():
            shutil.rmtree(
                TEMPORARY_DIRECTORY
            )

        raise

    print()
    print("=" * 88)
    print("EVALUACIÓN E4 COMPLETADA")
    print("=" * 88)

    print()
    print("MÉTRICAS ESTRUCTURALES — 40 GRUPOS")

    for condition in CONDITION_ORDER:
        metrics = structural_overall[
            condition
        ]

        print()
        print(condition)
        print(
            "- Hit@1:",
            format_float(
                metrics[
                    "structural_hit_at_1"
                ]
            ),
        )
        print(
            "- Hit@5:",
            format_float(
                metrics[
                    "structural_hit_at_5"
                ]
            ),
        )
        print(
            "- Fractional Recall@5:",
            format_float(
                metrics[
                    "structural_fractional_recall_at_5"
                ]
            ),
        )
        print(
            "- MRR:",
            format_float(
                metrics[
                    "structural_mrr"
                ]
            ),
        )
        print(
            "- nDCG@10:",
            format_float(
                metrics[
                    "structural_ndcg_at_10"
                ]
            ),
        )
        print(
            "- Best relevant margin:",
            format_float(
                metrics[
                    "best_relevant_margin"
                ]
            ),
        )

    print()
    print("EVALUACIÓN EXACTA SECUNDARIA")

    for condition in (
        "original_image_full_caption",
        "grayscale_image_full_caption",
    ):
        metrics = exact_overall[
            condition
        ]

        print()
        print(condition)
        print(
            "- Recall@1:",
            format_float(
                metrics["recall_at_1"]
            ),
        )
        print(
            "- Recall@5:",
            format_float(
                metrics["recall_at_5"]
            ),
        )
        print(
            "- MRR:",
            format_float(
                metrics["mrr"]
            ),
        )
        print(
            "- nDCG@10:",
            format_float(
                metrics["ndcg_at_10"]
            ),
        )
        print(
            "- Margen:",
            format_float(
                metrics[
                    "positive_margin"
                ]
            ),
        )

    print()
    print("Artefactos:", 9)
    print(
        "Directorio:",
        OUTPUT_DIRECTORY
        .relative_to(
            PROJECT_ROOT
        )
        .as_posix(),
    )
    print("Evaluación válida: True")


if __name__ == "__main__":
    main()
