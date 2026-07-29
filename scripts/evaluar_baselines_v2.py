"""Compara OpenCLIP con los baselines aleatorio y cromático HSV en E3."""

from __future__ import annotations

import colorsys
import csv
import hashlib
import json
import math
import platform
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from metricas_retrieval_v2 import (
    aggregate_query_metrics,
    evaluate_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINES_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "baselines_v2.json"
)

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "experimento_v2.json"
)

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
)

IMAGE_INDEX_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "embeddings"
    / "index_images_v2.csv"
)

OPENCLIP_RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "global_openclip_v2"
    / "resultados_consulta_global_openclip_v2.csv"
)

OPENCLIP_SUMMARY_PATH = (
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
    / "baselines_v2"
)

TEMPORARY_DIRECTORY = (
    OUTPUT_DIRECTORY.with_name(
        OUTPUT_DIRECTORY.name + ".tmp"
    )
)

RESULTS_FILENAME = (
    "resultados_consulta_baselines_v2.csv"
)

RANDOM_RANKING_FILENAME = (
    "ranking_random_v2.csv"
)

COLOR_RANKING_FILENAME = (
    "ranking_color_histogram_v2.csv"
)

AGGREGATES_FILENAME = (
    "agregados_baselines_v2.csv"
)

COMPARISON_FILENAME = (
    "comparacion_baselines_v2.csv"
)

SUMMARY_FILENAME = (
    "resumen_baselines_v2.json"
)

CONDITION_ORDER = (
    "openclip",
    "random",
    "color_histogram",
)

RESULT_FIELDS = (
    "condition",
    "source",
    "query_index",
    "caption_id",
    "image_id",
    "semantic_id",
    "split",
    "pattern_id",
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "ambiguity_level",
    "template_id",
    "is_canonical",
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

RANKING_FIELDS = (
    "condition",
    "query_index",
    "caption_id",
    "rank",
    "image_row_index",
    "image_id",
    "score",
    "is_relevant",
    "image_split",
    "image_pattern_id",
    "image_palette_id",
)

AGGREGATE_FIELDS = (
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

COMPARISON_FIELDS = (
    "condition",
    "query_count",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
    "positive_margin",
    "delta_recall_at_1_vs_openclip",
    "delta_mrr_vs_openclip",
    "delta_ndcg_at_10_vs_openclip",
    "delta_positive_margin_vs_openclip",
    "delta_recall_at_1_vs_random",
    "delta_mrr_vs_random",
    "delta_ndcg_at_10_vs_random",
    "delta_positive_margin_vs_random",
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


def parse_bool(
    value: str,
    context: str,
) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1"}:
        return True

    if normalized in {"false", "0"}:
        return False

    raise ValueError(
        f"{context}: booleano no reconocido: "
        f"{value!r}."
    )


def format_float(value: float) -> str:
    return format(
        float(value),
        ".12f",
    )


def assert_close(
    actual: float,
    expected: float,
    context: str,
    tolerance: float = 1e-12,
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


def histogram_bin(
    value: float,
    bin_count: int,
) -> int:
    assert value >= 0.0
    assert value <= 1.0
    assert bin_count > 0

    index = int(
        math.floor(
            value * bin_count
        )
    )

    return min(
        index,
        bin_count - 1,
    )


def descriptor_from_rgb_counts(
    rgb_colors: list[tuple[int, int, int]],
    weights: np.ndarray,
    h_bins: int,
    s_bins: int,
    v_bins: int,
) -> np.ndarray:
    assert rgb_colors
    assert weights.ndim == 1
    assert len(rgb_colors) == weights.size
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert float(weights.sum()) > 0.0

    dimensions = (
        h_bins
        * s_bins
        * v_bins
    )

    descriptor = np.zeros(
        dimensions,
        dtype=np.float64,
    )

    normalized_weights = (
        weights
        / weights.sum()
    )

    for rgb, weight in zip(
        rgb_colors,
        normalized_weights,
        strict=True,
    ):
        red, green, blue = rgb

        hue, saturation, value = (
            colorsys.rgb_to_hsv(
                red / 255.0,
                green / 255.0,
                blue / 255.0,
            )
        )

        h_index = histogram_bin(
            hue,
            h_bins,
        )

        s_index = histogram_bin(
            saturation,
            s_bins,
        )

        v_index = histogram_bin(
            value,
            v_bins,
        )

        flat_index = (
            (
                h_index
                * s_bins
                + s_index
            )
            * v_bins
            + v_index
        )

        descriptor[
            flat_index
        ] += float(weight)

    assert descriptor.shape == (
        dimensions,
    )

    assert np.isfinite(
        descriptor
    ).all()

    assert np.all(
        descriptor >= 0.0
    )

    assert_close(
        float(descriptor.sum()),
        1.0,
        "descriptor_l1",
    )

    return descriptor


def build_palette_mapping(
    experiment: dict[str, Any],
) -> dict[str, list[tuple[int, int, int]]]:
    dataset = experiment[
        "dataset_v2"
    ]

    mapping: dict[
        str,
        list[tuple[int, int, int]],
    ] = {}

    for section_name in (
        "base_palettes",
        "heldout_palettes",
    ):
        for palette_id, colors in (
            dataset[
                section_name
            ].items()
        ):
            assert palette_id not in mapping

            mapping[palette_id] = [
                tuple(
                    int(channel)
                    for channel in color
                )
                for color in colors
            ]

    assert len(mapping) == 7

    return mapping


def validate_contract(
    baselines_config: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    assert (
        baselines_config[
            "experiment_id"
        ]
        == "E3"
    )

    assert (
        baselines_config[
            "experiment_name"
        ]
        == "baseline_comparison"
    )

    experiments = {
        record["experiment_id"]: record
        for record in experiment[
            "experiments"
        ]
    }

    assert experiments["E3"] == {
        "experiment_id": "E3",
        "name": "baseline_comparison",
        "conditions": [
            "openclip",
            "random",
            "color_histogram",
        ],
    }

    conditions = baselines_config[
        "conditions"
    ]

    assert tuple(
        conditions
    ) == CONDITION_ORDER

    random_config = conditions[
        "random"
    ]

    assert random_config["seed"] == 225

    assert (
        random_config[
            "bit_generator"
        ]
        == "PCG64"
    )

    assert (
        random_config[
            "matrix_shape"
        ]
        == [280, 56]
    )

    color_config = conditions[
        "color_histogram"
    ]

    assert color_config["bins"] == {
        "h": 18,
        "s": 4,
        "v": 4,
    }

    assert (
        color_config[
            "descriptor_dimensions"
        ]
        == 288
    )

    assert (
        color_config["similarity"]
        == "histogram_intersection"
    )


def build_image_descriptors(
    ordered_images: list[dict[str, str]],
    h_bins: int,
    s_bins: int,
    v_bins: int,
) -> np.ndarray:
    rows = []

    for image_record in ordered_images:
        image = np.asarray(
            Image.open(
                PROJECT_ROOT
                / image_record["image_path"]
            ).convert("RGB"),
            dtype=np.uint8,
        )

        pixels = image.reshape(-1, 3)

        colors_array, counts = np.unique(
            pixels,
            axis=0,
            return_counts=True,
        )

        colors = [
            tuple(
                int(channel)
                for channel in color
            )
            for color in colors_array
        ]

        descriptor = descriptor_from_rgb_counts(
            rgb_colors=colors,
            weights=counts.astype(
                np.float64
            ),
            h_bins=h_bins,
            s_bins=s_bins,
            v_bins=v_bins,
        )

        rows.append(descriptor)

    matrix = np.asarray(
        rows,
        dtype=np.float64,
    )

    assert matrix.shape == (
        56,
        h_bins * s_bins * v_bins,
    )

    return matrix


def evaluate_computed_condition(
    condition: str,
    score_matrix: np.ndarray,
    captions: list[dict[str, str]],
    ordered_images: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    assert condition in {
        "random",
        "color_histogram",
    }

    assert score_matrix.shape == (
        280,
        56,
    )

    assert np.isfinite(
        score_matrix
    ).all()

    image_ids = [
        row["image_id"]
        for row in ordered_images
    ]

    image_row_by_id = {
        image_id: row_index
        for row_index, image_id
        in enumerate(image_ids)
    }

    image_record_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    query_rows: list[
        dict[str, Any]
    ] = []

    ranking_rows: list[
        dict[str, Any]
    ] = []

    metric_rows: list[
        dict[str, Any]
    ] = []

    for query_index, caption in enumerate(
        captions
    ):
        caption_id = caption["caption_id"]
        image_id = caption["image_id"]

        assert image_id in image_row_by_id

        relevant_row = image_row_by_id[
            image_id
        ]

        relevant_image = (
            image_record_by_id[
                image_id
            ]
        )

        for metadata_field in (
            "semantic_id",
            "split",
            "pattern_id",
            "palette_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
        ):
            assert (
                caption[metadata_field]
                == relevant_image[
                    metadata_field
                ]
            )

        scores = score_matrix[
            query_index
        ]

        result = evaluate_query(
            scores=scores,
            relevant_indices={
                relevant_row
            },
            candidate_keys=image_ids,
        )

        ranking = result[
            "ranking_indices"
        ]

        top1_row = int(ranking[0])

        canonical = str(
            parse_bool(
                caption["is_canonical"],
                caption_id,
            )
        ).lower()

        metric_rows.append(
            {
                "condition": condition,
                "split": caption["split"],
                "is_canonical": canonical,
                "recall_at_1": (
                    result["recall_at_1"]
                ),
                "recall_at_5": (
                    result["recall_at_5"]
                ),
                "mrr": result["mrr"],
                "ndcg_at_10": (
                    result["ndcg_at_10"]
                ),
                "positive_margin": (
                    result[
                        "positive_margin"
                    ]
                ),
            }
        )

        query_rows.append(
            {
                "condition": condition,
                "source": (
                    "computed_from_"
                    "baselines_v2_contract"
                ),
                "query_index": query_index,
                "caption_id": caption_id,
                "image_id": image_id,
                "semantic_id": (
                    caption["semantic_id"]
                ),
                "split": caption["split"],
                "pattern_id": (
                    caption["pattern_id"]
                ),
                "palette_id": (
                    caption["palette_id"]
                ),
                "motif": caption["motif"],
                "orientation": (
                    caption["orientation"]
                ),
                "composition": (
                    caption["composition"]
                ),
                "symmetry": (
                    caption["symmetry"]
                ),
                "ambiguity_level": (
                    relevant_image[
                        "ambiguity_level"
                    ]
                ),
                "template_id": (
                    caption["template_id"]
                ),
                "is_canonical": canonical,
                "relevant_rank": (
                    result[
                        "first_relevant_rank"
                    ]
                ),
                "top1_image_id": (
                    image_ids[top1_row]
                ),
                "top1_score": format_float(
                    scores[top1_row]
                ),
                "relevant_score": (
                    format_float(
                        scores[relevant_row]
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

        for rank, image_row_index in enumerate(
            ranking,
            start=1,
        ):
            row_index = int(
                image_row_index
            )

            image_record = (
                ordered_images[
                    row_index
                ]
            )

            ranking_rows.append(
                {
                    "condition": condition,
                    "query_index": (
                        query_index
                    ),
                    "caption_id": (
                        caption_id
                    ),
                    "rank": rank,
                    "image_row_index": (
                        row_index
                    ),
                    "image_id": (
                        image_record[
                            "image_id"
                        ]
                    ),
                    "score": format_float(
                        scores[row_index]
                    ),
                    "is_relevant": str(
                        row_index
                        == relevant_row
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

    assert len(query_rows) == 280

    assert len(ranking_rows) == (
        280 * 56
    )

    assert len(metric_rows) == 280

    return (
        query_rows,
        ranking_rows,
        metric_rows,
    )


def load_openclip_condition(
    captions: list[dict[str, str]],
    ordered_images: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    _, source_rows = load_csv(
        OPENCLIP_RESULTS_PATH
    )

    summary = load_json(
        OPENCLIP_SUMMARY_PATH
    )

    assert (
        summary["evaluation_valid"]
        is True
    )

    assert summary["experiment_id"] == "E1"

    ordered_source_rows = sorted(
        source_rows,
        key=lambda row: int(
            row["query_index"]
        ),
    )

    assert len(
        ordered_source_rows
    ) == 280

    image_record_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    query_rows: list[
        dict[str, Any]
    ] = []

    metric_rows: list[
        dict[str, Any]
    ] = []

    for query_index, (
        caption,
        source,
    ) in enumerate(
        zip(
            captions,
            ordered_source_rows,
            strict=True,
        )
    ):
        caption_id = caption[
            "caption_id"
        ]

        assert int(
            source["query_index"]
        ) == query_index

        assert (
            source["caption_id"]
            == caption_id
        )

        assert (
            source["image_id"]
            == caption["image_id"]
        )

        image_record = (
            image_record_by_id[
                caption["image_id"]
            ]
        )

        for field in (
            "semantic_id",
            "split",
            "pattern_id",
            "palette_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
        ):
            assert (
                source[field]
                == caption[field]
            )

            assert (
                caption[field]
                == image_record[field]
            )

        assert (
            source["ambiguity_level"]
            == image_record[
                "ambiguity_level"
            ]
        )

        canonical = str(
            parse_bool(
                source["is_canonical"],
                caption_id,
            )
        ).lower()

        query_rows.append(
            {
                "condition": "openclip",
                "source": (
                    "reused_validated_e1"
                ),
                "query_index": (
                    query_index
                ),
                "caption_id": (
                    caption_id
                ),
                "image_id": (
                    source["image_id"]
                ),
                "semantic_id": (
                    source["semantic_id"]
                ),
                "split": source["split"],
                "pattern_id": (
                    source["pattern_id"]
                ),
                "palette_id": (
                    source["palette_id"]
                ),
                "motif": source["motif"],
                "orientation": (
                    source["orientation"]
                ),
                "composition": (
                    source["composition"]
                ),
                "symmetry": (
                    source["symmetry"]
                ),
                "ambiguity_level": (
                    source[
                        "ambiguity_level"
                    ]
                ),
                "template_id": (
                    source["template_id"]
                ),
                "is_canonical": canonical,
                "relevant_rank": (
                    source["relevant_rank"]
                ),
                "top1_image_id": (
                    source["top1_image_id"]
                ),
                "top1_score": (
                    source["top1_score"]
                ),
                "relevant_score": (
                    source["relevant_score"]
                ),
                "positive_margin": (
                    source["positive_margin"]
                ),
                "recall_at_1": (
                    source["recall_at_1"]
                ),
                "recall_at_5": (
                    source["recall_at_5"]
                ),
                "mrr": source["mrr"],
                "ndcg_at_10": (
                    source["ndcg_at_10"]
                ),
            }
        )

        metric_rows.append(
            {
                "condition": "openclip",
                "split": source["split"],
                "is_canonical": canonical,
                "recall_at_1": float(
                    source[
                        "recall_at_1"
                    ]
                ),
                "recall_at_5": float(
                    source[
                        "recall_at_5"
                    ]
                ),
                "mrr": float(
                    source["mrr"]
                ),
                "ndcg_at_10": float(
                    source[
                        "ndcg_at_10"
                    ]
                ),
                "positive_margin": float(
                    source[
                        "positive_margin"
                    ]
                ),
            }
        )

    aggregate = aggregate_query_metrics(
        metric_rows
    )

    for metric in (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
        "query_count",
    ):
        assert_close(
            float(aggregate[metric]),
            float(
                summary[
                    "overall_metrics"
                ][metric]
            ),
            f"openclip.{metric}",
        )

    return query_rows, metric_rows


def aggregate_group(
    condition: str,
    rows: list[dict[str, Any]],
    dimension: str,
    value: str,
) -> dict[str, Any]:
    aggregate = aggregate_query_metrics(
        rows
    )

    return {
        "condition": condition,
        "group_dimension": dimension,
        "group_value": value,
        "query_count": len(rows),
        "recall_at_1": format_float(
            aggregate["recall_at_1"]
        ),
        "recall_at_5": format_float(
            aggregate["recall_at_5"]
        ),
        "mrr": format_float(
            aggregate["mrr"]
        ),
        "ndcg_at_10": format_float(
            aggregate["ndcg_at_10"]
        ),
        "positive_margin": format_float(
            aggregate["positive_margin"]
        ),
    }


def build_aggregates(
    metric_rows_by_condition: dict[
        str,
        list[dict[str, Any]],
    ],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, float]],
]:
    aggregate_rows: list[
        dict[str, Any]
    ] = []

    overall_by_condition: dict[
        str,
        dict[str, float],
    ] = {}

    for condition in CONDITION_ORDER:
        rows = metric_rows_by_condition[
            condition
        ]

        raw_overall = (
            aggregate_query_metrics(
                rows
            )
        )

        overall_by_condition[
            condition
        ] = {
            key: float(
                raw_overall[key]
            )
            for key in (
                "recall_at_1",
                "recall_at_5",
                "mrr",
                "ndcg_at_10",
                "positive_margin",
                "query_count",
            )
        }

        aggregate_rows.append(
            aggregate_group(
                condition,
                rows,
                "overall",
                "all",
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
                aggregate_rows.append(
                    aggregate_group(
                        condition,
                        grouped[value],
                        dimension,
                        value,
                    )
                )

    assert len(
        aggregate_rows
    ) == 21

    return (
        aggregate_rows,
        overall_by_condition,
    )


def build_comparison(
    overall: dict[
        str,
        dict[str, float],
    ],
) -> list[dict[str, Any]]:
    openclip = overall["openclip"]
    random = overall["random"]

    rows = []

    for condition in CONDITION_ORDER:
        metrics = overall[condition]

        rows.append(
            {
                "condition": condition,
                "query_count": int(
                    metrics["query_count"]
                ),
                "recall_at_1": (
                    format_float(
                        metrics[
                            "recall_at_1"
                        ]
                    )
                ),
                "recall_at_5": (
                    format_float(
                        metrics[
                            "recall_at_5"
                        ]
                    )
                ),
                "mrr": format_float(
                    metrics["mrr"]
                ),
                "ndcg_at_10": (
                    format_float(
                        metrics[
                            "ndcg_at_10"
                        ]
                    )
                ),
                "positive_margin": (
                    format_float(
                        metrics[
                            "positive_margin"
                        ]
                    )
                ),
                "delta_recall_at_1_vs_openclip": (
                    format_float(
                        metrics[
                            "recall_at_1"
                        ]
                        - openclip[
                            "recall_at_1"
                        ]
                    )
                ),
                "delta_mrr_vs_openclip": (
                    format_float(
                        metrics["mrr"]
                        - openclip["mrr"]
                    )
                ),
                "delta_ndcg_at_10_vs_openclip": (
                    format_float(
                        metrics[
                            "ndcg_at_10"
                        ]
                        - openclip[
                            "ndcg_at_10"
                        ]
                    )
                ),
                "delta_positive_margin_vs_openclip": (
                    format_float(
                        metrics[
                            "positive_margin"
                        ]
                        - openclip[
                            "positive_margin"
                        ]
                    )
                ),
                "delta_recall_at_1_vs_random": (
                    format_float(
                        metrics[
                            "recall_at_1"
                        ]
                        - random[
                            "recall_at_1"
                        ]
                    )
                ),
                "delta_mrr_vs_random": (
                    format_float(
                        metrics["mrr"]
                        - random["mrr"]
                    )
                ),
                "delta_ndcg_at_10_vs_random": (
                    format_float(
                        metrics[
                            "ndcg_at_10"
                        ]
                        - random[
                            "ndcg_at_10"
                        ]
                    )
                ),
                "delta_positive_margin_vs_random": (
                    format_float(
                        metrics[
                            "positive_margin"
                        ]
                        - random[
                            "positive_margin"
                        ]
                    )
                ),
            }
        )

    assert len(rows) == 3

    return rows


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
        if backup_directory.exists():
            if not output_directory.exists():
                backup_directory.replace(
                    output_directory
                )

        raise

    if backup_directory.exists():
        shutil.rmtree(
            backup_directory
        )


def main() -> None:
    baselines_config = load_json(
        BASELINES_CONFIG_PATH
    )

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    validate_contract(
        baselines_config,
        experiment,
    )

    _, captions = load_csv(
        CAPTIONS_PATH
    )

    _, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    assert len(captions) == 280
    assert len(image_rows) == 56

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

    assert len(
        {
            row["caption_id"]
            for row in captions
        }
    ) == 280

    assert len(
        {
            row["image_id"]
            for row in ordered_images
        }
    ) == 56

    print("=" * 80)
    print("EVALUACIÓN DE BASELINES V2 — E3")
    print("=" * 80)
    print("Consultas:", len(captions))
    print("Galería:", len(ordered_images))
    print(
        "Condiciones:",
        ", ".join(CONDITION_ORDER),
    )

    openclip_query_rows, openclip_metric_rows = (
        load_openclip_condition(
            captions,
            ordered_images,
        )
    )

    random_config = baselines_config[
        "conditions"
    ][
        "random"
    ]

    random_generator = np.random.Generator(
        np.random.PCG64(
            int(random_config["seed"])
        )
    )

    random_scores = (
        random_generator.random(
            (280, 56)
        )
    )

    assert format_float(
        random_scores[0, 0]
    ) == "0.602528611399"

    (
        random_query_rows,
        random_ranking_rows,
        random_metric_rows,
    ) = evaluate_computed_condition(
        condition="random",
        score_matrix=random_scores,
        captions=captions,
        ordered_images=ordered_images,
    )

    color_config = baselines_config[
        "conditions"
    ][
        "color_histogram"
    ]

    h_bins = int(
        color_config["bins"]["h"]
    )

    s_bins = int(
        color_config["bins"]["s"]
    )

    v_bins = int(
        color_config["bins"]["v"]
    )

    image_descriptors = (
        build_image_descriptors(
            ordered_images,
            h_bins,
            s_bins,
            v_bins,
        )
    )

    palette_mapping = (
        build_palette_mapping(
            experiment
        )
    )

    palette_descriptors = {
        palette_id: (
            descriptor_from_rgb_counts(
                rgb_colors=colors,
                weights=np.ones(
                    len(colors),
                    dtype=np.float64,
                ),
                h_bins=h_bins,
                s_bins=s_bins,
                v_bins=v_bins,
            )
        )
        for palette_id, colors
        in palette_mapping.items()
    }

    color_scores = np.zeros(
        (280, 56),
        dtype=np.float64,
    )

    for query_index, caption in enumerate(
        captions
    ):
        palette_id = caption[
            "palette_id"
        ]

        assert (
            palette_id
            in palette_descriptors
        )

        prototype = (
            palette_descriptors[
                palette_id
            ]
        )

        color_scores[
            query_index
        ] = np.minimum(
            image_descriptors,
            prototype[None, :],
        ).sum(axis=1)

    assert np.isfinite(
        color_scores
    ).all()

    assert float(
        color_scores.min()
    ) >= -1e-12

    assert float(
        color_scores.max()
    ) <= 1.0 + 1e-12

    (
        color_query_rows,
        color_ranking_rows,
        color_metric_rows,
    ) = evaluate_computed_condition(
        condition="color_histogram",
        score_matrix=color_scores,
        captions=captions,
        ordered_images=ordered_images,
    )

    result_rows = (
        openclip_query_rows
        + random_query_rows
        + color_query_rows
    )

    assert len(result_rows) == 840

    metric_rows_by_condition = {
        "openclip": (
            openclip_metric_rows
        ),
        "random": (
            random_metric_rows
        ),
        "color_histogram": (
            color_metric_rows
        ),
    }

    aggregate_rows, overall = (
        build_aggregates(
            metric_rows_by_condition
        )
    )

    comparison_rows = build_comparison(
        overall
    )

    theoretical_random = (
        baselines_config[
            "theoretical_random_expectation"
        ]
    )

    random_deviation = {
        metric: (
            overall["random"][metric]
            - float(
                theoretical_random[
                    metric
                ]
            )
        )
        for metric in (
            "recall_at_1",
            "recall_at_5",
            "mrr",
            "ndcg_at_10",
        )
    }

    split_counts = Counter(
        row["split"]
        for row in captions
    )

    canonical_counts = Counter(
        str(
            parse_bool(
                row["is_canonical"],
                row["caption_id"],
            )
        ).lower()
        for row in captions
    )

    if TEMPORARY_DIRECTORY.exists():
        shutil.rmtree(
            TEMPORARY_DIRECTORY
        )

    TEMPORARY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    results_path = (
        TEMPORARY_DIRECTORY
        / RESULTS_FILENAME
    )

    random_ranking_path = (
        TEMPORARY_DIRECTORY
        / RANDOM_RANKING_FILENAME
    )

    color_ranking_path = (
        TEMPORARY_DIRECTORY
        / COLOR_RANKING_FILENAME
    )

    aggregates_path = (
        TEMPORARY_DIRECTORY
        / AGGREGATES_FILENAME
    )

    comparison_path = (
        TEMPORARY_DIRECTORY
        / COMPARISON_FILENAME
    )

    summary_path = (
        TEMPORARY_DIRECTORY
        / SUMMARY_FILENAME
    )

    try:
        write_csv(
            results_path,
            RESULT_FIELDS,
            result_rows,
        )

        write_csv(
            random_ranking_path,
            RANKING_FIELDS,
            random_ranking_rows,
        )

        write_csv(
            color_ranking_path,
            RANKING_FIELDS,
            color_ranking_rows,
        )

        write_csv(
            aggregates_path,
            AGGREGATE_FIELDS,
            aggregate_rows,
        )

        write_csv(
            comparison_path,
            COMPARISON_FIELDS,
            comparison_rows,
        )

        output_artifacts = {
            "query_results": {
                "path": (
                    OUTPUT_DIRECTORY
                    / RESULTS_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 840,
                "sha256": sha256_file(
                    results_path
                ),
            },
            "random_ranking": {
                "path": (
                    OUTPUT_DIRECTORY
                    / RANDOM_RANKING_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 15680,
                "sha256": sha256_file(
                    random_ranking_path
                ),
            },
            "color_ranking": {
                "path": (
                    OUTPUT_DIRECTORY
                    / COLOR_RANKING_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 15680,
                "sha256": sha256_file(
                    color_ranking_path
                ),
            },
            "aggregates": {
                "path": (
                    OUTPUT_DIRECTORY
                    / AGGREGATES_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 21,
                "sha256": sha256_file(
                    aggregates_path
                ),
            },
            "comparison": {
                "path": (
                    OUTPUT_DIRECTORY
                    / COMPARISON_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 3,
                "sha256": sha256_file(
                    comparison_path
                ),
            },
        }

        summary = {
            "schema_version": "1.0",
            "dataset_version": "v2",
            "experiment_id": "E3",
            "experiment_name": (
                "baseline_comparison"
            ),
            "conditions": list(
                CONDITION_ORDER
            ),
            "protocol": {
                "task": (
                    "global_text_to_image_"
                    "retrieval"
                ),
                "query_count": 280,
                "gallery_count": 56,
                "relevant_images_per_query": 1,
                "tie_breaker": (
                    "image_id_ascending"
                ),
                "openclip_source": (
                    "validated_E1_results"
                ),
                "random_seed": 225,
                "random_generator": (
                    "numpy_PCG64"
                ),
                "color_space": "HSV",
                "color_bins": {
                    "h": h_bins,
                    "s": s_bins,
                    "v": v_bins,
                },
                "color_descriptor_dimensions": (
                    h_bins
                    * s_bins
                    * v_bins
                ),
                "color_similarity": (
                    "histogram_intersection"
                ),
            },
            "counts": {
                "query_rows": 840,
                "random_ranking_rows": 15680,
                "color_ranking_rows": 15680,
                "aggregate_rows": 21,
                "comparison_rows": 3,
                "splits": dict(
                    sorted(
                        split_counts.items()
                    )
                ),
                "is_canonical": dict(
                    sorted(
                        canonical_counts.items()
                    )
                ),
            },
            "overall_metrics": {
                condition: {
                    metric: float(value)
                    for metric, value
                    in overall[
                        condition
                    ].items()
                }
                for condition in CONDITION_ORDER
            },
            "theoretical_random_expectation": (
                theoretical_random
            ),
            "realized_random_minus_theoretical": {
                metric: float(value)
                for metric, value
                in random_deviation.items()
            },
            "input_artifacts": {
                "baselines_config": {
                    "path": (
                        BASELINES_CONFIG_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        BASELINES_CONFIG_PATH
                    ),
                },
                "experiment_config": {
                    "path": (
                        EXPERIMENT_CONFIG_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        EXPERIMENT_CONFIG_PATH
                    ),
                },
                "captions": {
                    "path": (
                        CAPTIONS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 280,
                    "sha256": sha256_file(
                        CAPTIONS_PATH
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
                "openclip_results": {
                    "path": (
                        OPENCLIP_RESULTS_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 280,
                    "sha256": sha256_file(
                        OPENCLIP_RESULTS_PATH
                    ),
                },
                "openclip_summary": {
                    "path": (
                        OPENCLIP_SUMMARY_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        OPENCLIP_SUMMARY_PATH
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
                "pillow": Image.__version__,
            },
            "evaluation_valid": True,
        }

        write_json(
            summary_path,
            summary,
        )

        expected_names = {
            RESULTS_FILENAME,
            RANDOM_RANKING_FILENAME,
            COLOR_RANKING_FILENAME,
            AGGREGATES_FILENAME,
            COMPARISON_FILENAME,
            SUMMARY_FILENAME,
        }

        actual_names = {
            path.name
            for path
            in TEMPORARY_DIRECTORY.iterdir()
            if path.is_file()
        }

        assert actual_names == expected_names

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
    print("=" * 80)
    print(
        "EVALUACIÓN DE BASELINES V2 COMPLETADA"
    )
    print("=" * 80)

    for condition in CONDITION_ORDER:
        metrics = overall[condition]

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
            "- Margen positivo:",
            format_float(
                metrics[
                    "positive_margin"
                ]
            ),
        )

    print()
    print(
        "Expectativa teórica aleatoria R@1:",
        format_float(
            theoretical_random[
                "recall_at_1"
            ]
        ),
    )

    print(
        "Realización aleatoria R@1:",
        format_float(
            overall[
                "random"
            ][
                "recall_at_1"
            ]
        ),
    )

    print()
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
