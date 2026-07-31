"""Valida independientemente la comparación de baselines de E3."""

from __future__ import annotations

import colorsys
import csv
import hashlib
import json
import math
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

QUERY_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "resultados_consulta_baselines_v2.csv"
)

RANDOM_RANKING_PATH = (
    OUTPUT_DIRECTORY
    / "ranking_random_v2.csv"
)

COLOR_RANKING_PATH = (
    OUTPUT_DIRECTORY
    / "ranking_color_histogram_v2.csv"
)

AGGREGATES_PATH = (
    OUTPUT_DIRECTORY
    / "agregados_baselines_v2.csv"
)

COMPARISON_PATH = (
    OUTPUT_DIRECTORY
    / "comparacion_baselines_v2.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "resumen_baselines_v2.json"
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

    raw_index = int(
        math.floor(
            value * bin_count
        )
    )

    return min(
        raw_index,
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
    mapping: dict[
        str,
        list[tuple[int, int, int]],
    ] = {}

    dataset = experiment[
        "dataset_v2"
    ]

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


def build_image_descriptors(
    ordered_images: list[dict[str, str]],
    h_bins: int,
    s_bins: int,
    v_bins: int,
) -> np.ndarray:
    descriptors = []

    for image_record in ordered_images:
        image_path = (
            PROJECT_ROOT
            / image_record["image_path"]
        )

        image = np.asarray(
            Image.open(
                image_path
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

        descriptors.append(
            descriptor_from_rgb_counts(
                rgb_colors=colors,
                weights=counts.astype(
                    np.float64
                ),
                h_bins=h_bins,
                s_bins=s_bins,
                v_bins=v_bins,
            )
        )

    matrix = np.asarray(
        descriptors,
        dtype=np.float64,
    )

    assert matrix.shape == (
        56,
        h_bins * s_bins * v_bins,
    )

    return matrix


def build_random_scores(
    config: dict[str, Any],
) -> np.ndarray:
    random_config = config[
        "conditions"
    ][
        "random"
    ]

    seed = int(
        random_config["seed"]
    )

    generator = np.random.Generator(
        np.random.PCG64(seed)
    )

    scores = generator.random(
        (280, 56)
    )

    assert scores.shape == (
        280,
        56,
    )

    assert format_float(
        scores[0, 0]
    ) == "0.602528611399"

    return scores


def build_color_scores(
    captions: list[dict[str, str]],
    ordered_images: list[dict[str, str]],
    baseline_config: dict[str, Any],
    experiment: dict[str, Any],
) -> np.ndarray:
    color_config = baseline_config[
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

    assert (
        h_bins,
        s_bins,
        v_bins,
    ) == (
        18,
        4,
        4,
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

    palette_descriptors = {}

    for palette_id, colors in (
        palette_mapping.items()
    ):
        palette_descriptors[
            palette_id
        ] = descriptor_from_rgb_counts(
            rgb_colors=colors,
            weights=np.ones(
                len(colors),
                dtype=np.float64,
            ),
            h_bins=h_bins,
            s_bins=s_bins,
            v_bins=v_bins,
        )

    scores = np.zeros(
        (280, 56),
        dtype=np.float64,
    )

    for query_index, caption in enumerate(
        captions
    ):
        prototype = (
            palette_descriptors[
                caption["palette_id"]
            ]
        )

        scores[
            query_index
        ] = np.minimum(
            image_descriptors,
            prototype[None, :],
        ).sum(axis=1)

    assert scores.shape == (
        280,
        56,
    )

    assert np.isfinite(scores).all()

    return scores


def evaluate_computed_condition(
    condition: str,
    scores_matrix: np.ndarray,
    captions: list[dict[str, str]],
    ordered_images: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    image_ids = [
        row["image_id"]
        for row in ordered_images
    ]

    image_row_by_id = {
        image_id: row_index
        for row_index, image_id
        in enumerate(image_ids)
    }

    query_rows = []
    ranking_rows = []
    metric_rows = []

    for query_index, caption in enumerate(
        captions
    ):
        caption_id = caption[
            "caption_id"
        ]

        image_id = caption["image_id"]

        relevant_row = (
            image_row_by_id[image_id]
        )

        relevant_image = (
            ordered_images[
                relevant_row
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
                caption[field]
                == relevant_image[field]
            )

        scores = scores_matrix[
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
                "query_index": str(
                    query_index
                ),
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
                "relevant_rank": str(
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
                    "query_index": str(
                        query_index
                    ),
                    "caption_id": (
                        caption_id
                    ),
                    "rank": str(rank),
                    "image_row_index": str(
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
    assert len(ranking_rows) == 15680
    assert len(metric_rows) == 280

    return (
        query_rows,
        ranking_rows,
        metric_rows,
    )


def reconstruct_openclip(
    captions: list[dict[str, str]],
    ordered_images: list[dict[str, str]],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    _, source_rows = load_csv(
        OPENCLIP_RESULTS_PATH
    )

    summary = load_json(
        OPENCLIP_SUMMARY_PATH
    )

    assert summary["evaluation_valid"] is True
    assert summary["experiment_id"] == "E1"

    source_rows = sorted(
        source_rows,
        key=lambda row: int(
            row["query_index"]
        ),
    )

    image_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    output_rows = []
    metric_rows = []

    for query_index, (
        caption,
        source,
    ) in enumerate(
        zip(
            captions,
            source_rows,
            strict=True,
        )
    ):
        caption_id = caption[
            "caption_id"
        ]

        assert (
            source["caption_id"]
            == caption_id
        )

        assert int(
            source["query_index"]
        ) == query_index

        image_record = image_by_id[
            caption["image_id"]
        ]

        canonical = str(
            parse_bool(
                source["is_canonical"],
                caption_id,
            )
        ).lower()

        output_rows.append(
            {
                "condition": "openclip",
                "source": (
                    "reused_validated_e1"
                ),
                "query_index": str(
                    query_index
                ),
                "caption_id": caption_id,
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

        assert (
            source["ambiguity_level"]
            == image_record[
                "ambiguity_level"
            ]
        )

        metric_rows.append(
            {
                "condition": "openclip",
                "split": source["split"],
                "is_canonical": canonical,
                "recall_at_1": float(
                    source["recall_at_1"]
                ),
                "recall_at_5": float(
                    source["recall_at_5"]
                ),
                "mrr": float(
                    source["mrr"]
                ),
                "ndcg_at_10": float(
                    source["ndcg_at_10"]
                ),
                "positive_margin": float(
                    source["positive_margin"]
                ),
            }
        )

    assert len(output_rows) == 280
    assert len(metric_rows) == 280

    return output_rows, metric_rows


def aggregate_group(
    condition: str,
    rows: list[dict[str, Any]],
    dimension: str,
    value: str,
) -> dict[str, str]:
    aggregate = aggregate_query_metrics(
        rows
    )

    return {
        "condition": condition,
        "group_dimension": dimension,
        "group_value": value,
        "query_count": str(len(rows)),
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
    metrics_by_condition: dict[
        str,
        list[dict[str, Any]],
    ],
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, float]],
]:
    rows = []
    overall = {}

    for condition in CONDITION_ORDER:
        condition_rows = (
            metrics_by_condition[
                condition
            ]
        )

        raw = aggregate_query_metrics(
            condition_rows
        )

        overall[condition] = {
            key: float(raw[key])
            for key in (
                "recall_at_1",
                "recall_at_5",
                "mrr",
                "ndcg_at_10",
                "positive_margin",
                "query_count",
            )
        }

        rows.append(
            aggregate_group(
                condition,
                condition_rows,
                "overall",
                "all",
            )
        )

        for dimension in (
            "split",
            "is_canonical",
        ):
            grouped = defaultdict(list)

            for record in condition_rows:
                grouped[
                    str(record[dimension])
                ].append(record)

            for value in sorted(grouped):
                rows.append(
                    aggregate_group(
                        condition,
                        grouped[value],
                        dimension,
                        value,
                    )
                )

    assert len(rows) == 21

    return rows, overall


def build_comparison(
    overall: dict[
        str,
        dict[str, float],
    ],
) -> list[dict[str, str]]:
    openclip = overall["openclip"]
    random = overall["random"]

    rows = []

    for condition in CONDITION_ORDER:
        metrics = overall[condition]

        rows.append(
            {
                "condition": condition,
                "query_count": str(
                    int(
                        metrics[
                            "query_count"
                        ]
                    )
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

    return rows


def validate_color_only_behavior(
    color_query_rows: list[dict[str, str]],
) -> None:
    rows_by_palette = defaultdict(list)

    for row in color_query_rows:
        rows_by_palette[
            row["palette_id"]
        ].append(row)

    assert len(rows_by_palette) == 7

    harmonic_eight = (
        sum(
            1.0 / rank
            for rank in range(1, 9)
        )
        / 8.0
    )

    expected_ndcg = (
        sum(
            1.0
            / math.log2(rank + 1.0)
            for rank in range(1, 9)
        )
        / 8.0
    )

    for palette_id, rows in (
        rows_by_palette.items()
    ):
        assert len(rows) == 40

        ranks = Counter(
            int(row["relevant_rank"])
            for row in rows
        )

        assert ranks == {
            rank: 5
            for rank in range(1, 9)
        }

        top1_ids = {
            row["top1_image_id"]
            for row in rows
        }

        assert len(top1_ids) == 1

    all_metrics = [
        {
            "recall_at_1": float(
                row["recall_at_1"]
            ),
            "recall_at_5": float(
                row["recall_at_5"]
            ),
            "mrr": float(row["mrr"]),
            "ndcg_at_10": float(
                row["ndcg_at_10"]
            ),
            "positive_margin": float(
                row["positive_margin"]
            ),
        }
        for row in color_query_rows
    ]

    aggregate = aggregate_query_metrics(
        all_metrics
    )

    assert_close(
        aggregate["recall_at_1"],
        1.0 / 8.0,
        "color.recall_at_1",
    )

    assert_close(
        aggregate["recall_at_5"],
        5.0 / 8.0,
        "color.recall_at_5",
    )

    assert_close(
        aggregate["mrr"],
        harmonic_eight,
        "color.mrr",
    )

    assert_close(
        aggregate["ndcg_at_10"],
        expected_ndcg,
        "color.ndcg_at_10",
    )


def validate_summary(
    overall: dict[
        str,
        dict[str, float],
    ],
    baseline_config: dict[str, Any],
) -> None:
    summary = load_json(
        SUMMARY_PATH
    )

    assert summary["schema_version"] == "1.0"
    assert summary["dataset_version"] == "v2"
    assert summary["experiment_id"] == "E3"

    assert summary["evaluation_valid"] is True

    assert tuple(
        summary["conditions"]
    ) == CONDITION_ORDER

    counts = summary["counts"]

    assert counts["query_rows"] == 840
    assert counts["random_ranking_rows"] == 15680
    assert counts["color_ranking_rows"] == 15680
    assert counts["aggregate_rows"] == 21
    assert counts["comparison_rows"] == 3

    for condition in CONDITION_ORDER:
        summary_metrics = (
            summary[
                "overall_metrics"
            ][condition]
        )

        assert set(summary_metrics) == {
            "recall_at_1",
            "recall_at_5",
            "mrr",
            "ndcg_at_10",
            "positive_margin",
            "query_count",
        }

        for metric, expected in (
            overall[condition].items()
        ):
            assert_close(
                float(
                    summary_metrics[
                        metric
                    ]
                ),
                expected,
                (
                    f"summary."
                    f"{condition}.{metric}"
                ),
            )

    theoretical = baseline_config[
        "theoretical_random_expectation"
    ]

    assert (
        summary[
            "theoretical_random_expectation"
        ]
        == theoretical
    )

    for metric in (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
    ):
        expected_deviation = (
            overall["random"][metric]
            - float(theoretical[metric])
        )

        assert_close(
            float(
                summary[
                    "realized_random_minus_theoretical"
                ][metric]
            ),
            expected_deviation,
            f"random_deviation.{metric}",
        )

    input_paths = {
        "baselines_config": (
            BASELINES_CONFIG_PATH
        ),
        "experiment_config": (
            EXPERIMENT_CONFIG_PATH
        ),
        "captions": CAPTIONS_PATH,
        "image_index": IMAGE_INDEX_PATH,
        "openclip_results": (
            OPENCLIP_RESULTS_PATH
        ),
        "openclip_summary": (
            OPENCLIP_SUMMARY_PATH
        ),
    }

    for key, path in input_paths.items():
        record = summary[
            "input_artifacts"
        ][key]

        assert (
            record["path"]
            == path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )

    output_paths = {
        "query_results": (
            QUERY_RESULTS_PATH
        ),
        "random_ranking": (
            RANDOM_RANKING_PATH
        ),
        "color_ranking": (
            COLOR_RANKING_PATH
        ),
        "aggregates": (
            AGGREGATES_PATH
        ),
        "comparison": (
            COMPARISON_PATH
        ),
    }

    expected_rows = {
        "query_results": 840,
        "random_ranking": 15680,
        "color_ranking": 15680,
        "aggregates": 21,
        "comparison": 3,
    }

    for key, path in output_paths.items():
        record = summary[
            "output_artifacts"
        ][key]

        assert (
            record["path"]
            == path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        )

        assert (
            record["rows"]
            == expected_rows[key]
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )


def main() -> None:
    baseline_config = load_json(
        BASELINES_CONFIG_PATH
    )

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    _, captions = load_csv(
        CAPTIONS_PATH
    )

    _, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    ordered_images = sorted(
        image_rows,
        key=lambda row: int(
            row["image_row_index"]
        ),
    )

    assert len(captions) == 280
    assert len(ordered_images) == 56

    assert [
        int(row["image_row_index"])
        for row in ordered_images
    ] == list(range(56))

    (
        openclip_query_rows,
        openclip_metric_rows,
    ) = reconstruct_openclip(
        captions,
        ordered_images,
    )

    random_scores = build_random_scores(
        baseline_config
    )

    (
        random_query_rows,
        random_ranking_rows,
        random_metric_rows,
    ) = evaluate_computed_condition(
        "random",
        random_scores,
        captions,
        ordered_images,
    )

    color_scores = build_color_scores(
        captions,
        ordered_images,
        baseline_config,
        experiment,
    )

    (
        color_query_rows,
        color_ranking_rows,
        color_metric_rows,
    ) = evaluate_computed_condition(
        "color_histogram",
        color_scores,
        captions,
        ordered_images,
    )

    validate_color_only_behavior(
        color_query_rows
    )

    expected_query_rows = (
        openclip_query_rows
        + random_query_rows
        + color_query_rows
    )

    metrics_by_condition = {
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

    (
        expected_aggregates,
        overall,
    ) = build_aggregates(
        metrics_by_condition
    )

    expected_comparison = build_comparison(
        overall
    )

    query_fields, actual_query_rows = (
        load_csv(
            QUERY_RESULTS_PATH
        )
    )

    random_fields, actual_random_ranking = (
        load_csv(
            RANDOM_RANKING_PATH
        )
    )

    color_fields, actual_color_ranking = (
        load_csv(
            COLOR_RANKING_PATH
        )
    )

    aggregate_fields, actual_aggregates = (
        load_csv(
            AGGREGATES_PATH
        )
    )

    comparison_fields, actual_comparison = (
        load_csv(
            COMPARISON_PATH
        )
    )

    assert tuple(query_fields) == (
        RESULT_FIELDS
    )

    assert tuple(random_fields) == (
        RANKING_FIELDS
    )

    assert tuple(color_fields) == (
        RANKING_FIELDS
    )

    assert tuple(aggregate_fields) == (
        AGGREGATE_FIELDS
    )

    assert tuple(comparison_fields) == (
        COMPARISON_FIELDS
    )

    assert actual_query_rows == (
        expected_query_rows
    )

    assert actual_random_ranking == (
        random_ranking_rows
    )

    assert actual_color_ranking == (
        color_ranking_rows
    )

    assert actual_aggregates == (
        expected_aggregates
    )

    assert actual_comparison == (
        expected_comparison
    )

    validate_summary(
        overall,
        baseline_config,
    )

    expected_names = {
        QUERY_RESULTS_PATH.name,
        RANDOM_RANKING_PATH.name,
        COLOR_RANKING_PATH.name,
        AGGREGATES_PATH.name,
        COMPARISON_PATH.name,
        SUMMARY_PATH.name,
    }

    actual_names = {
        path.name
        for path in OUTPUT_DIRECTORY.iterdir()
        if path.is_file()
    }

    assert actual_names == expected_names

    print("=" * 80)
    print("VALIDACIÓN INDEPENDIENTE DE E3 SUPERADA")
    print("=" * 80)
    print("Filas de consulta reconstruidas: 840")
    print("Ranking aleatorio reconstruido: 15680")
    print("Ranking cromático reconstruido: 15680")
    print("Agregados reconstruidos: 21")
    print("Comparación reconstruida: 3")
    print()

    for condition in CONDITION_ORDER:
        metrics = overall[condition]

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

    print()
    print(
        "Comportamiento cromático 1/8 y 5/8: válido"
    )
    print("Hashes de entradas y salidas: válidos")
    print("Resumen: válido")
    print("Evaluación válida: True")


if __name__ == "__main__":
    main()
