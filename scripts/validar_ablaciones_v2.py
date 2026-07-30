"""Valida independientemente los resultados de las ablaciones E4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "ablaciones_v2"
)

RAW_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "resultados_consulta_estructural_ablaciones_v2.csv"
)

RANKING_PATH = (
    OUTPUT_DIRECTORY
    / "ranking_estructural_ablaciones_v2.csv"
)

GROUP_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "resultados_grupo_estructural_ablaciones_v2.csv"
)

STRUCTURAL_AGGREGATES_PATH = (
    OUTPUT_DIRECTORY
    / "agregados_estructurales_ablaciones_v2.csv"
)

PAIRED_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "comparaciones_pareadas_estructurales_v2.csv"
)

PAIRED_AGGREGATES_PATH = (
    OUTPUT_DIRECTORY
    / "agregados_comparaciones_estructurales_v2.csv"
)

EXACT_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "resultados_exactos_ablaciones_v2.csv"
)

EXACT_AGGREGATES_PATH = (
    OUTPUT_DIRECTORY
    / "agregados_exactos_ablaciones_v2.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "resumen_ablaciones_v2.json"
)

CONDITION_ORDER = (
    "original_image_full_caption",
    "grayscale_image_full_caption",
    "original_image_caption_without_color",
    "grayscale_image_caption_without_color",
)

EXACT_CONDITION_ORDER = (
    "original_image_full_caption",
    "grayscale_image_full_caption",
)

STRUCTURAL_METRICS = (
    "structural_hit_at_1",
    "structural_hit_at_5",
    "structural_fractional_recall_at_5",
    "structural_mrr",
    "structural_ndcg_at_10",
    "best_relevant_margin",
)

EXACT_METRICS = (
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
    )

    assert b"\r\n" not in raw

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


def split_ids(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def assert_close(
    actual: float,
    expected: float,
    context: str,
    tolerance: float = 1e-9,
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


def assert_float_field(
    row: dict[str, str],
    field: str,
    expected: float,
    context: str,
) -> None:
    assert_close(
        float(row[field]),
        expected,
        f"{context}.{field}",
    )


def rank_scores(
    scores: np.ndarray,
    image_ids: list[str],
) -> list[int]:
    assert scores.shape == (
        len(image_ids),
    )

    assert np.isfinite(scores).all()

    return sorted(
        range(scores.size),
        key=lambda index: (
            -float(scores[index]),
            image_ids[index],
        ),
    )


def structural_metrics(
    scores: np.ndarray,
    relevant_indices: set[int],
    image_ids: list[str],
) -> dict[str, Any]:
    ranking = rank_scores(
        scores,
        image_ids,
    )

    relevant_ranks = [
        rank
        for rank, index in enumerate(
            ranking,
            start=1,
        )
        if index in relevant_indices
    ]

    assert len(relevant_ranks) == 7

    first_rank = min(
        relevant_ranks
    )

    relevant_in_top_5 = sum(
        index in relevant_indices
        for index in ranking[:5]
    )

    dcg_at_10 = sum(
        1.0 / math.log2(rank + 1.0)
        for rank in relevant_ranks
        if rank <= 10
    )

    ideal_dcg_at_10 = sum(
        1.0 / math.log2(rank + 1.0)
        for rank in range(1, 8)
    )

    relevant_scores = [
        float(scores[index])
        for index in relevant_indices
    ]

    nonrelevant_scores = [
        float(scores[index])
        for index in range(
            scores.size
        )
        if index not in relevant_indices
    ]

    best_relevant_score = max(
        relevant_scores
    )

    best_nonrelevant_score = max(
        nonrelevant_scores
    )

    return {
        "ranking": ranking,
        "first_relevant_rank": first_rank,
        "structural_hit_at_1": float(
            first_rank <= 1
        ),
        "structural_hit_at_5": float(
            first_rank <= 5
        ),
        "structural_fractional_recall_at_5": (
            relevant_in_top_5
            / len(relevant_indices)
        ),
        "structural_mrr": (
            1.0 / first_rank
        ),
        "structural_ndcg_at_10": (
            dcg_at_10
            / ideal_dcg_at_10
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


def exact_metrics(
    scores: np.ndarray,
    relevant_index: int,
    image_ids: list[str],
) -> dict[str, Any]:
    ranking = rank_scores(
        scores,
        image_ids,
    )

    relevant_rank = (
        ranking.index(
            relevant_index
        )
        + 1
    )

    relevant_score = float(
        scores[relevant_index]
    )

    nonrelevant_score = max(
        float(scores[index])
        for index in range(
            scores.size
        )
        if index != relevant_index
    )

    return {
        "ranking": ranking,
        "relevant_rank": relevant_rank,
        "recall_at_1": float(
            relevant_rank <= 1
        ),
        "recall_at_5": float(
            relevant_rank <= 5
        ),
        "mrr": 1.0 / relevant_rank,
        "ndcg_at_10": (
            1.0
            / math.log2(
                relevant_rank + 1.0
            )
            if relevant_rank <= 10
            else 0.0
        ),
        "relevant_score": (
            relevant_score
        ),
        "positive_margin": (
            relevant_score
            - nonrelevant_score
        ),
    }


def mean_metric(
    rows: Iterable[dict[str, str]],
    metric: str,
) -> float:
    values = [
        float(row[metric])
        for row in rows
    ]

    assert values

    return float(
        np.mean(values)
    )


def determine_regime(
    relevant_ids: set[str],
    image_by_id: dict[str, dict[str, str]],
) -> str:
    composition = Counter(
        image_by_id[
            image_id
        ]["split"]
        for image_id in relevant_ids
    )

    if composition == {
        "id": 5,
        "ood_palette": 2,
    }:
        return "base_pattern"

    if composition == {
        "ood_pattern": 5,
        "ood_both": 2,
    }:
        return "heldout_pattern"

    raise AssertionError(
        f"Composición inesperada: "
        f"{dict(composition)}"
    )


def validate_mean_fields(
    row: dict[str, str],
    source_rows: list[dict[str, str]],
    metrics: tuple[str, ...],
    context: str,
) -> None:
    for metric in metrics:
        assert_float_field(
            row,
            metric,
            mean_metric(
                source_rows,
                metric,
            ),
            context,
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

    raw_fields, raw_rows = load_csv(
        RAW_RESULTS_PATH
    )

    ranking_fields, ranking_rows = load_csv(
        RANKING_PATH
    )

    group_fields, group_rows = load_csv(
        GROUP_RESULTS_PATH
    )

    aggregate_fields, aggregate_rows = (
        load_csv(
            STRUCTURAL_AGGREGATES_PATH
        )
    )

    paired_fields, paired_rows = load_csv(
        PAIRED_RESULTS_PATH
    )

    (
        paired_aggregate_fields,
        paired_aggregate_rows,
    ) = load_csv(
        PAIRED_AGGREGATES_PATH
    )

    exact_fields, exact_rows = load_csv(
        EXACT_RESULTS_PATH
    )

    (
        exact_aggregate_fields,
        exact_aggregate_rows,
    ) = load_csv(
        EXACT_AGGREGATES_PATH
    )

    summary = load_json(
        SUMMARY_PATH
    )

    assert tuple(raw_fields) == RAW_FIELDS
    assert tuple(ranking_fields) == RANKING_FIELDS
    assert tuple(group_fields) == GROUP_FIELDS

    assert tuple(
        aggregate_fields
    ) == STRUCTURAL_AGGREGATE_FIELDS

    assert tuple(
        paired_fields
    ) == PAIRED_FIELDS

    assert tuple(
        paired_aggregate_fields
    ) == PAIRED_AGGREGATE_FIELDS

    assert tuple(exact_fields) == EXACT_FIELDS

    assert tuple(
        exact_aggregate_fields
    ) == EXACT_AGGREGATE_FIELDS

    assert len(full_rows) == 280
    assert len(colorless_rows) == 40
    assert len(image_rows) == 56
    assert len(usage_rows) == 600
    assert len(e1_rows) == 280

    assert len(raw_rows) == 640
    assert len(ranking_rows) == 35840
    assert len(group_rows) == 160
    assert len(aggregate_rows) == 40
    assert len(paired_rows) == 160
    assert len(paired_aggregate_rows) == 12
    assert len(exact_rows) == 560
    assert len(exact_aggregate_rows) == 14

    ordered_images = sorted(
        image_rows,
        key=lambda row: int(
            row["image_row_index"]
        ),
    )

    image_ids = [
        row["image_id"]
        for row in ordered_images
    ]

    assert [
        int(row["image_row_index"])
        for row in ordered_images
    ] == list(range(56))

    image_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    image_row_by_id = {
        image_id: index
        for index, image_id
        in enumerate(image_ids)
    }

    usage_map = {
        (
            row["source_id"],
            row["record_id"],
        ): int(
            row["text_row_index"]
        )
        for row in usage_rows
    }

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

    group_ids = [
        row["colorless_caption_id"]
        for row in colorless_rows
    ]

    group_index_by_id = {
        group_id: index
        for index, group_id
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

    group_query_index_by_caption = {}

    for group_id in group_ids:
        for index, row in enumerate(
            full_by_group[group_id]
        ):
            group_query_index_by_caption[
                row["caption_id"]
            ] = index

    e1_by_caption = {
        row["caption_id"]: row
        for row in e1_rows
    }

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

    ranking_by_raw: dict[
        int,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in ranking_rows:
        ranking_by_raw[
            int(row["raw_row_index"])
        ].append(row)

    assert len(ranking_by_raw) == 640

    raw_by_condition_group: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for expected_raw_index, row in enumerate(
        raw_rows
    ):
        context = (
            f"raw[{expected_raw_index}]"
        )

        assert int(
            row["raw_row_index"]
        ) == expected_raw_index

        condition = row["condition"]

        assert condition in CONDITION_ORDER

        specification = condition_specs[
            condition
        ]

        condition_query_index = int(
            row["condition_query_index"]
        )

        query = specification[
            "queries"
        ][condition_query_index]

        if (
            specification["text_variant"]
            == "full_caption"
        ):
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

        group_index = group_index_by_id[
            group_id
        ]

        assert row["query_id"] == query_id

        assert (
            row["colorless_caption_id"]
            == group_id
        )

        assert int(
            row["group_index"]
        ) == group_index

        assert int(
            row["group_query_index"]
        ) == group_query_index

        assert (
            row["image_variant"]
            == specification[
                "image_variant"
            ]
        )

        assert (
            row["text_variant"]
            == specification[
                "text_variant"
            ]
        )

        assert (
            row["source_id"]
            == specification[
                "source_id"
            ]
        )

        text_row_index = usage_map[
            (
                specification[
                    "source_id"
                ],
                query_id,
            )
        ]

        assert int(
            row["text_row_index"]
        ) == text_row_index

        relevant_ids = set(
            split_ids(
                colorless[
                    "relevant_image_ids"
                ]
            )
        )

        relevant_indices = {
            image_row_by_id[
                image_id
            ]
            for image_id in relevant_ids
        }

        regime = determine_regime(
            relevant_ids,
            image_by_id,
        )

        assert row["regime"] == regime

        for field in (
            "structure_id",
            "template_id",
            "pattern_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
        ):
            assert (
                row[field]
                == colorless[field]
            )

        assert (
            row["is_canonical"]
            == colorless[
                "is_canonical"
            ].strip().lower()
        )

        assert int(
            row["relevant_count"]
        ) == 7

        image_matrix = (
            image_matrix_by_variant[
                specification[
                    "image_variant"
                ]
            ]
        )

        scores = np.asarray(
            text_embeddings[
                text_row_index
            ]
            @ image_matrix.T,
            dtype=np.float64,
        )

        metrics = structural_metrics(
            scores,
            relevant_indices,
            image_ids,
        )

        assert int(
            row["first_relevant_rank"]
        ) == metrics[
            "first_relevant_rank"
        ]

        top1_index = metrics[
            "ranking"
        ][0]

        assert (
            row["top1_image_id"]
            == image_ids[top1_index]
        )

        assert_float_field(
            row,
            "top1_score",
            float(scores[top1_index]),
            context,
        )

        for field in (
            "best_relevant_score",
            "best_nonrelevant_score",
            "best_relevant_margin",
            "structural_hit_at_1",
            "structural_hit_at_5",
            "structural_fractional_recall_at_5",
            "structural_mrr",
            "structural_ndcg_at_10",
        ):
            assert_float_field(
                row,
                field,
                float(metrics[field]),
                context,
            )

        actual_ranking = sorted(
            ranking_by_raw[
                expected_raw_index
            ],
            key=lambda item: int(
                item["rank"]
            ),
        )

        assert len(actual_ranking) == 56

        for rank, (
            ranking_row,
            image_index,
        ) in enumerate(
            zip(
                actual_ranking,
                metrics["ranking"],
                strict=True,
            ),
            start=1,
        ):
            image_record = (
                ordered_images[
                    image_index
                ]
            )

            assert int(
                ranking_row["rank"]
            ) == rank

            assert (
                ranking_row["condition"]
                == condition
            )

            assert (
                ranking_row["query_id"]
                == query_id
            )

            assert int(
                ranking_row[
                    "image_row_index"
                ]
            ) == image_index

            assert (
                ranking_row["image_id"]
                == image_record[
                    "image_id"
                ]
            )

            assert_float_field(
                ranking_row,
                "score",
                float(
                    scores[image_index]
                ),
                (
                    f"{context}.ranking"
                    f"[{rank}]"
                ),
            )

            assert (
                ranking_row[
                    "is_relevant"
                ]
                == str(
                    image_index
                    in relevant_indices
                ).lower()
            )

        raw_by_condition_group[
            (
                condition,
                group_id,
            )
        ].append(row)

    assert len(
        raw_by_condition_group
    ) == 160

    group_map = {}

    for row_index, row in enumerate(
        group_rows
    ):
        condition_index = (
            row_index // 40
        )

        expected_group_index = (
            row_index % 40
        )

        condition = CONDITION_ORDER[
            condition_index
        ]

        group_id = group_ids[
            expected_group_index
        ]

        colorless = colorless_by_id[
            group_id
        ]

        specification = condition_specs[
            condition
        ]

        source_rows = (
            raw_by_condition_group[
                (
                    condition,
                    group_id,
                )
            ]
        )

        expected_count = (
            7
            if (
                specification[
                    "text_variant"
                ]
                == "full_caption"
            )
            else 1
        )

        assert row["condition"] == condition

        assert int(
            row["group_index"]
        ) == expected_group_index

        assert (
            row["colorless_caption_id"]
            == group_id
        )

        assert int(
            row["raw_query_count"]
        ) == expected_count

        assert len(
            source_rows
        ) == expected_count

        relevant_ids = set(
            split_ids(
                colorless[
                    "relevant_image_ids"
                ]
            )
        )

        assert (
            row["regime"]
            == determine_regime(
                relevant_ids,
                image_by_id,
            )
        )

        validate_mean_fields(
            row,
            source_rows,
            STRUCTURAL_METRICS,
            (
                f"group[{condition}]"
                f"[{group_id}]"
            ),
        )

        group_map[
            (
                condition,
                group_id,
            )
        ] = row

    assert len(group_map) == 160

    aggregate_map = {
        (
            row["condition"],
            row["group_dimension"],
            row["group_value"],
        ): row
        for row in aggregate_rows
    }

    assert len(aggregate_map) == 40

    for condition in CONDITION_ORDER:
        condition_groups = [
            group_map[
                (
                    condition,
                    group_id,
                )
            ]
            for group_id in group_ids
        ]

        expected_groups = {
            (
                "overall",
                "all",
            ): condition_groups
        }

        for dimension in (
            "regime",
            "template_id",
            "is_canonical",
        ):
            grouped = defaultdict(list)

            for row in condition_groups:
                grouped[
                    row[dimension]
                ].append(row)

            for value, rows in grouped.items():
                expected_groups[
                    (
                        dimension,
                        value,
                    )
                ] = rows

        assert len(
            expected_groups
        ) == 10

        for (
            dimension,
            value,
        ), rows in expected_groups.items():
            aggregate = aggregate_map[
                (
                    condition,
                    dimension,
                    value,
                )
            ]

            assert int(
                aggregate["group_count"]
            ) == len(rows)

            validate_mean_fields(
                aggregate,
                rows,
                STRUCTURAL_METRICS,
                (
                    f"aggregate."
                    f"{condition}."
                    f"{dimension}."
                    f"{value}"
                ),
            )

    comparison_by_id = {
        row["comparison_id"]: row
        for row in contract[
            "paired_comparisons"
        ]
    }

    assert len(comparison_by_id) == 4

    paired_by_comparison = defaultdict(list)

    for row in paired_rows:
        comparison = comparison_by_id[
            row["comparison_id"]
        ]

        group_id = row[
            "colorless_caption_id"
        ]

        minuend = group_map[
            (
                comparison["minuend"],
                group_id,
            )
        ]

        subtrahend = group_map[
            (
                comparison[
                    "subtrahend"
                ],
                group_id,
            )
        ]

        assert (
            row["minuend"]
            == comparison["minuend"]
        )

        assert (
            row["subtrahend"]
            == comparison[
                "subtrahend"
            ]
        )

        for metric in STRUCTURAL_METRICS:
            assert_float_field(
                row,
                f"delta_{metric}",
                (
                    float(
                        minuend[metric]
                    )
                    - float(
                        subtrahend[
                            metric
                        ]
                    )
                ),
                (
                    f"paired."
                    f"{row['comparison_id']}."
                    f"{group_id}"
                ),
            )

        paired_by_comparison[
            row["comparison_id"]
        ].append(row)

    assert {
        key: len(value)
        for key, value
        in paired_by_comparison.items()
    } == {
        comparison_id: 40
        for comparison_id
        in comparison_by_id
    }

    paired_aggregate_map = {
        (
            row["comparison_id"],
            row["group_dimension"],
            row["group_value"],
        ): row
        for row in paired_aggregate_rows
    }

    assert len(
        paired_aggregate_map
    ) == 12

    delta_metrics = tuple(
        f"delta_{metric}"
        for metric
        in STRUCTURAL_METRICS
    )

    for comparison_id, rows in (
        paired_by_comparison.items()
    ):
        expected_groups = {
            (
                "overall",
                "all",
            ): rows
        }

        by_regime = defaultdict(list)

        for row in rows:
            by_regime[
                row["regime"]
            ].append(row)

        for regime, regime_rows in (
            by_regime.items()
        ):
            expected_groups[
                (
                    "regime",
                    regime,
                )
            ] = regime_rows

        for (
            dimension,
            value,
        ), source_rows in (
            expected_groups.items()
        ):
            aggregate = (
                paired_aggregate_map[
                    (
                        comparison_id,
                        dimension,
                        value,
                    )
                ]
            )

            assert int(
                aggregate["group_count"]
            ) == len(source_rows)

            validate_mean_fields(
                aggregate,
                source_rows,
                delta_metrics,
                (
                    f"paired_aggregate."
                    f"{comparison_id}."
                    f"{dimension}."
                    f"{value}"
                ),
            )

    exact_by_condition = defaultdict(list)

    for row_index, row in enumerate(
        exact_rows
    ):
        condition_index = (
            row_index // 280
        )

        query_index = (
            row_index % 280
        )

        condition = (
            EXACT_CONDITION_ORDER[
                condition_index
            ]
        )

        caption = full_rows[
            query_index
        ]

        caption_id = caption[
            "caption_id"
        ]

        text_row_index = usage_map[
            (
                "positivos",
                caption_id,
            )
        ]

        relevant_index = (
            image_row_by_id[
                caption["image_id"]
            ]
        )

        image_variant = (
            "original"
            if condition
            == "original_image_full_caption"
            else "grayscale"
        )

        scores = np.asarray(
            text_embeddings[
                text_row_index
            ]
            @ image_matrix_by_variant[
                image_variant
            ].T,
            dtype=np.float64,
        )

        metrics = exact_metrics(
            scores,
            relevant_index,
            image_ids,
        )

        assert row["condition"] == condition

        assert int(
            row["query_index"]
        ) == query_index

        assert (
            row["caption_id"]
            == caption_id
        )

        assert (
            row["image_id"]
            == caption["image_id"]
        )

        assert int(
            row["relevant_rank"]
        ) == metrics[
            "relevant_rank"
        ]

        top1_index = metrics[
            "ranking"
        ][0]

        assert (
            row["top1_image_id"]
            == image_ids[top1_index]
        )

        assert_float_field(
            row,
            "top1_score",
            float(scores[top1_index]),
            (
                f"exact.{condition}."
                f"{caption_id}"
            ),
        )

        assert_float_field(
            row,
            "relevant_score",
            metrics["relevant_score"],
            (
                f"exact.{condition}."
                f"{caption_id}"
            ),
        )

        for metric in EXACT_METRICS:
            assert_float_field(
                row,
                metric,
                float(metrics[metric]),
                (
                    f"exact.{condition}."
                    f"{caption_id}"
                ),
            )

        if condition == (
            "original_image_full_caption"
        ):
            e1 = e1_by_caption[
                caption_id
            ]

            assert int(
                e1["relevant_rank"]
            ) == metrics[
                "relevant_rank"
            ]

            assert (
                e1["top1_image_id"]
                == image_ids[top1_index]
            )

            for metric in EXACT_METRICS:
                assert_close(
                    float(e1[metric]),
                    float(metrics[metric]),
                    (
                        f"E1.{caption_id}."
                        f"{metric}"
                    ),
                    tolerance=1e-7,
                )

        exact_by_condition[
            condition
        ].append(row)

    assert {
        key: len(value)
        for key, value
        in exact_by_condition.items()
    } == {
        condition: 280
        for condition
        in EXACT_CONDITION_ORDER
    }

    exact_aggregate_map = {
        (
            row["condition"],
            row["group_dimension"],
            row["group_value"],
        ): row
        for row in exact_aggregate_rows
    }

    assert len(
        exact_aggregate_map
    ) == 14

    for condition in EXACT_CONDITION_ORDER:
        rows = exact_by_condition[
            condition
        ]

        expected_groups = {
            (
                "overall",
                "all",
            ): rows
        }

        for dimension in (
            "split",
            "is_canonical",
        ):
            grouped = defaultdict(list)

            for row in rows:
                grouped[
                    row[dimension]
                ].append(row)

            for value, source_rows in (
                grouped.items()
            ):
                expected_groups[
                    (
                        dimension,
                        value,
                    )
                ] = source_rows

        assert len(
            expected_groups
        ) == 7

        for (
            dimension,
            value,
        ), source_rows in (
            expected_groups.items()
        ):
            aggregate = exact_aggregate_map[
                (
                    condition,
                    dimension,
                    value,
                )
            ]

            assert int(
                aggregate["query_count"]
            ) == len(source_rows)

            validate_mean_fields(
                aggregate,
                source_rows,
                EXACT_METRICS,
                (
                    f"exact_aggregate."
                    f"{condition}."
                    f"{dimension}."
                    f"{value}"
                ),
            )

    assert summary["evaluation_valid"] is True

    assert summary["experiment_id"] == "E4"

    assert summary["counts"] == {
        "raw_structural_results": 640,
        "structural_ranking_rows": 35840,
        "structural_group_results": 160,
        "structural_aggregate_rows": 40,
        "paired_group_rows": 160,
        "paired_aggregate_rows": 12,
        "exact_results": 560,
        "exact_aggregate_rows": 14,
    }

    for condition in CONDITION_ORDER:
        overall = aggregate_map[
            (
                condition,
                "overall",
                "all",
            )
        ]

        summary_metrics = summary[
            "structural_overall_metrics"
        ][condition]

        assert (
            summary_metrics[
                "group_count"
            ]
            == 40
        )

        for metric in STRUCTURAL_METRICS:
            assert_close(
                float(
                    summary_metrics[
                        metric
                    ]
                ),
                float(
                    overall[metric]
                ),
                (
                    f"summary.structural."
                    f"{condition}.{metric}"
                ),
            )

    for condition in EXACT_CONDITION_ORDER:
        overall = exact_aggregate_map[
            (
                condition,
                "overall",
                "all",
            )
        ]

        summary_metrics = summary[
            "exact_overall_metrics"
        ][condition]

        assert (
            summary_metrics[
                "query_count"
            ]
            == 280
        )

        for metric in EXACT_METRICS:
            assert_close(
                float(
                    summary_metrics[
                        metric
                    ]
                ),
                float(
                    overall[metric]
                ),
                (
                    f"summary.exact."
                    f"{condition}.{metric}"
                ),
            )

    input_paths = {
        "contract": CONFIG_PATH,
        "full_captions": (
            FULL_CAPTIONS_PATH
        ),
        "colorless_captions": (
            COLORLESS_CAPTIONS_PATH
        ),
        "image_index": (
            IMAGE_INDEX_PATH
        ),
        "text_usage": (
            TEXT_USAGE_PATH
        ),
        "original_image_embeddings": (
            ORIGINAL_IMAGE_EMBEDDINGS_PATH
        ),
        "grayscale_image_embeddings": (
            GRAYSCALE_IMAGE_EMBEDDINGS_PATH
        ),
        "text_embeddings": (
            TEXT_EMBEDDINGS_PATH
        ),
        "e1_results": (
            E1_RESULTS_PATH
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
        "raw_results": RAW_RESULTS_PATH,
        "ranking": RANKING_PATH,
        "group_results": (
            GROUP_RESULTS_PATH
        ),
        "structural_aggregates": (
            STRUCTURAL_AGGREGATES_PATH
        ),
        "paired_results": (
            PAIRED_RESULTS_PATH
        ),
        "paired_aggregates": (
            PAIRED_AGGREGATES_PATH
        ),
        "exact_results": (
            EXACT_RESULTS_PATH
        ),
        "exact_aggregates": (
            EXACT_AGGREGATES_PATH
        ),
    }

    expected_output_rows = {
        "raw_results": 640,
        "ranking": 35840,
        "group_results": 160,
        "structural_aggregates": 40,
        "paired_results": 160,
        "paired_aggregates": 12,
        "exact_results": 560,
        "exact_aggregates": 14,
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
            == expected_output_rows[key]
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )

    expected_names = {
        path.name
        for path in (
            RAW_RESULTS_PATH,
            RANKING_PATH,
            GROUP_RESULTS_PATH,
            STRUCTURAL_AGGREGATES_PATH,
            PAIRED_RESULTS_PATH,
            PAIRED_AGGREGATES_PATH,
            EXACT_RESULTS_PATH,
            EXACT_AGGREGATES_PATH,
            SUMMARY_PATH,
        )
    }

    actual_names = {
        path.name
        for path in OUTPUT_DIRECTORY.iterdir()
        if path.is_file()
    }

    assert actual_names == expected_names

    print("=" * 88)
    print("VALIDACIÓN INDEPENDIENTE DE E4 SUPERADA")
    print("=" * 88)
    print("Consultas estructurales verificadas: 640")
    print("Filas de ranking verificadas: 35840")
    print("Grupos estructurales verificados: 160")
    print("Comparaciones pareadas verificadas: 160")
    print("Consultas exactas verificadas: 560")
    print("Ancla E1 reconstruida: 280 / 280")
    print()

    print("Métricas estructurales:")

    for condition in CONDITION_ORDER:
        metrics = summary[
            "structural_overall_metrics"
        ][condition]

        print()
        print(condition)
        print(
            "- Hit@1:",
            format(
                metrics[
                    "structural_hit_at_1"
                ],
                ".12f",
            ),
        )
        print(
            "- Hit@5:",
            format(
                metrics[
                    "structural_hit_at_5"
                ],
                ".12f",
            ),
        )
        print(
            "- Fractional Recall@5:",
            format(
                metrics[
                    "structural_fractional_recall_at_5"
                ],
                ".12f",
            ),
        )
        print(
            "- MRR:",
            format(
                metrics[
                    "structural_mrr"
                ],
                ".12f",
            ),
        )
        print(
            "- nDCG@10:",
            format(
                metrics[
                    "structural_ndcg_at_10"
                ],
                ".12f",
            ),
        )

    print()
    print("Hashes de entradas y salidas: válidos")
    print("Resumen: válido")
    print("Evaluación válida: True")


if __name__ == "__main__":
    main()
