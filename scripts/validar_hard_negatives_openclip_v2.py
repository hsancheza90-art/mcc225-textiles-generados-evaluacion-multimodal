"""Valida independientemente la evaluación OpenCLIP con negativos difíciles v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from metricas_retrieval_v2 import (
    aggregate_query_metrics,
    evaluate_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "experimento_v2.json"
)

QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "consultas_negativos_dificiles_v2.csv"
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "candidatos_negativos_dificiles_v2.csv"
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

TEXT_INDEX_PATH = (
    EMBEDDINGS_DIRECTORY
    / "index_textos_unicos_v2.csv"
)

TEXT_USAGE_PATH = (
    EMBEDDINGS_DIRECTORY
    / "usos_textos_v2.csv"
)

IMAGE_MATRIX_PATH = (
    EMBEDDINGS_DIRECTORY
    / "embeddings_imagen_original_v2.npy"
)

TEXT_MATRIX_PATH = (
    EMBEDDINGS_DIRECTORY
    / "embeddings_textos_unicos_v2.npy"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "hard_negatives_openclip_v2"
)

QUERY_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "resultados_consulta_hard_negatives_openclip_v2.csv"
)

RANKING_PATH = (
    OUTPUT_DIRECTORY
    / "ranking_hard_negatives_openclip_v2.csv"
)

PAIRWISE_PATH = (
    OUTPUT_DIRECTORY
    / "comparaciones_pareadas_hard_negatives_openclip_v2.csv"
)

AGGREGATES_PATH = (
    OUTPUT_DIRECTORY
    / "agregados_hard_negatives_openclip_v2.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "resumen_hard_negatives_openclip_v2.json"
)


QUERY_RESULT_FIELDS = (
    "query_index",
    "query_id",
    "image_id",
    "semantic_id",
    "split",
    "pattern_id",
    "palette_id",
    "ambiguity_level",
    "template_id",
    "omitted_attribute",
    "image_row_index",
    "positive_candidate_id",
    "positive_position_source",
    "positive_rank",
    "top1_candidate_id",
    "top1_candidate_role",
    "top1_score",
    "positive_score",
    "positive_margin",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
    "hard_negative_accuracy",
)

RANKING_FIELDS = (
    "query_index",
    "query_id",
    "rank",
    "candidate_id",
    "candidate_position",
    "text_row_index",
    "score",
    "is_positive",
    "relevance_label",
    "candidate_role",
    "changed_attribute",
    "negative_type",
    "is_global_positive_text",
    "is_negative_global_overlap",
)

PAIRWISE_FIELDS = (
    "query_index",
    "query_id",
    "image_id",
    "split",
    "ambiguity_level",
    "positive_candidate_id",
    "negative_candidate_id",
    "changed_attribute",
    "negative_type",
    "positive_score",
    "negative_score",
    "paired_difference",
    "positive_above_negative",
)

AGGREGATE_FIELDS = (
    "group_dimension",
    "group_value",
    "query_count",
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
    "positive_margin",
    "hard_negative_accuracy",
)


def load_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Carga un CSV UTF-8 sin BOM y con finales LF."""

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
    """Carga un JSON UTF-8 sin BOM y con finales LF."""

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
    """Calcula el SHA-256 de un archivo."""

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
    """Interpreta los booleanos textuales de los CSV."""

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
    """Serializa un número real con precisión estable."""

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
    """Compara números reales con tolerancia absoluta."""

    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ), (
        f"{context}: actual={actual}, "
        f"esperado={expected}."
    )


def aggregate_group(
    rows: list[dict[str, Any]],
    dimension: str,
    value: str,
) -> dict[str, str]:
    """Reconstruye una fila agregada."""

    aggregate = aggregate_query_metrics(
        rows
    )

    return {
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
        "hard_negative_accuracy": (
            format_float(
                aggregate[
                    "hard_negative_accuracy"
                ]
            )
        ),
    }


def reconstruct() -> dict[str, Any]:
    """Reconstruye E2 desde las fuentes y embeddings."""

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    experiments = {
        record["experiment_id"]: record
        for record in experiment["experiments"]
    }

    assert experiments["E2"] == {
        "experiment_id": "E2",
        "name": "hard_negative_forced_choice",
        "dataset": "v2",
    }

    chance_level = float(
        experiment[
            "metrics"
        ][
            "hard_negative_chance_level"
        ]
    )

    assert chance_level == 0.2

    _, queries = load_csv(
        QUERIES_PATH
    )

    _, candidates = load_csv(
        CANDIDATES_PATH
    )

    _, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    _, text_rows = load_csv(
        TEXT_INDEX_PATH
    )

    _, usage_rows = load_csv(
        TEXT_USAGE_PATH
    )

    image_embeddings = np.load(
        IMAGE_MATRIX_PATH,
        allow_pickle=False,
    )

    text_embeddings = np.load(
        TEXT_MATRIX_PATH,
        allow_pickle=False,
    )

    assert len(queries) == 56
    assert len(candidates) == 280
    assert len(image_rows) == 56
    assert len(text_rows) == 494
    assert len(usage_rows) == 600

    assert image_embeddings.shape == (
        56,
        512,
    )

    assert text_embeddings.shape == (
        494,
        512,
    )

    assert image_embeddings.dtype == np.float32
    assert text_embeddings.dtype == np.float32

    assert np.isfinite(
        image_embeddings
    ).all()

    assert np.isfinite(
        text_embeddings
    ).all()

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

    image_row_by_id = {
        row["image_id"]: int(
            row["image_row_index"]
        )
        for row in ordered_images
    }

    image_record_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    text_hash_by_row = {
        int(row["text_row_index"]): (
            row["text_sha256"]
        )
        for row in text_rows
    }

    usage_map = {
        (
            row["source_id"],
            row["record_id"],
        ): int(row["text_row_index"])
        for row in usage_rows
    }

    assert len(usage_map) == 600

    candidates_by_query: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for candidate in candidates:
        candidates_by_query[
            candidate["query_id"]
        ].append(candidate)

    ordered_queries = sorted(
        queries,
        key=lambda row: int(
            row["query_index"]
        ),
    )

    assert [
        int(row["query_index"])
        for row in ordered_queries
    ] == list(range(56))

    expected_query_rows: list[
        dict[str, str]
    ] = []

    expected_ranking_rows: list[
        dict[str, str]
    ] = []

    expected_pairwise_rows: list[
        dict[str, str]
    ] = []

    metric_rows: list[
        dict[str, Any]
    ] = []

    rank_distribution: Counter[int] = (
        Counter()
    )

    for query in ordered_queries:
        query_index = int(
            query["query_index"]
        )

        query_id = query["query_id"]
        image_id = query["image_id"]

        assert image_id in image_row_by_id

        image_row_index = (
            image_row_by_id[image_id]
        )

        image_record = (
            image_record_by_id[image_id]
        )

        for field in (
            "semantic_id",
            "split",
            "pattern_id",
            "palette_id",
            "ambiguity_level",
        ):
            assert (
                query[field]
                == image_record[field]
            )

        query_candidates = sorted(
            candidates_by_query[
                query_id
            ],
            key=lambda row: int(
                row["candidate_position"]
            ),
        )

        assert len(query_candidates) == 5

        positions = [
            int(row["candidate_position"])
            for row in query_candidates
        ]

        assert positions == [
            1,
            2,
            3,
            4,
            5,
        ]

        assert (
            query["candidate_ids"].split("|")
            == [
                row["candidate_id"]
                for row in query_candidates
            ]
        )

        positive_indices = [
            index
            for index, candidate
            in enumerate(query_candidates)
            if parse_bool(
                candidate["is_positive"],
                candidate["candidate_id"],
            )
        ]

        assert len(positive_indices) == 1

        positive_index = positive_indices[0]

        positive_candidate = (
            query_candidates[
                positive_index
            ]
        )

        assert (
            positive_candidate[
                "candidate_id"
            ]
            == query[
                "positive_candidate_id"
            ]
        )

        assert int(
            positive_candidate[
                "candidate_position"
            ]
        ) == int(
            query["positive_position"]
        )

        assert (
            positive_candidate[
                "candidate_text"
            ]
            == query[
                "positive_caption_text"
            ]
        )

        candidate_text_rows: list[int] = []

        for candidate in query_candidates:
            candidate_id = candidate[
                "candidate_id"
            ]

            assert (
                candidate["query_id"]
                == query_id
            )

            assert (
                candidate["image_id"]
                == image_id
            )

            assert (
                candidate["semantic_id"]
                == query["semantic_id"]
            )

            assert (
                candidate["split"]
                == query["split"]
            )

            assert (
                candidate[
                    "ambiguity_level"
                ]
                == query[
                    "ambiguity_level"
                ]
            )

            usage_key = (
                "negativos_dificiles",
                candidate_id,
            )

            assert usage_key in usage_map

            text_row_index = (
                usage_map[usage_key]
            )

            assert (
                text_hash_by_row[
                    text_row_index
                ]
                == candidate[
                    "candidate_sha256"
                ]
            )

            candidate_text_rows.append(
                text_row_index
            )

        candidate_matrix = (
            text_embeddings[
                candidate_text_rows
            ]
        )

        image_vector = image_embeddings[
            image_row_index
        ]

        scores = (
            candidate_matrix
            @ image_vector
        )

        assert scores.shape == (5,)
        assert np.isfinite(scores).all()

        candidate_keys = tuple(
            row["candidate_id"]
            for row in query_candidates
        )

        result = evaluate_query(
            scores=scores,
            relevant_indices={
                positive_index
            },
            candidate_keys=candidate_keys,
        )

        ranking = result[
            "ranking_indices"
        ]

        top1_index = int(
            ranking[0]
        )

        top1_candidate = (
            query_candidates[
                top1_index
            ]
        )

        positive_score = float(
            scores[positive_index]
        )

        rank_distribution[
            int(
                result[
                    "first_relevant_rank"
                ]
            )
        ] += 1

        metric_rows.append(
            {
                "split": query["split"],
                "ambiguity_level": (
                    query[
                        "ambiguity_level"
                    ]
                ),
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

        expected_query_rows.append(
            {
                "query_index": str(
                    query_index
                ),
                "query_id": query_id,
                "image_id": image_id,
                "semantic_id": (
                    query["semantic_id"]
                ),
                "split": query["split"],
                "pattern_id": (
                    query["pattern_id"]
                ),
                "palette_id": (
                    query["palette_id"]
                ),
                "ambiguity_level": (
                    query[
                        "ambiguity_level"
                    ]
                ),
                "template_id": (
                    query["template_id"]
                ),
                "omitted_attribute": (
                    query[
                        "omitted_attribute"
                    ]
                ),
                "image_row_index": str(
                    image_row_index
                ),
                "positive_candidate_id": (
                    positive_candidate[
                        "candidate_id"
                    ]
                ),
                "positive_position_source": (
                    positive_candidate[
                        "candidate_position"
                    ]
                ),
                "positive_rank": str(
                    result[
                        "first_relevant_rank"
                    ]
                ),
                "top1_candidate_id": (
                    top1_candidate[
                        "candidate_id"
                    ]
                ),
                "top1_candidate_role": (
                    top1_candidate[
                        "candidate_role"
                    ]
                ),
                "top1_score": format_float(
                    scores[top1_index]
                ),
                "positive_score": (
                    format_float(
                        positive_score
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
                "hard_negative_accuracy": (
                    format_float(
                        result[
                            "recall_at_1"
                        ]
                    )
                ),
            }
        )

        rank_by_index = {
            int(local_index): rank
            for rank, local_index
            in enumerate(
                ranking,
                start=1,
            )
        }

        assert (
            rank_by_index[
                positive_index
            ]
            == result[
                "first_relevant_rank"
            ]
        )

        for rank, local_index in enumerate(
            ranking,
            start=1,
        ):
            local_index = int(
                local_index
            )

            candidate = (
                query_candidates[
                    local_index
                ]
            )

            expected_ranking_rows.append(
                {
                    "query_index": str(
                        query_index
                    ),
                    "query_id": query_id,
                    "rank": str(rank),
                    "candidate_id": (
                        candidate[
                            "candidate_id"
                        ]
                    ),
                    "candidate_position": (
                        candidate[
                            "candidate_position"
                        ]
                    ),
                    "text_row_index": str(
                        candidate_text_rows[
                            local_index
                        ]
                    ),
                    "score": format_float(
                        scores[local_index]
                    ),
                    "is_positive": str(
                        local_index
                        == positive_index
                    ).lower(),
                    "relevance_label": (
                        candidate[
                            "relevance_label"
                        ]
                    ),
                    "candidate_role": (
                        candidate[
                            "candidate_role"
                        ]
                    ),
                    "changed_attribute": (
                        candidate[
                            "changed_attribute"
                        ]
                    ),
                    "negative_type": (
                        candidate[
                            "negative_type"
                        ]
                    ),
                    "is_global_positive_text": (
                        candidate[
                            "is_global_positive_text"
                        ]
                    ),
                    "is_negative_global_overlap": (
                        candidate[
                            "is_negative_global_overlap"
                        ]
                    ),
                }
            )

        for local_index, candidate in enumerate(
            query_candidates
        ):
            if local_index == positive_index:
                continue

            assert not parse_bool(
                candidate["is_positive"],
                candidate["candidate_id"],
            )

            negative_score = float(
                scores[local_index]
            )

            paired_difference = (
                positive_score
                - negative_score
            )

            positive_above_negative = (
                rank_by_index[
                    positive_index
                ]
                < rank_by_index[
                    local_index
                ]
            )

            expected_pairwise_rows.append(
                {
                    "query_index": str(
                        query_index
                    ),
                    "query_id": query_id,
                    "image_id": image_id,
                    "split": query["split"],
                    "ambiguity_level": (
                        query[
                            "ambiguity_level"
                        ]
                    ),
                    "positive_candidate_id": (
                        positive_candidate[
                            "candidate_id"
                        ]
                    ),
                    "negative_candidate_id": (
                        candidate[
                            "candidate_id"
                        ]
                    ),
                    "changed_attribute": (
                        candidate[
                            "changed_attribute"
                        ]
                    ),
                    "negative_type": (
                        candidate[
                            "negative_type"
                        ]
                    ),
                    "positive_score": (
                        format_float(
                            positive_score
                        )
                    ),
                    "negative_score": (
                        format_float(
                            negative_score
                        )
                    ),
                    "paired_difference": (
                        format_float(
                            paired_difference
                        )
                    ),
                    "positive_above_negative": (
                        str(
                            positive_above_negative
                        ).lower()
                    ),
                }
            )

    assert len(expected_query_rows) == 56
    assert len(expected_ranking_rows) == 280
    assert len(expected_pairwise_rows) == 224
    assert len(metric_rows) == 56

    aggregate_rows = [
        aggregate_group(
            metric_rows,
            "overall",
            "all",
        )
    ]

    for dimension in (
        "split",
        "ambiguity_level",
    ):
        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for row in metric_rows:
            grouped[
                str(row[dimension])
            ].append(row)

        for value in sorted(grouped):
            aggregate_rows.append(
                aggregate_group(
                    grouped[value],
                    dimension,
                    value,
                )
            )

    assert len(aggregate_rows) == 8

    overall = aggregate_query_metrics(
        metric_rows
    )

    assert (
        overall[
            "hard_negative_accuracy"
        ]
        == overall["recall_at_1"]
    )

    pairwise_differences = np.asarray(
        [
            float(
                row[
                    "paired_difference"
                ]
            )
            for row in expected_pairwise_rows
        ],
        dtype=np.float64,
    )

    pairwise_wins = np.asarray(
        [
            parse_bool(
                row[
                    "positive_above_negative"
                ],
                row[
                    "negative_candidate_id"
                ],
            )
            for row in expected_pairwise_rows
        ],
        dtype=np.float64,
    )

    mean_paired_difference = float(
        np.mean(
            pairwise_differences
        )
    )

    pairwise_win_rate = float(
        np.mean(pairwise_wins)
    )

    pairwise_win_count = int(
        np.sum(pairwise_wins)
    )

    expected_wins_from_ranks = sum(
        (
            5 - rank
        )
        * count
        for rank, count
        in rank_distribution.items()
    )

    assert (
        pairwise_win_count
        == expected_wins_from_ranks
    )

    return {
        "query_rows": expected_query_rows,
        "ranking_rows": expected_ranking_rows,
        "pairwise_rows": expected_pairwise_rows,
        "aggregate_rows": aggregate_rows,
        "overall": overall,
        "rank_distribution": (
            rank_distribution
        ),
        "mean_paired_difference": (
            mean_paired_difference
        ),
        "pairwise_win_rate": (
            pairwise_win_rate
        ),
        "pairwise_win_count": (
            pairwise_win_count
        ),
        "chance_level": chance_level,
    }


def validate_summary(
    reconstruction: dict[str, Any],
) -> None:
    """Valida el resumen y todos sus hashes."""

    summary = load_json(
        SUMMARY_PATH
    )

    assert summary["schema_version"] == "1.0"
    assert summary["dataset_version"] == "v2"
    assert summary["experiment_id"] == "E2"

    assert summary["experiment_name"] == (
        "hard_negative_forced_choice"
    )

    assert summary["evaluation_valid"] is True

    protocol = summary["protocol"]

    assert protocol["query_count"] == 56

    assert (
        protocol[
            "candidates_per_query"
        ]
        == 5
    )

    assert (
        protocol[
            "positives_per_query"
        ]
        == 1
    )

    assert (
        protocol[
            "negatives_per_query"
        ]
        == 4
    )

    assert_close(
        float(protocol["chance_level"]),
        reconstruction[
            "chance_level"
        ],
        "protocol.chance_level",
    )

    counts = summary["counts"]

    assert counts["queries"] == 56
    assert counts["candidate_rows"] == 280
    assert counts["pairwise_rows"] == 224

    expected_split_counts = Counter(
        row["split"]
        for row in reconstruction[
            "query_rows"
        ]
    )

    expected_ambiguity_counts = Counter(
        row["ambiguity_level"]
        for row in reconstruction[
            "query_rows"
        ]
    )

    expected_attribute_counts = Counter(
        row["changed_attribute"]
        for row in reconstruction[
            "pairwise_rows"
        ]
    )

    expected_negative_type_counts = Counter(
        row["negative_type"]
        for row in reconstruction[
            "pairwise_rows"
        ]
    )

    assert counts["splits"] == dict(
        sorted(
            expected_split_counts.items()
        )
    )

    assert (
        counts["ambiguity_level"]
        == dict(
            sorted(
                expected_ambiguity_counts.items()
            )
        )
    )

    assert (
        counts["changed_attribute"]
        == dict(
            sorted(
                expected_attribute_counts.items()
            )
        )
    )

    assert (
        counts["negative_type"]
        == dict(
            sorted(
                expected_negative_type_counts.items()
            )
        )
    )

    overall = reconstruction[
        "overall"
    ]

    summary_metrics = summary[
        "overall_metrics"
    ]

    expected_metric_keys = {
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
        "hard_negative_accuracy",
        "query_count",
        "mean_paired_difference",
        "pairwise_win_rate",
    }

    assert set(summary_metrics) == (
        expected_metric_keys
    )

    for metric in (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
        "hard_negative_accuracy",
        "query_count",
    ):
        assert_close(
            float(
                summary_metrics[metric]
            ),
            float(overall[metric]),
            f"overall_metrics.{metric}",
        )

    assert_close(
        float(
            summary_metrics[
                "mean_paired_difference"
            ]
        ),
        reconstruction[
            "mean_paired_difference"
        ],
        "mean_paired_difference",
    )

    assert_close(
        float(
            summary_metrics[
                "pairwise_win_rate"
            ]
        ),
        reconstruction[
            "pairwise_win_rate"
        ],
        "pairwise_win_rate",
    )

    assert_close(
        float(
            summary_metrics[
                "hard_negative_accuracy"
            ]
        ),
        float(
            summary_metrics[
                "recall_at_1"
            ]
        ),
        "accuracy_equals_recall_at_1",
    )

    expected_rank_distribution = {
        str(rank): count
        for rank, count
        in sorted(
            reconstruction[
                "rank_distribution"
            ].items()
        )
    }

    assert (
        summary[
            "positive_rank_distribution"
        ]
        == expected_rank_distribution
    )

    input_paths = {
        "experiment_config": (
            EXPERIMENT_CONFIG_PATH
        ),
        "queries": QUERIES_PATH,
        "candidates": CANDIDATES_PATH,
        "image_index": IMAGE_INDEX_PATH,
        "text_index": TEXT_INDEX_PATH,
        "text_usages": TEXT_USAGE_PATH,
        "image_embeddings": (
            IMAGE_MATRIX_PATH
        ),
        "text_embeddings": (
            TEXT_MATRIX_PATH
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
        "ranking": RANKING_PATH,
        "pairwise": PAIRWISE_PATH,
        "aggregates": AGGREGATES_PATH,
    }

    expected_rows = {
        "query_results": 56,
        "ranking": 280,
        "pairwise": 224,
        "aggregates": 8,
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
    """Ejecuta la validación independiente."""

    expected_names = {
        QUERY_RESULTS_PATH.name,
        RANKING_PATH.name,
        PAIRWISE_PATH.name,
        AGGREGATES_PATH.name,
        SUMMARY_PATH.name,
    }

    actual_names = {
        path.name
        for path in OUTPUT_DIRECTORY.iterdir()
        if path.is_file()
    }

    assert actual_names == expected_names

    query_fields, actual_queries = (
        load_csv(
            QUERY_RESULTS_PATH
        )
    )

    ranking_fields, actual_rankings = (
        load_csv(
            RANKING_PATH
        )
    )

    pairwise_fields, actual_pairwise = (
        load_csv(
            PAIRWISE_PATH
        )
    )

    aggregate_fields, actual_aggregates = (
        load_csv(
            AGGREGATES_PATH
        )
    )

    assert tuple(query_fields) == (
        QUERY_RESULT_FIELDS
    )

    assert tuple(ranking_fields) == (
        RANKING_FIELDS
    )

    assert tuple(pairwise_fields) == (
        PAIRWISE_FIELDS
    )

    assert tuple(aggregate_fields) == (
        AGGREGATE_FIELDS
    )

    reconstruction = reconstruct()

    assert actual_queries == (
        reconstruction[
            "query_rows"
        ]
    )

    assert actual_rankings == (
        reconstruction[
            "ranking_rows"
        ]
    )

    assert actual_pairwise == (
        reconstruction[
            "pairwise_rows"
        ]
    )

    assert actual_aggregates == (
        reconstruction[
            "aggregate_rows"
        ]
    )

    validate_summary(
        reconstruction
    )

    overall = reconstruction["overall"]

    print("=" * 80)
    print("VALIDACIÓN INDEPENDIENTE DE E2 SUPERADA")
    print("=" * 80)
    print("Consultas reconstruidas: 56")
    print("Filas de ranking: 280")
    print("Comparaciones pareadas: 224")
    print("Agregados: 8")
    print(
        "Recall@1:",
        format_float(
            overall["recall_at_1"]
        ),
    )
    print(
        "Recall@5:",
        format_float(
            overall["recall_at_5"]
        ),
    )
    print(
        "MRR:",
        format_float(
            overall["mrr"]
        ),
    )
    print(
        "nDCG@10:",
        format_float(
            overall["ndcg_at_10"]
        ),
    )
    print(
        "Hard-negative accuracy:",
        format_float(
            overall[
                "hard_negative_accuracy"
            ]
        ),
    )
    print(
        "Margen positivo:",
        format_float(
            overall[
                "positive_margin"
            ]
        ),
    )
    print(
        "Diferencia pareada media:",
        format_float(
            reconstruction[
                "mean_paired_difference"
            ]
        ),
    )
    print(
        "Victorias pareadas:",
        reconstruction[
            "pairwise_win_count"
        ],
        "/ 224",
    )
    print(
        "Tasa de victorias pareadas:",
        format_float(
            reconstruction[
                "pairwise_win_rate"
            ]
        ),
    )
    print(
        "Distribución de rangos:",
        dict(
            sorted(
                reconstruction[
                    "rank_distribution"
                ].items()
            )
        ),
    )
    print("Rankings: reconstruidos")
    print("Comparaciones pareadas: reconstruidas")
    print("Agregados: reconstruidos")
    print("Hashes de entradas y salidas: válidos")
    print("Resumen: válido")
    print("Evaluación válida: True")


if __name__ == "__main__":
    main()
