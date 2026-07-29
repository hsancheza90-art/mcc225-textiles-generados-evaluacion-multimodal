"""Evalúa OpenCLIP mediante elección forzada con negativos difíciles v2."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
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

EMBEDDINGS_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "embeddings_v2.json"
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

TEMPORARY_DIRECTORY = (
    OUTPUT_DIRECTORY.with_name(
        OUTPUT_DIRECTORY.name + ".tmp"
    )
)

RESULTS_FILENAME = (
    "resultados_consulta_"
    "hard_negatives_openclip_v2.csv"
)

RANKING_FILENAME = (
    "ranking_hard_negatives_openclip_v2.csv"
)

PAIRWISE_FILENAME = (
    "comparaciones_pareadas_"
    "hard_negatives_openclip_v2.csv"
)

AGGREGATES_FILENAME = (
    "agregados_hard_negatives_openclip_v2.csv"
)

SUMMARY_FILENAME = (
    "resumen_hard_negatives_openclip_v2.json"
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


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    """Escribe un CSV UTF-8/LF determinista."""

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
    """Escribe un JSON UTF-8/LF determinista."""

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
    """Calcula SHA-256 sobre un archivo."""

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
    """Convierte booleanos CSV serializados como texto."""

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
    """Serializa reales de manera estable."""

    return format(
        float(value),
        ".12f",
    )


def validate_experiment_contract(
    experiment: dict[str, Any],
) -> float:
    """Valida el contrato de E2."""

    experiments = {
        record["experiment_id"]: record
        for record in experiment["experiments"]
    }

    assert experiments["E2"] == {
        "experiment_id": "E2",
        "name": "hard_negative_forced_choice",
        "dataset": "v2",
    }

    primary = set(
        experiment["metrics"]["primary"]
    )

    supplementary = set(
        experiment["metrics"]["supplementary"]
    )

    assert {
        "recall_at_1",
        "mrr",
        "ndcg_at_10",
    }.issubset(primary)

    assert {
        "recall_at_5",
        "hard_negative_accuracy",
        "positive_margin",
        "paired_difference",
    }.issubset(supplementary)

    chance_level = float(
        experiment[
            "metrics"
        ][
            "hard_negative_chance_level"
        ]
    )

    assert chance_level == 0.2

    return chance_level


def aggregate_group(
    rows: list[dict[str, Any]],
    dimension: str,
    value: str,
) -> dict[str, Any]:
    """Agrega métricas por grupo."""

    aggregate = aggregate_query_metrics(
        rows
    )

    return {
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
        "hard_negative_accuracy": (
            format_float(
                aggregate[
                    "hard_negative_accuracy"
                ]
            )
        ),
    }


def build_aggregates(
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construye agregados generales, por split y ambigüedad."""

    rows = [
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

        for record in metric_rows:
            grouped[
                str(record[dimension])
            ].append(record)

        for value in sorted(grouped):
            rows.append(
                aggregate_group(
                    grouped[value],
                    dimension,
                    value,
                )
            )

    assert len(rows) == 8

    return rows


def publish_directory(
    temporary_directory: Path,
    output_directory: Path,
) -> None:
    """Publica los artefactos de manera atómica."""

    backup_directory = (
        output_directory.with_name(
            output_directory.name + ".previous"
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
    """Ejecuta E2 con los embeddings OpenCLIP existentes."""

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    embeddings_config = load_json(
        EMBEDDINGS_CONFIG_PATH
    )

    chance_level = (
        validate_experiment_contract(
            experiment
        )
    )

    query_fields, queries = load_csv(
        QUERIES_PATH
    )

    candidate_fields, candidates = (
        load_csv(
            CANDIDATES_PATH
        )
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

    required_query_fields = {
        "query_id",
        "query_index",
        "image_id",
        "semantic_id",
        "split",
        "pattern_id",
        "palette_id",
        "ambiguity_level",
        "template_id",
        "omitted_attribute",
        "positive_position",
        "positive_candidate_id",
        "positive_caption_text",
        "candidate_ids",
        "candidate_count",
        "negative_count",
    }

    required_candidate_fields = {
        "query_id",
        "query_index",
        "candidate_id",
        "candidate_position",
        "image_id",
        "semantic_id",
        "split",
        "pattern_id",
        "ambiguity_level",
        "template_id",
        "candidate_role",
        "is_positive",
        "relevance_label",
        "changed_attribute",
        "negative_type",
        "is_global_positive_text",
        "is_negative_global_overlap",
        "candidate_text",
        "candidate_sha256",
    }

    assert required_query_fields.issubset(
        set(query_fields)
    )

    assert required_candidate_fields.issubset(
        set(candidate_fields)
    )

    assert len(queries) == 56
    assert len(candidates) == 280
    assert len(image_rows) == 56
    assert len(text_rows) == 494
    assert len(usage_rows) == 600

    image_embeddings = np.load(
        IMAGE_MATRIX_PATH,
        allow_pickle=False,
    )

    text_embeddings = np.load(
        TEXT_MATRIX_PATH,
        allow_pickle=False,
    )

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

    print("=" * 80)
    print("EVALUACIÓN HARD NEGATIVES OPENCLIP V2")
    print("=" * 80)
    print("Consultas:", len(ordered_queries))
    print("Candidatos por consulta: 5")
    print("Negativos por consulta: 4")
    print(
        "Nivel de azar:",
        format_float(chance_level),
    )
    print(
        "Similitud:",
        embeddings_config["model"]["similarity"],
    )
    print(
        "Desempate: candidate_id ascendente"
    )

    query_output_rows: list[
        dict[str, Any]
    ] = []

    ranking_output_rows: list[
        dict[str, Any]
    ] = []

    pairwise_output_rows: list[
        dict[str, Any]
    ] = []

    metric_rows: list[
        dict[str, Any]
    ] = []

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

        shared_query_image_fields = (
            "semantic_id",
            "split",
            "pattern_id",
            "palette_id",
            "ambiguity_level",
        )

        for field in shared_query_image_fields:
            assert (
                query[field]
                == image_record[field]
            ), (
                f"{query_id}: diferencia en "
                f"{field} entre consulta e imagen."
            )

        query_candidates = sorted(
            candidates_by_query[query_id],
            key=lambda row: int(
                row["candidate_position"]
            ),
        )

        assert len(query_candidates) == 5

        positions = [
            int(row["candidate_position"])
            for row in query_candidates
        ]

        assert positions == [1, 2, 3, 4, 5]

        assert query["candidate_ids"].split(
            "|"
        ) == [
            row["candidate_id"]
            for row in query_candidates
        ]

        assert int(
            query["candidate_count"]
        ) == 5

        assert int(
            query["negative_count"]
        ) == 4

        positive_local_indices = [
            index
            for index, candidate
            in enumerate(query_candidates)
            if parse_bool(
                candidate["is_positive"],
                candidate["candidate_id"],
            )
        ]

        assert len(
            positive_local_indices
        ) == 1

        positive_local_index = (
            positive_local_indices[0]
        )

        positive_candidate = (
            query_candidates[
                positive_local_index
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

        assert (
            positive_candidate[
                "candidate_role"
            ]
            == "positive"
        )

        assert (
            positive_candidate[
                "relevance_label"
            ]
            == "1"
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
                positive_local_index
            },
            candidate_keys=candidate_keys,
        )

        ranking = result[
            "ranking_indices"
        ]

        top1_local_index = int(
            ranking[0]
        )

        top1_candidate = (
            query_candidates[
                top1_local_index
            ]
        )

        positive_score = float(
            scores[
                positive_local_index
            ]
        )

        metric_record: dict[
            str,
            Any,
        ] = {
            "split": query["split"],
            "ambiguity_level": (
                query["ambiguity_level"]
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
                result["positive_margin"]
            ),
        }

        metric_rows.append(
            metric_record
        )

        query_output_rows.append(
            {
                "query_index": query_index,
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
                "image_row_index": (
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
                "positive_rank": (
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
                    scores[
                        top1_local_index
                    ]
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

        rank_by_local_index = {
            int(local_index): rank
            for rank, local_index
            in enumerate(
                ranking,
                start=1,
            )
        }

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

            ranking_output_rows.append(
                {
                    "query_index": (
                        query_index
                    ),
                    "query_id": query_id,
                    "rank": rank,
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
                    "text_row_index": (
                        candidate_text_rows[
                            local_index
                        ]
                    ),
                    "score": format_float(
                        scores[local_index]
                    ),
                    "is_positive": str(
                        local_index
                        == positive_local_index
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

        assert (
            rank_by_local_index[
                positive_local_index
            ]
            == result[
                "first_relevant_rank"
            ]
        )

        for local_index, candidate in enumerate(
            query_candidates
        ):
            if local_index == positive_local_index:
                continue

            assert not parse_bool(
                candidate["is_positive"],
                candidate["candidate_id"],
            )

            assert (
                candidate[
                    "candidate_role"
                ]
                == "hard_negative"
            )

            assert (
                candidate[
                    "relevance_label"
                ]
                == "0"
            )

            negative_score = float(
                scores[local_index]
            )

            difference = (
                positive_score
                - negative_score
            )

            positive_above_negative = (
                rank_by_local_index[
                    positive_local_index
                ]
                < rank_by_local_index[
                    local_index
                ]
            )

            pairwise_output_rows.append(
                {
                    "query_index": (
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
                            difference
                        )
                    ),
                    "positive_above_negative": (
                        str(
                            positive_above_negative
                        ).lower()
                    ),
                }
            )

    assert len(query_output_rows) == 56
    assert len(ranking_output_rows) == 280
    assert len(pairwise_output_rows) == 224
    assert len(metric_rows) == 56

    aggregate_rows = build_aggregates(
        metric_rows
    )

    overall = aggregate_query_metrics(
        metric_rows
    )

    assert (
        overall[
            "hard_negative_accuracy"
        ]
        == overall["recall_at_1"]
    )

    rank_distribution = Counter(
        int(row["positive_rank"])
        for row in query_output_rows
    )

    split_counts = Counter(
        row["split"]
        for row in query_output_rows
    )

    ambiguity_counts = Counter(
        row["ambiguity_level"]
        for row in query_output_rows
    )

    changed_attribute_counts = Counter(
        row["changed_attribute"]
        for row in pairwise_output_rows
    )

    negative_type_counts = Counter(
        row["negative_type"]
        for row in pairwise_output_rows
    )

    pairwise_differences = np.asarray(
        [
            float(
                row[
                    "paired_difference"
                ]
            )
            for row in pairwise_output_rows
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
            for row in pairwise_output_rows
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

    ranking_path = (
        TEMPORARY_DIRECTORY
        / RANKING_FILENAME
    )

    pairwise_path = (
        TEMPORARY_DIRECTORY
        / PAIRWISE_FILENAME
    )

    aggregates_path = (
        TEMPORARY_DIRECTORY
        / AGGREGATES_FILENAME
    )

    summary_path = (
        TEMPORARY_DIRECTORY
        / SUMMARY_FILENAME
    )

    try:
        write_csv(
            results_path,
            QUERY_RESULT_FIELDS,
            query_output_rows,
        )

        write_csv(
            ranking_path,
            RANKING_FIELDS,
            ranking_output_rows,
        )

        write_csv(
            pairwise_path,
            PAIRWISE_FIELDS,
            pairwise_output_rows,
        )

        write_csv(
            aggregates_path,
            AGGREGATE_FIELDS,
            aggregate_rows,
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
                "rows": 56,
                "sha256": sha256_file(
                    results_path
                ),
            },
            "ranking": {
                "path": (
                    OUTPUT_DIRECTORY
                    / RANKING_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 280,
                "sha256": sha256_file(
                    ranking_path
                ),
            },
            "pairwise": {
                "path": (
                    OUTPUT_DIRECTORY
                    / PAIRWISE_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 224,
                "sha256": sha256_file(
                    pairwise_path
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
                "rows": 8,
                "sha256": sha256_file(
                    aggregates_path
                ),
            },
        }

        summary = {
            "schema_version": "1.0",
            "dataset_version": "v2",
            "experiment_id": "E2",
            "experiment_name": (
                "hard_negative_forced_choice"
            ),
            "condition": (
                "openclip_original_image_"
                "five_candidate_texts"
            ),
            "protocol": {
                "query_source": (
                    "consultas_negativos_"
                    "dificiles_v2"
                ),
                "query_count": 56,
                "candidate_source": (
                    "candidatos_negativos_"
                    "dificiles_v2"
                ),
                "candidates_per_query": 5,
                "positives_per_query": 1,
                "negatives_per_query": 4,
                "chance_level": (
                    chance_level
                ),
                "similarity": (
                    "dot_product_of_"
                    "normalized_embeddings"
                ),
                "tie_breaker": (
                    "candidate_id_ascending"
                ),
            },
            "metrics": {
                "primary": [
                    "recall_at_1",
                    "mrr",
                    "ndcg_at_10",
                ],
                "supplementary": [
                    "recall_at_5",
                    "hard_negative_accuracy",
                    "positive_margin",
                    "paired_difference",
                ],
            },
            "counts": {
                "queries": 56,
                "candidate_rows": 280,
                "pairwise_rows": 224,
                "splits": dict(
                    sorted(
                        split_counts.items()
                    )
                ),
                "ambiguity_level": dict(
                    sorted(
                        ambiguity_counts.items()
                    )
                ),
                "changed_attribute": dict(
                    sorted(
                        changed_attribute_counts.items()
                    )
                ),
                "negative_type": dict(
                    sorted(
                        negative_type_counts.items()
                    )
                ),
            },
            "overall_metrics": {
                "recall_at_1": float(
                    overall["recall_at_1"]
                ),
                "recall_at_5": float(
                    overall["recall_at_5"]
                ),
                "mrr": float(
                    overall["mrr"]
                ),
                "ndcg_at_10": float(
                    overall["ndcg_at_10"]
                ),
                "positive_margin": float(
                    overall[
                        "positive_margin"
                    ]
                ),
                "hard_negative_accuracy": (
                    float(
                        overall[
                            "hard_negative_accuracy"
                        ]
                    )
                ),
                "query_count": float(
                    overall["query_count"]
                ),
                "mean_paired_difference": (
                    mean_paired_difference
                ),
                "pairwise_win_rate": (
                    pairwise_win_rate
                ),
            },
            "positive_rank_distribution": {
                str(rank): count
                for rank, count
                in sorted(
                    rank_distribution.items()
                )
            },
            "input_artifacts": {
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
                "embeddings_config": {
                    "path": (
                        EMBEDDINGS_CONFIG_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "sha256": sha256_file(
                        EMBEDDINGS_CONFIG_PATH
                    ),
                },
                "queries": {
                    "path": (
                        QUERIES_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 56,
                    "sha256": sha256_file(
                        QUERIES_PATH
                    ),
                },
                "candidates": {
                    "path": (
                        CANDIDATES_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 280,
                    "sha256": sha256_file(
                        CANDIDATES_PATH
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
                "text_index": {
                    "path": (
                        TEXT_INDEX_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 494,
                    "sha256": sha256_file(
                        TEXT_INDEX_PATH
                    ),
                },
                "text_usages": {
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
                "image_embeddings": {
                    "path": (
                        IMAGE_MATRIX_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "shape": [56, 512],
                    "sha256": sha256_file(
                        IMAGE_MATRIX_PATH
                    ),
                },
                "text_embeddings": {
                    "path": (
                        TEXT_MATRIX_PATH
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "shape": [494, 512],
                    "sha256": sha256_file(
                        TEXT_MATRIX_PATH
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
                "device": (
                    "precomputed_cpu_embeddings"
                ),
            },
            "evaluation_valid": True,
        }

        write_json(
            summary_path,
            summary,
        )

        expected_names = {
            RESULTS_FILENAME,
            RANKING_FILENAME,
            PAIRWISE_FILENAME,
            AGGREGATES_FILENAME,
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
        "EVALUACIÓN HARD NEGATIVES "
        "OPENCLIP V2 COMPLETADA"
    )
    print("=" * 80)
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
        "Nivel de azar:",
        format_float(chance_level),
    )
    print(
        "Margen positivo medio:",
        format_float(
            overall["positive_margin"]
        ),
    )
    print(
        "Diferencia pareada media:",
        format_float(
            mean_paired_difference
        ),
    )
    print(
        "Tasa de victorias pareadas:",
        format_float(
            pairwise_win_rate
        ),
    )
    print(
        "Distribución del rango positivo:",
        dict(
            sorted(
                rank_distribution.items()
            )
        ),
    )

    print()
    print("Resultados por split:")

    for row in aggregate_rows:
        if row["group_dimension"] != "split":
            continue

        print(
            "-",
            row["group_value"],
            "n=",
            row["query_count"],
            "accuracy=",
            row[
                "hard_negative_accuracy"
            ],
            "MRR=",
            row["mrr"],
            "margin=",
            row["positive_margin"],
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
