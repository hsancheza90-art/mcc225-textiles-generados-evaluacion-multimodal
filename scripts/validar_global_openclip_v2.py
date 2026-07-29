"""Valida independientemente la evaluación global OpenCLIP v2."""

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

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
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
    / "global_openclip_v2"
)

QUERY_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "resultados_consulta_global_openclip_v2.csv"
)

RANKING_PATH = (
    OUTPUT_DIRECTORY
    / "ranking_global_openclip_v2.csv"
)

AGGREGATES_PATH = (
    OUTPUT_DIRECTORY
    / "agregados_global_openclip_v2.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "resumen_global_openclip_v2.json"
)


QUERY_RESULT_FIELDS = (
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

RANKING_FIELDS = (
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


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1"}:
        return True

    if normalized in {"false", "0"}:
        return False

    raise ValueError(
        f"Booleano no reconocido: {value!r}"
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


def aggregate_group(
    rows: list[dict[str, Any]],
    dimension: str,
    value: str,
) -> dict[str, str]:
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
    }


def reconstruct() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, float],
    Counter[int],
]:
    _, captions = load_csv(
        CAPTIONS_PATH
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

    assert len(captions) == 280
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
        image_id: row_index
        for row_index, image_id
        in enumerate(image_ids)
    }

    usage_map = {
        (
            row["source_id"],
            row["record_id"],
        ): int(row["text_row_index"])
        for row in usage_rows
    }

    text_hash_by_row = {
        int(row["text_row_index"]): (
            row["text_sha256"]
        )
        for row in text_rows
    }

    expected_queries: list[
        dict[str, str]
    ] = []

    expected_rankings: list[
        dict[str, str]
    ] = []

    metric_rows: list[
        dict[str, Any]
    ] = []

    rank_distribution: Counter[int] = (
        Counter()
    )

    for query_index, caption in enumerate(
        captions
    ):
        caption_id = caption[
            "caption_id"
        ]

        image_id = caption["image_id"]

        text_row_index = usage_map[
            (
                "positivos",
                caption_id,
            )
        ]

        assert (
            text_hash_by_row[
                text_row_index
            ]
            == caption["caption_sha256"]
        )

        relevant_row = image_row_by_id[
            image_id
        ]

        image_record = ordered_images[
            relevant_row
        ]

        shared_fields = (
            "semantic_id",
            "split",
            "pattern_id",
            "palette_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
        )

        for field in shared_fields:
            assert (
                caption[field]
                == image_record[field]
            )

        scores = (
            image_embeddings
            @ text_embeddings[
                text_row_index
            ]
        )

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
        top1_id = image_ids[top1_row]

        canonical = str(
            parse_bool(
                caption["is_canonical"]
            )
        ).lower()

        rank_distribution[
            int(
                result[
                    "first_relevant_rank"
                ]
            )
        ] += 1

        metric_rows.append(
            {
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

        expected_queries.append(
            {
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
                    image_record[
                        "ambiguity_level"
                    ]
                ),
                "template_id": (
                    caption["template_id"]
                ),
                "is_canonical": canonical,
                "text_row_index": str(
                    text_row_index
                ),
                "relevant_image_row_index": (
                    str(relevant_row)
                ),
                "relevant_rank": str(
                    result[
                        "first_relevant_rank"
                    ]
                ),
                "top1_image_id": top1_id,
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

        for rank, row_index in enumerate(
            ranking,
            start=1,
        ):
            row_index = int(row_index)
            ranked_image = ordered_images[
                row_index
            ]

            expected_rankings.append(
                {
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
                        ranked_image[
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
                        ranked_image[
                            "split"
                        ]
                    ),
                    "image_pattern_id": (
                        ranked_image[
                            "pattern_id"
                        ]
                    ),
                    "image_palette_id": (
                        ranked_image[
                            "palette_id"
                        ]
                    ),
                }
            )

    aggregate_rows = [
        aggregate_group(
            metric_rows,
            "overall",
            "all",
        )
    ]

    for dimension in (
        "split",
        "is_canonical",
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

    raw_overall = (
        aggregate_query_metrics(
            metric_rows
        )
    )

    overall_keys = (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
        "query_count",
    )

    overall = {
        key: float(raw_overall[key])
        for key in overall_keys
    }

    assert len(expected_queries) == 280
    assert len(expected_rankings) == 15680
    assert len(aggregate_rows) == 7

    return (
        expected_queries,
        expected_rankings,
        aggregate_rows,
        overall,
        rank_distribution,
    )


def validate_summary(
    overall: dict[str, float],
    rank_distribution: Counter[int],
) -> None:
    summary = load_json(
        SUMMARY_PATH
    )

    assert summary["schema_version"] == "1.0"
    assert summary["dataset_version"] == "v2"
    assert summary["experiment_id"] == "E1"

    assert summary["experiment_name"] == (
        "global_retrieval_v2"
    )

    assert summary["evaluation_valid"] is True

    assert summary["counts"]["queries"] == 280
    assert (
        summary["counts"][
            "gallery_images"
        ]
        == 56
    )
    assert (
        summary["counts"][
            "ranking_rows"
        ]
        == 15680
    )

    assert (
        "hard_negative_accuracy"
        not in summary[
            "overall_metrics"
        ]
    )

    assert set(
        summary["overall_metrics"]
    ) == set(overall)

    for key, expected in overall.items():
        assert_close(
            float(
                summary[
                    "overall_metrics"
                ][key]
            ),
            expected,
            f"overall_metrics.{key}",
        )

    expected_distribution = {
        str(rank): count
        for rank, count in sorted(
            rank_distribution.items()
        )
    }

    assert (
        summary[
            "relevant_rank_distribution"
        ]
        == expected_distribution
    )

    output_paths = {
        "query_results": (
            QUERY_RESULTS_PATH
        ),
        "ranking": RANKING_PATH,
        "aggregates": AGGREGATES_PATH,
    }

    expected_rows = {
        "query_results": 280,
        "ranking": 15680,
        "aggregates": 7,
    }

    for key, path in (
        output_paths.items()
    ):
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
    expected_names = {
        QUERY_RESULTS_PATH.name,
        RANKING_PATH.name,
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

    assert tuple(aggregate_fields) == (
        AGGREGATE_FIELDS
    )

    (
        expected_queries,
        expected_rankings,
        expected_aggregates,
        overall,
        rank_distribution,
    ) = reconstruct()

    assert actual_queries == (
        expected_queries
    )

    assert actual_rankings == (
        expected_rankings
    )

    assert actual_aggregates == (
        expected_aggregates
    )

    validate_summary(
        overall,
        rank_distribution,
    )

    print("=" * 80)
    print("VALIDACIÓN INDEPENDIENTE DE E1 SUPERADA")
    print("=" * 80)
    print("Consultas reconstruidas: 280")
    print("Filas de ranking: 15680")
    print("Agregados: 7")
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
        "Margen positivo:",
        format_float(
            overall["positive_margin"]
        ),
    )
    print("Rankings: reconstruidos")
    print("Agregados: reconstruidos")
    print("Hashes de salidas: válidos")
    print("Resumen: válido")
    print("Evaluación válida: True")


if __name__ == "__main__":
    main()
