"""Evalúa recuperación global texto-imagen con OpenCLIP sobre el dataset v2."""

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

TEMPORARY_DIRECTORY = (
    OUTPUT_DIRECTORY.with_name(
        OUTPUT_DIRECTORY.name + ".tmp"
    )
)

RESULTS_FILENAME = (
    "resultados_consulta_global_openclip_v2.csv"
)

RANKING_FILENAME = (
    "ranking_global_openclip_v2.csv"
)

AGGREGATES_FILENAME = (
    "agregados_global_openclip_v2.csv"
)

SUMMARY_FILENAME = (
    "resumen_global_openclip_v2.json"
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


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON UTF-8 sin BOM y con finales LF."""

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM UTF-8."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
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
    ), f"{path}: contiene BOM UTF-8."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
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
    """Calcula SHA-256 sobre los bytes de un archivo."""

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
    """Serializa un real con precisión estable."""

    return format(
        float(value),
        ".12f",
    )


def validate_experiment_contract(
    experiment: dict[str, Any],
) -> None:
    """Comprueba que E1 y sus métricas estén definidos."""

    experiments = {
        record["experiment_id"]: record
        for record in experiment["experiments"]
    }

    assert "E1" in experiments

    assert experiments["E1"] == {
        "experiment_id": "E1",
        "name": "global_retrieval_v2",
        "dataset": "v2",
    }

    primary_metrics = set(
        experiment["metrics"]["primary"]
    )

    supplementary_metrics = set(
        experiment["metrics"]["supplementary"]
    )

    assert {
        "recall_at_1",
        "mrr",
        "ndcg_at_10",
    }.issubset(primary_metrics)

    assert {
        "recall_at_5",
        "positive_margin",
    }.issubset(supplementary_metrics)


def load_embedding_contract() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    np.ndarray,
    np.ndarray,
]:
    """Carga índices y matrices ya validados."""

    image_fields, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    text_fields, text_rows = load_csv(
        TEXT_INDEX_PATH
    )

    usage_fields, usage_rows = load_csv(
        TEXT_USAGE_PATH
    )

    assert "image_row_index" in image_fields
    assert "image_id" in image_fields
    assert "text_row_index" in text_fields
    assert "text_sha256" in text_fields
    assert "source_id" in usage_fields
    assert "record_id" in usage_fields

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

    return (
        image_rows,
        text_rows,
        usage_rows,
        image_embeddings,
        text_embeddings,
    )


def aggregate_group(
    rows: list[dict[str, Any]],
    dimension: str,
    value: str,
) -> dict[str, Any]:
    """Agrega las métricas de una agrupación."""

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
    }


def build_aggregates(
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Construye agregados generales y por grupos básicos."""

    aggregate_rows = [
        aggregate_group(
            metric_rows,
            "overall",
            "all",
        )
    ]

    group_fields = (
        "split",
        "is_canonical",
    )

    for field in group_fields:
        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = defaultdict(list)

        for row in metric_rows:
            grouped[
                str(row[field])
            ].append(row)

        for value in sorted(grouped):
            aggregate_rows.append(
                aggregate_group(
                    grouped[value],
                    field,
                    value,
                )
            )

    assert len(aggregate_rows) == 7

    return aggregate_rows


def publish_directory(
    temporary_directory: Path,
    output_directory: Path,
) -> None:
    """Publica resultados de forma atómica."""

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
    """Ejecuta la evaluación global OpenCLIP."""

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    embeddings_config = load_json(
        EMBEDDINGS_CONFIG_PATH
    )

    validate_experiment_contract(
        experiment
    )

    (
        image_rows,
        text_rows,
        usage_rows,
        image_embeddings,
        text_embeddings,
    ) = load_embedding_contract()

    caption_fields, captions = load_csv(
        CAPTIONS_PATH
    )

    required_caption_fields = {
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
        "template_id",
        "is_canonical",
        "caption_sha256",
    }

    assert required_caption_fields.issubset(
        set(caption_fields)
    )

    assert len(captions) == 280

    caption_ids = [
        row["caption_id"]
        for row in captions
    ]

    assert len(set(caption_ids)) == 280

    image_rows_ordered = sorted(
        image_rows,
        key=lambda row: int(
            row["image_row_index"]
        ),
    )

    assert [
        int(row["image_row_index"])
        for row in image_rows_ordered
    ] == list(range(56))

    image_id_by_row = [
        row["image_id"]
        for row in image_rows_ordered
    ]

    image_row_by_id = {
        image_id: row_index
        for row_index, image_id
        in enumerate(image_id_by_row)
    }

    image_keys = tuple(
        image_id_by_row
    )

    usage_row_by_key = {
        (
            row["source_id"],
            row["record_id"],
        ): int(row["text_row_index"])
        for row in usage_rows
    }

    assert len(usage_row_by_key) == 600

    text_hash_by_row = {
        int(row["text_row_index"]): (
            row["text_sha256"]
        )
        for row in text_rows
    }

    print("=" * 80)
    print("EVALUACIÓN GLOBAL OPENCLIP V2")
    print("=" * 80)
    print("Consultas:", len(captions))
    print("Galería:", len(image_rows_ordered))
    print("Dimensión:", image_embeddings.shape[1])
    print(
        "Similitud:",
        embeddings_config["model"]["similarity"],
    )
    print("Desempate: image_id ascendente")

    query_output_rows: list[
        dict[str, Any]
    ] = []

    ranking_output_rows: list[
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

        usage_key = (
            "positivos",
            caption_id,
        )

        assert usage_key in usage_row_by_key

        text_row_index = (
            usage_row_by_key[
                usage_key
            ]
        )

        assert (
            text_hash_by_row[
                text_row_index
            ]
            == caption["caption_sha256"]
        )

        assert image_id in image_row_by_id

        relevant_image_row_index = (
            image_row_by_id[image_id]
        )

        relevant_image_record = (
            image_rows_ordered[
                relevant_image_row_index
            ]
        )

        shared_metadata_fields = (
            "semantic_id",
            "split",
            "pattern_id",
            "palette_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
        )

        for metadata_field in (
            shared_metadata_fields
        ):
            assert (
                caption[metadata_field]
                == relevant_image_record[
                    metadata_field
                ]
            ), (
                f"{caption_id}: diferencia en "
                f"{metadata_field} entre caption "
                "e índice visual."
            )

        ambiguity_level = (
            relevant_image_record[
                "ambiguity_level"
            ]
        )

        assert ambiguity_level in {
            "bajo",
            "medio",
            "alto",
        }

        text_vector = text_embeddings[
            text_row_index
        ]

        scores = (
            image_embeddings
            @ text_vector
        )

        assert scores.shape == (56,)
        assert np.isfinite(scores).all()

        assert float(scores.min()) >= -1.0001
        assert float(scores.max()) <= 1.0001

        result = evaluate_query(
            scores=scores,
            relevant_indices={
                relevant_image_row_index
            },
            candidate_keys=image_keys,
        )

        ranking = result[
            "ranking_indices"
        ]

        top1_image_row_index = int(
            ranking[0]
        )

        top1_image_id = image_id_by_row[
            top1_image_row_index
        ]

        is_canonical = parse_bool(
            caption["is_canonical"],
            caption_id,
        )

        metric_record: dict[
            str,
            Any,
        ] = {
            "split": caption["split"],
            "is_canonical": str(
                is_canonical
            ).lower(),
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
                    ambiguity_level
                ),
                "template_id": (
                    caption["template_id"]
                ),
                "is_canonical": str(
                    is_canonical
                ).lower(),
                "text_row_index": (
                    text_row_index
                ),
                "relevant_image_row_index": (
                    relevant_image_row_index
                ),
                "relevant_rank": (
                    result[
                        "first_relevant_rank"
                    ]
                ),
                "top1_image_id": (
                    top1_image_id
                ),
                "top1_score": format_float(
                    scores[
                        top1_image_row_index
                    ]
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

        for rank, image_row_index in enumerate(
            ranking,
            start=1,
        ):
            row_index = int(
                image_row_index
            )

            image_record = (
                image_rows_ordered[
                    row_index
                ]
            )

            ranking_output_rows.append(
                {
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
                        == relevant_image_row_index
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

    assert len(query_output_rows) == 280

    assert len(ranking_output_rows) == (
        280 * 56
    )

    aggregate_rows = build_aggregates(
        metric_rows
    )

    raw_overall_metrics = (
        aggregate_query_metrics(
            metric_rows
        )
    )

    overall_metric_keys = (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
        "query_count",
    )

    overall_metrics = {
        metric_key: (
            raw_overall_metrics[
                metric_key
            ]
        )
        for metric_key in overall_metric_keys
    }

    assert (
        "hard_negative_accuracy"
        not in overall_metrics
    )

    rank_distribution = Counter(
        int(
            row["relevant_rank"]
        )
        for row in query_output_rows
    )

    split_counts = Counter(
        row["split"]
        for row in query_output_rows
    )

    canonical_counts = Counter(
        row["is_canonical"]
        for row in query_output_rows
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
                "rows": 280,
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
                "rows": 15680,
                "sha256": sha256_file(
                    ranking_path
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
                "rows": len(
                    aggregate_rows
                ),
                "sha256": sha256_file(
                    aggregates_path
                ),
            },
        }

        summary = {
            "schema_version": "1.0",
            "dataset_version": "v2",
            "experiment_id": "E1",
            "experiment_name": (
                "global_retrieval_v2"
            ),
            "condition": (
                "openclip_original_image_"
                "full_caption"
            ),
            "protocol": {
                "query_source": (
                    "captions_positivos_v2"
                ),
                "query_count": 280,
                "gallery_source": (
                    "image_original_embeddings"
                ),
                "gallery_count": 56,
                "relevance": (
                    "one associated image "
                    "per full caption"
                ),
                "similarity": (
                    "dot_product_of_"
                    "normalized_embeddings"
                ),
                "tie_breaker": (
                    "image_id_ascending"
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
                    "positive_margin",
                ],
            },
            "counts": {
                "queries": 280,
                "gallery_images": 56,
                "ranking_rows": 15680,
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
                key: float(value)
                for key, value
                in overall_metrics.items()
            },
            "relevant_rank_distribution": {
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
                "device": "precomputed_cpu_embeddings",
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
    print("EVALUACIÓN GLOBAL OPENCLIP V2 COMPLETADA")
    print("=" * 80)
    print(
        "Recall@1:",
        format_float(
            overall_metrics[
                "recall_at_1"
            ]
        ),
    )
    print(
        "Recall@5:",
        format_float(
            overall_metrics[
                "recall_at_5"
            ]
        ),
    )
    print(
        "MRR:",
        format_float(
            overall_metrics["mrr"]
        ),
    )
    print(
        "nDCG@10:",
        format_float(
            overall_metrics[
                "ndcg_at_10"
            ]
        ),
    )
    print(
        "Margen positivo medio:",
        format_float(
            overall_metrics[
                "positive_margin"
            ]
        ),
    )
    print(
        "Distribución de rangos:",
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
            "R@1=",
            row["recall_at_1"],
            "MRR=",
            row["mrr"],
            "nDCG@10=",
            row["ndcg_at_10"],
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
