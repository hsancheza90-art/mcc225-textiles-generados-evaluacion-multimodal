"""Valida el contrato reproducible de las ablaciones cromáticas E4."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from metricas_retrieval_v2 import evaluate_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABLATIONS_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "ablaciones_v2.json"
)

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

TEXT_INDEX_PATH = (
    EMBEDDINGS_DIRECTORY
    / "index_textos_unicos_v2.csv"
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

STRUCTURAL_FIELDS = (
    "pattern_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
)

CONDITION_IDS = (
    "original_image_full_caption",
    "grayscale_image_full_caption",
    "original_image_caption_without_color",
    "grayscale_image_caption_without_color",
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def split_ids(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def structural_signature(
    record: dict[str, str],
) -> tuple[str, ...]:
    return tuple(
        record[field]
        for field in STRUCTURAL_FIELDS
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


def main() -> None:
    contract = load_json(
        ABLATIONS_CONFIG_PATH
    )

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    embeddings_config = load_json(
        EMBEDDINGS_CONFIG_PATH
    )

    full_fields, full_rows = load_csv(
        FULL_CAPTIONS_PATH
    )

    colorless_fields, colorless_rows = (
        load_csv(
            COLORLESS_CAPTIONS_PATH
        )
    )

    image_fields, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    text_fields, text_rows = load_csv(
        TEXT_INDEX_PATH
    )

    usage_fields, usage_rows = load_csv(
        TEXT_USAGE_PATH
    )

    _, e1_rows = load_csv(
        E1_RESULTS_PATH
    )

    e1_summary = load_json(
        E1_SUMMARY_PATH
    )

    assert contract["schema_version"] == "1.0"
    assert contract["dataset_version"] == "v2"
    assert contract["experiment_id"] == "E4"

    assert (
        contract["experiment_name"]
        == "color_ablations"
    )

    experiments = {
        record["experiment_id"]: record
        for record in experiment[
            "experiments"
        ]
    }

    assert experiments["E4"] == {
        "experiment_id": "E4",
        "name": "color_ablations",
        "conditions": list(
            CONDITION_IDS
        ),
    }

    condition_records = contract[
        "conditions"
    ]

    assert tuple(
        record["condition_id"]
        for record in condition_records
    ) == CONDITION_IDS

    expected_condition_mapping = {
        "original_image_full_caption": (
            "original",
            "full_caption",
            7,
        ),
        "grayscale_image_full_caption": (
            "grayscale",
            "full_caption",
            7,
        ),
        "original_image_caption_without_color": (
            "original",
            "caption_without_color",
            1,
        ),
        "grayscale_image_caption_without_color": (
            "grayscale",
            "caption_without_color",
            1,
        ),
    }

    for record in condition_records:
        condition_id = record[
            "condition_id"
        ]

        expected = (
            expected_condition_mapping[
                condition_id
            ]
        )

        actual = (
            record["image_variant"],
            record["text_variant"],
            int(
                record[
                    "raw_queries_per_group"
                ]
            ),
        )

        assert actual == expected

    primary = contract[
        "primary_structural_evaluation"
    ]

    assert (
        primary["unit_of_analysis"]
        == "colorless_caption_group"
    )

    assert (
        primary["group_id_field"]
        == "colorless_caption_id"
    )

    assert primary["group_count"] == 40
    assert primary["gallery_count"] == 56

    assert (
        primary[
            "relevant_images_per_group"
        ]
        == 7
    )

    assert tuple(
        primary[
            "structural_signature_fields"
        ]
    ) == STRUCTURAL_FIELDS

    assert (
        primary[
            "full_caption_queries_per_group"
        ]
        == 7
    )

    assert (
        primary[
            "caption_without_color_queries_per_group"
        ]
        == 1
    )

    assert (
        primary[
            "raw_query_condition_rows"
        ]
        == 640
    )

    assert (
        primary[
            "group_condition_rows"
        ]
        == 160
    )

    assert (
        primary[
            "equal_weight_per_group"
        ]
        is True
    )

    metric_config = primary["metrics"]

    assert metric_config["primary"] == [
        "structural_hit_at_1",
        "structural_mrr",
        "structural_ndcg_at_10",
    ]

    assert metric_config[
        "supplementary"
    ] == [
        "structural_hit_at_5",
        "structural_fractional_recall_at_5",
        "best_relevant_margin",
    ]

    assert (
        metric_config["ndcg_cutoff"]
        == 10
    )

    assert (
        metric_config[
            "hit_at_k_definition"
        ]
        == (
            "one_if_at_least_one_relevant_"
            "image_appears_in_top_k_else_zero"
        )
    )

    assert (
        metric_config[
            "fractional_recall_at_5_definition"
        ]
        == (
            "number_of_relevant_images_in_"
            "top_5_divided_by_total_relevant_"
            "images"
        )
    )

    assert (
        metric_config[
            "fractional_recall_at_5_denominator"
        ]
        == 7
    )

    assert (
        metric_config[
            "best_relevant_margin_definition"
        ]
        == (
            "maximum_relevant_score_minus_"
            "maximum_nonrelevant_score"
        )
    )

    assert metric_config[
        "legacy_metric_mapping"
    ] == {
        "metricas_retrieval_v2.recall_at_1": (
            "structural_hit_at_1"
        ),
        "metricas_retrieval_v2.recall_at_5": (
            "structural_hit_at_5"
        ),
    }

    synthetic_scores = np.asarray(
        [
            12.0,
            11.0,
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
        ],
        dtype=np.float64,
    )

    synthetic_relevant = {
        0,
        2,
        4,
        6,
        8,
        10,
        11,
    }

    synthetic_keys = [
        f"IMG_{index:02d}"
        for index in range(12)
    ]

    synthetic_result = evaluate_query(
        scores=synthetic_scores,
        relevant_indices=synthetic_relevant,
        candidate_keys=synthetic_keys,
    )

    synthetic_ranking = [
        int(index)
        for index in synthetic_result[
            "ranking_indices"
        ]
    ]

    assert synthetic_ranking == list(
        range(12)
    )

    assert (
        synthetic_result["recall_at_1"]
        == 1.0
    )

    assert (
        synthetic_result["recall_at_5"]
        == 1.0
    )

    assert (
        synthetic_result["mrr"]
        == 1.0
    )

    assert (
        "positive_margin"
        not in synthetic_result
    )

    relevant_in_top_5 = sum(
        candidate_index
        in synthetic_relevant
        for candidate_index
        in synthetic_ranking[:5]
    )

    synthetic_fractional_recall_at_5 = (
        relevant_in_top_5
        / len(synthetic_relevant)
    )

    assert abs(
        synthetic_fractional_recall_at_5
        - (3.0 / 7.0)
    ) < 1e-12

    relevant_scores = [
        float(
            synthetic_scores[index]
        )
        for index in synthetic_relevant
    ]

    nonrelevant_scores = [
        float(
            synthetic_scores[index]
        )
        for index in range(
            synthetic_scores.size
        )
        if index
        not in synthetic_relevant
    ]

    synthetic_best_relevant_margin = (
        max(relevant_scores)
        - max(nonrelevant_scores)
    )

    assert abs(
        synthetic_best_relevant_margin
        - 1.0
    ) < 1e-12

    assert abs(
        float(
            synthetic_result[
                "ndcg_at_10"
            ]
        )
        - 0.6930226460117498
    ) < 1e-12

    secondary = contract[
        "secondary_exact_evaluation"
    ]

    assert secondary[
        "conditions"
    ] == [
        "original_image_full_caption",
        "grayscale_image_full_caption",
    ]

    assert (
        secondary[
            "query_count_per_condition"
        ]
        == 280
    )

    assert (
        secondary[
            "query_condition_rows"
        ]
        == 560
    )

    assert len(full_rows) == 280
    assert len(colorless_rows) == 40
    assert len(image_rows) == 56
    assert len(text_rows) == 494
    assert len(usage_rows) == 600
    assert len(e1_rows) == 280

    assert {
        "caption_id",
        "colorless_caption_id",
        "structure_id",
        "image_id",
        "semantic_id",
        "template_id",
        "is_canonical",
        "split",
        "pattern_id",
        "palette_id",
        "motif",
        "orientation",
        "composition",
        "symmetry",
        "caption_text",
        "caption_sha256",
    }.issubset(
        set(full_fields)
    )

    assert {
        "colorless_caption_id",
        "structure_id",
        "template_id",
        "is_canonical",
        "pattern_id",
        "motif",
        "orientation",
        "composition",
        "symmetry",
        "caption_text",
        "caption_sha256",
        "relevant_image_ids",
        "relevant_semantic_ids",
        "relevant_count",
    }.issubset(
        set(colorless_fields)
    )

    assert {
        "image_row_index",
        "image_id",
        "semantic_id",
        "split",
        "pattern_id",
        "palette_id",
        "motif",
        "orientation",
        "composition",
        "symmetry",
    }.issubset(
        set(image_fields)
    )

    assert {
        "text_row_index",
        "text_sha256",
    }.issubset(
        set(text_fields)
    )

    assert {
        "source_id",
        "record_id",
        "text_row_index",
        "text_sha256",
    }.issubset(
        set(usage_fields)
    )

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
        ): {
            "text_row_index": int(
                row["text_row_index"]
            ),
            "text_sha256": (
                row["text_sha256"]
            ),
        }
        for row in usage_rows
    }

    assert len(text_hash_by_row) == 494
    assert len(usage_map) == 600

    usage_counts = Counter(
        row["source_id"]
        for row in usage_rows
    )

    assert usage_counts == {
        "positivos": 280,
        "negativos_dificiles": 280,
        "sin_color": 40,
    }

    def find_source_record_lists(
        value,
        current_path=(),
    ):
        matches = []

        if isinstance(value, dict):
            for key, child in value.items():
                matches.extend(
                    find_source_record_lists(
                        child,
                        current_path + (key,),
                    )
                )

            return matches

        if isinstance(value, list):
            dictionaries_only = bool(value)

            if dictionaries_only:
                dictionaries_only = all(
                    isinstance(item, dict)
                    for item in value
                )

            if dictionaries_only:
                contains_source_id = all(
                    "source_id" in item
                    for item in value
                )

                if contains_source_id:
                    return [
                        (
                            current_path,
                            value,
                        )
                    ]

            for index, child in enumerate(value):
                matches.extend(
                    find_source_record_lists(
                        child,
                        current_path + (index,),
                    )
                )

        return matches


    source_record_lists = (
        find_source_record_lists(
            embeddings_config
        )
    )

    compatible_source_lists = []

    expected_source_ids = {
        "positivos",
        "negativos_dificiles",
        "sin_color",
    }

    for source_path, records in (
        source_record_lists
    ):
        source_ids = {
            str(record["source_id"])
            for record in records
        }

        if source_ids == expected_source_ids:
            compatible_source_lists.append(
                (
                    source_path,
                    records,
                )
            )

    assert len(
        compatible_source_lists
    ) == 1, (
        "No se encontró una única lista "
        "compatible de fuentes textuales. "
        f"Candidatas: {source_record_lists}"
    )

    (
        embedding_sources_path,
        embedding_source_records,
    ) = compatible_source_lists[0]

    embedding_sources = {
        record["source_id"]: record
        for record
        in embedding_source_records
    }

    readable_source_path = ".".join(
        str(component)
        for component
        in embedding_sources_path
    )

    print(
        "Ruta de fuentes textuales:",
        readable_source_path,
    )

    print(
        "Fuentes textuales:",
        sorted(
            embedding_sources
        ),
    )

    assert (
        embedding_sources[
            "sin_color"
        ][
            "record_id_field"
        ]
        == "colorless_caption_id"
    )

    assert (
        embedding_sources[
            "sin_color"
        ][
            "path"
        ]
        == (
            "data/v2/"
            "captions_sin_color_v2.csv"
        )
    )

    full_by_group: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in full_rows:
        full_by_group[
            row["colorless_caption_id"]
        ].append(row)

        usage_key = (
            "positivos",
            row["caption_id"],
        )

        assert usage_key in usage_map

        usage = usage_map[usage_key]

        assert (
            usage["text_sha256"]
            == row["caption_sha256"]
        )

        assert (
            text_hash_by_row[
                usage["text_row_index"]
            ]
            == row["caption_sha256"]
        )

        assert (
            sha256_text(
                row["caption_text"]
            )
            == row["caption_sha256"]
        )

    assert len(full_by_group) == 40

    assert Counter(
        len(rows)
        for rows in full_by_group.values()
    ) == {
        7: 40
    }

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

    image_by_id = {
        row["image_id"]: row
        for row in ordered_images
    }

    image_by_semantic_id = {
        row["semantic_id"]: row
        for row in ordered_images
    }

    images_by_signature: dict[
        tuple[str, ...],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in ordered_images:
        images_by_signature[
            structural_signature(row)
        ].append(row)

    assert len(image_by_id) == 56
    assert len(image_by_semantic_id) == 56
    assert len(images_by_signature) == 8

    assert Counter(
        len(rows)
        for rows in images_by_signature.values()
    ) == {
        7: 8
    }

    group_regime_counts = Counter()
    structure_id_counts = Counter()
    template_counts = Counter()
    canonical_counts = Counter()

    for colorless in colorless_rows:
        group_id = colorless[
            "colorless_caption_id"
        ]

        assert group_id in full_by_group

        usage_key = (
            "sin_color",
            group_id,
        )

        assert usage_key in usage_map

        usage = usage_map[usage_key]

        assert (
            usage["text_sha256"]
            == colorless[
                "caption_sha256"
            ]
        )

        assert (
            text_hash_by_row[
                usage["text_row_index"]
            ]
            == colorless[
                "caption_sha256"
            ]
        )

        assert (
            sha256_text(
                colorless["caption_text"]
            )
            == colorless[
                "caption_sha256"
            ]
        )

        full_group = full_by_group[
            group_id
        ]

        relevant_image_ids = set(
            split_ids(
                colorless[
                    "relevant_image_ids"
                ]
            )
        )

        relevant_semantic_ids = set(
            split_ids(
                colorless[
                    "relevant_semantic_ids"
                ]
            )
        )

        assert int(
            colorless["relevant_count"]
        ) == 7

        assert len(
            relevant_image_ids
        ) == 7

        assert len(
            relevant_semantic_ids
        ) == 7

        assert relevant_image_ids == {
            row["image_id"]
            for row in full_group
        }

        assert relevant_semantic_ids == {
            row["semantic_id"]
            for row in full_group
        }

        signature = structural_signature(
            colorless
        )

        signature_images = (
            images_by_signature[
                signature
            ]
        )

        assert {
            row["image_id"]
            for row in signature_images
        } == relevant_image_ids

        assert {
            row["semantic_id"]
            for row in signature_images
        } == relevant_semantic_ids

        assert {
            row["palette_id"]
            for row in signature_images
        } == {
            row["palette_id"]
            for row in full_group
        }

        for full_caption in full_group:
            assert (
                full_caption[
                    "structure_id"
                ]
                == colorless[
                    "structure_id"
                ]
            )

            assert (
                structural_signature(
                    full_caption
                )
                == signature
            )

        split_composition = Counter(
            image_by_id[
                image_id
            ]["split"]
            for image_id
            in relevant_image_ids
        )

        if split_composition == {
            "id": 5,
            "ood_palette": 2,
        }:
            regime = "base_pattern"
        elif split_composition == {
            "ood_pattern": 5,
            "ood_both": 2,
        }:
            regime = "heldout_pattern"
        else:
            raise AssertionError(
                f"{group_id}: composición "
                f"de splits inesperada: "
                f"{dict(split_composition)}"
            )

        group_regime_counts[
            regime
        ] += 1

        structure_id_counts[
            colorless["structure_id"]
        ] += 1

        template_counts[
            colorless["template_id"]
        ] += 1

        canonical_counts[
            colorless[
                "is_canonical"
            ].strip().lower()
        ] += 1

    assert group_regime_counts == {
        "base_pattern": 30,
        "heldout_pattern": 10,
    }

    assert len(
        structure_id_counts
    ) == 8

    assert set(
        structure_id_counts.values()
    ) == {5}

    assert template_counts == {
        "TPL_01": 8,
        "TPL_02": 8,
        "TPL_03": 8,
        "TPL_04": 8,
        "TPL_05": 8,
    }

    assert canonical_counts == {
        "true": 8,
        "false": 32,
    }

    original_embeddings = np.load(
        ORIGINAL_IMAGE_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    grayscale_embeddings = np.load(
        GRAYSCALE_IMAGE_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    text_embeddings = np.load(
        TEXT_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    assert original_embeddings.shape == (
        56,
        512,
    )

    assert grayscale_embeddings.shape == (
        56,
        512,
    )

    assert text_embeddings.shape == (
        494,
        512,
    )

    validate_norms(
        original_embeddings,
        "original",
    )

    validate_norms(
        grayscale_embeddings,
        "grayscale",
    )

    validate_norms(
        text_embeddings,
        "textos",
    )

    image_cosines = np.sum(
        original_embeddings
        * grayscale_embeddings,
        axis=1,
    )

    assert (
        float(image_cosines.min())
        < float(image_cosines.mean())
    )

    assert (
        float(image_cosines.mean())
        < float(image_cosines.max())
    )

    assert e1_summary[
        "evaluation_valid"
    ] is True

    assert e1_summary[
        "experiment_id"
    ] == "E1"

    assert e1_summary[
        "counts"
    ][
        "queries"
    ] == 280

    assert e1_summary[
        "counts"
    ][
        "gallery_images"
    ] == 56

    assert {
        row["caption_id"]
        for row in e1_rows
    } == {
        row["caption_id"]
        for row in full_rows
    }

    raw_rows = (
        280
        + 280
        + 40
        + 40
    )

    group_rows = (
        40
        * len(CONDITION_IDS)
    )

    secondary_rows = (
        280
        * 2
    )

    assert raw_rows == 640
    assert group_rows == 160
    assert secondary_rows == 560

    paired_comparisons = contract[
        "paired_comparisons"
    ]

    assert len(
        paired_comparisons
    ) == 4

    comparison_ids = {
        row["comparison_id"]
        for row in paired_comparisons
    }

    assert comparison_ids == {
        "grayscale_effect_with_full_caption",
        "remove_text_color_with_original_image",
        "remove_text_color_with_grayscale_image",
        "remove_visual_and_text_color",
    }

    print("=" * 80)
    print("VALIDACIÓN DEL CONTRATO E4 SUPERADA")
    print("=" * 80)
    print("Condiciones:", len(CONDITION_IDS))
    print("Grupos estructurales:", 40)
    print("Estructuras:", len(structure_id_counts))
    print("Templates por estructura: 5")
    print("Imágenes relevantes por grupo: 7")
    print(
        "Consultas crudas por condición:",
        {
            "original_image_full_caption": 280,
            "grayscale_image_full_caption": 280,
            "original_image_caption_without_color": 40,
            "grayscale_image_caption_without_color": 40,
        },
    )
    print(
        "Filas crudas condición-consulta:",
        raw_rows,
    )
    print(
        "Filas estructurales comparables:",
        group_rows,
    )
    print(
        "Filas de evaluación exacta secundaria:",
        secondary_rows,
    )
    print(
        "Regímenes:",
        dict(
            sorted(
                group_regime_counts.items()
            )
        ),
    )
    print(
        "Templates:",
        dict(
            sorted(
                template_counts.items()
            )
        ),
    )
    print(
        "Canónicos:",
        dict(
            sorted(
                canonical_counts.items()
            )
        ),
    )
    print(
        "Coseno original–gris mínimo:",
        format(
            float(image_cosines.min()),
            ".12f",
        ),
    )
    print(
        "Coseno original–gris medio:",
        format(
            float(image_cosines.mean()),
            ".12f",
        ),
    )
    print(
        "Coseno original–gris máximo:",
        format(
            float(image_cosines.max()),
            ".12f",
        ),
    )
    print("E1 disponible como ancla exacta: True")
    print(
        "Semántica legacy recall_at_k = Hit@K:",
        True,
    )
    print(
        "Recall fraccional@5 sintético:",
        format(
            synthetic_fractional_recall_at_5,
            ".12f",
        ),
    )
    print(
        "Best relevant margin sintético:",
        format(
            synthetic_best_relevant_margin,
            ".12f",
        ),
    )
    print(
        "positive_margin ausente con "
        "relevancia múltiple:",
        "positive_margin"
        not in synthetic_result,
    )
    print("No se generaron resultados experimentales.")
    print("Contrato E4 válido: True")


if __name__ == "__main__":
    main()
