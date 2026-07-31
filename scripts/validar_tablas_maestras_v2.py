"""Valida independientemente las tablas maestras de E1 a E4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "tablas_maestras_v2.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "tablas_maestras"
)

CATALOG_PATH = (
    OUTPUT_DIRECTORY
    / "catalogo_experimentos_v2.csv"
)

METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "metricas_maestras_v2.csv"
)

COMPARISONS_PATH = (
    OUTPUT_DIRECTORY
    / "comparaciones_maestras_v2.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "resumen_tablas_maestras_v2.json"
)

CATALOG_FIELDS = (
    "catalog_row_index",
    "experiment_id",
    "experiment_name",
    "dataset_version",
    "evaluation_valid",
    "comparability_families",
    "summary_path",
    "summary_sha256",
    "metric_row_count",
    "comparison_row_count",
    "source_counts_json",
)

METRIC_FIELDS = (
    "metric_row_index",
    "comparability_family",
    "experiment_id",
    "experiment_name",
    "source_section",
    "condition",
    "condition_label",
    "metric_name",
    "metric_value",
    "sample_count",
    "sample_unit",
    "relevance_type",
    "candidate_or_gallery_count",
    "anchor_id",
    "directly_comparable_within_family",
    "source_summary_path",
)

COMPARISON_FIELDS = (
    "comparison_row_index",
    "comparability_family",
    "experiment_id",
    "experiment_name",
    "source_section",
    "comparison_id",
    "minuend",
    "subtrahend",
    "statistic_name",
    "statistic_value",
    "sample_count",
    "sample_unit",
    "source_artifact_path",
)

CONDITION_LABELS = {
    "openclip": "OpenCLIP",
    "random": "Aleatorio",
    "color_histogram": "Histograma HSV",
    "openclip_hard_negatives": (
        "OpenCLIP con negativos difíciles"
    ),
    "original_image_full_caption": (
        "Imagen original + caption completo"
    ),
    "grayscale_image_full_caption": (
        "Imagen en gris + caption completo"
    ),
    "original_image_caption_without_color": (
        "Imagen original + caption sin color"
    ),
    "grayscale_image_caption_without_color": (
        "Imagen en gris + caption sin color"
    ),
}

SAMPLE_UNITS = {
    "global_exact_retrieval": (
        "caption_query"
    ),
    "hard_negative_forced_choice": (
        "image_query"
    ),
    "structural_multi_relevance": (
        "structural_caption_group"
    ),
}

GLOBAL_RECORD_ORDER = (
    (
        "E1",
        "overall_metrics",
        "openclip",
    ),
    (
        "E3",
        "overall_metrics",
        "openclip",
    ),
    (
        "E3",
        "overall_metrics",
        "random",
    ),
    (
        "E3",
        "overall_metrics",
        "color_histogram",
    ),
    (
        "E4",
        "exact_overall_metrics",
        "original_image_full_caption",
    ),
    (
        "E4",
        "exact_overall_metrics",
        "grayscale_image_full_caption",
    ),
)

STRUCTURAL_CONDITION_ORDER = (
    "original_image_full_caption",
    "grayscale_image_full_caption",
    "original_image_caption_without_color",
    "grayscale_image_caption_without_color",
)

E3_COMPARISON_ORDER = (
    (
        "openclip_minus_random",
        "openclip",
        "random",
    ),
    (
        "color_histogram_minus_random",
        "color_histogram",
        "random",
    ),
    (
        "openclip_minus_color_histogram",
        "openclip",
        "color_histogram",
    ),
)

E3_COMPARISON_METRICS = (
    "recall_at_1",
    "mrr",
    "ndcg_at_10",
    "positive_margin",
)

E4_COMPARISON_ORDER = (
    "grayscale_effect_with_full_caption",
    "remove_text_color_with_original_image",
    "remove_text_color_with_grayscale_image",
    "remove_visual_and_text_color",
)

E4_DELTA_METRICS = (
    "delta_structural_hit_at_1",
    "delta_structural_hit_at_5",
    (
        "delta_structural_fractional_"
        "recall_at_5"
    ),
    "delta_structural_mrr",
    "delta_structural_ndcg_at_10",
    "delta_best_relevant_margin",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def format_float(value: float) -> str:
    numeric = float(value)

    assert math.isfinite(numeric)

    return format(
        numeric,
        ".12f",
    )


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def resolve_metric_record(
    summaries: dict[str, dict[str, Any]],
    experiment_id: str,
    section: str,
    condition: str,
) -> dict[str, Any]:
    section_record = summaries[
        experiment_id
    ][section]

    if (
        experiment_id == "E1"
        and section == "overall_metrics"
    ):
        assert condition == "openclip"

        return section_record

    return section_record[
        condition
    ]


def compare_rows_exactly(
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    context: str,
) -> None:
    assert len(actual_rows) == len(
        expected_rows
    ), (
        f"{context}: filas actuales "
        f"{len(actual_rows)}, esperadas "
        f"{len(expected_rows)}."
    )

    for row_index, (
        actual,
        expected,
    ) in enumerate(
        zip(
            actual_rows,
            expected_rows,
            strict=True,
        )
    ):
        assert actual == expected, (
            f"{context}[{row_index}] difiere.\n"
            f"Actual: {actual}\n"
            f"Esperado: {expected}"
        )


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
    )

    catalog_fields, catalog_rows = (
        load_csv(
            CATALOG_PATH
        )
    )

    metric_fields, metric_rows = load_csv(
        METRICS_PATH
    )

    (
        comparison_fields,
        comparison_rows,
    ) = load_csv(
        COMPARISONS_PATH
    )

    summary = load_json(
        SUMMARY_PATH
    )

    assert tuple(
        catalog_fields
    ) == CATALOG_FIELDS

    assert tuple(
        metric_fields
    ) == METRIC_FIELDS

    assert tuple(
        comparison_fields
    ) == COMPARISON_FIELDS

    assert len(catalog_rows) == 4
    assert len(metric_rows) == 62
    assert len(comparison_rows) == 38

    assert [
        int(row["catalog_row_index"])
        for row in catalog_rows
    ] == list(range(4))

    assert [
        int(row["metric_row_index"])
        for row in metric_rows
    ] == list(range(62))

    assert [
        int(
            row["comparison_row_index"]
        )
        for row in comparison_rows
    ] == list(range(38))

    summary_paths = {
        experiment_id: (
            PROJECT_ROOT
            / relative_path
        )
        for experiment_id, relative_path
        in contract[
            "input_summaries"
        ].items()
    }

    summaries = {
        experiment_id: load_json(path)
        for experiment_id, path
        in summary_paths.items()
    }

    for experiment_id in (
        "E1",
        "E2",
        "E3",
        "E4",
    ):
        experiment_summary = (
            summaries[experiment_id]
        )

        assert (
            experiment_summary[
                "experiment_id"
            ]
            == experiment_id
        )

        assert (
            experiment_summary[
                "evaluation_valid"
            ]
            is True
        )

    comparison_source_paths = {
        experiment_id: (
            PROJECT_ROOT
            / relative_path
        )
        for experiment_id, relative_path
        in contract[
            "input_comparison_tables"
        ].items()
    }

    _, e3_source_rows = load_csv(
        comparison_source_paths["E3"]
    )

    _, e4_source_rows = load_csv(
        comparison_source_paths["E4"]
    )

    assert len(e3_source_rows) == 3
    assert len(e4_source_rows) == 12

    families = {
        record["family_id"]: record
        for record in contract[
            "comparability_families"
        ]
    }

    assert set(families) == {
        "global_exact_retrieval",
        "hard_negative_forced_choice",
        "structural_multi_relevance",
    }

    anchor_map = {}

    for anchor in contract[
        "anchor_equivalences"
    ]:
        for record in anchor[
            "records"
        ]:
            anchor_map[
                (
                    record[
                        "experiment_id"
                    ],
                    record["section"],
                    record["condition"],
                )
            ] = anchor["anchor_id"]

    expected_metric_rows: list[
        dict[str, str]
    ] = []

    global_family = families[
        "global_exact_retrieval"
    ]

    for (
        experiment_id,
        section,
        condition,
    ) in GLOBAL_RECORD_ORDER:
        record = resolve_metric_record(
            summaries,
            experiment_id,
            section,
            condition,
        )

        sample_count = int(
            round(
                float(
                    record["query_count"]
                )
            )
        )

        for metric_name in global_family[
            "metrics"
        ]:
            expected_metric_rows.append(
                {
                    "metric_row_index": str(
                        len(
                            expected_metric_rows
                        )
                    ),
                    "comparability_family": (
                        "global_exact_retrieval"
                    ),
                    "experiment_id": (
                        experiment_id
                    ),
                    "experiment_name": (
                        summaries[
                            experiment_id
                        ]["experiment_name"]
                    ),
                    "source_section": (
                        section
                    ),
                    "condition": condition,
                    "condition_label": (
                        CONDITION_LABELS[
                            condition
                        ]
                    ),
                    "metric_name": (
                        metric_name
                    ),
                    "metric_value": (
                        format_float(
                            record[
                                metric_name
                            ]
                        )
                    ),
                    "sample_count": str(
                        sample_count
                    ),
                    "sample_unit": (
                        SAMPLE_UNITS[
                            "global_exact_retrieval"
                        ]
                    ),
                    "relevance_type": (
                        global_family[
                            "relevance_type"
                        ]
                    ),
                    "candidate_or_gallery_count": (
                        "56"
                    ),
                    "anchor_id": (
                        anchor_map.get(
                            (
                                experiment_id,
                                section,
                                condition,
                            ),
                            "",
                        )
                    ),
                    "directly_comparable_within_family": (
                        str(
                            global_family[
                                "directly_comparable_within_family"
                            ]
                        ).lower()
                    ),
                    "source_summary_path": (
                        summary_paths[
                            experiment_id
                        ]
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                }
            )

    hard_family = families[
        "hard_negative_forced_choice"
    ]

    e2_record = summaries["E2"][
        "overall_metrics"
    ]

    for metric_name in hard_family[
        "metrics"
    ]:
        expected_metric_rows.append(
            {
                "metric_row_index": str(
                    len(
                        expected_metric_rows
                    )
                ),
                "comparability_family": (
                    "hard_negative_forced_choice"
                ),
                "experiment_id": "E2",
                "experiment_name": (
                    summaries["E2"][
                        "experiment_name"
                    ]
                ),
                "source_section": (
                    "overall_metrics"
                ),
                "condition": (
                    "openclip_hard_negatives"
                ),
                "condition_label": (
                    CONDITION_LABELS[
                        "openclip_hard_negatives"
                    ]
                ),
                "metric_name": (
                    metric_name
                ),
                "metric_value": (
                    format_float(
                        e2_record[
                            metric_name
                        ]
                    )
                ),
                "sample_count": str(
                    int(
                        round(
                            float(
                                e2_record[
                                    "query_count"
                                ]
                            )
                        )
                    )
                ),
                "sample_unit": (
                    SAMPLE_UNITS[
                        "hard_negative_forced_choice"
                    ]
                ),
                "relevance_type": (
                    hard_family[
                        "relevance_type"
                    ]
                ),
                "candidate_or_gallery_count": (
                    "5"
                ),
                "anchor_id": "",
                "directly_comparable_within_family": (
                    str(
                        hard_family[
                            "directly_comparable_within_family"
                        ]
                    ).lower()
                ),
                "source_summary_path": (
                    summary_paths["E2"]
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
            }
        )

    structural_family = families[
        "structural_multi_relevance"
    ]

    for condition in (
        STRUCTURAL_CONDITION_ORDER
    ):
        record = summaries["E4"][
            "structural_overall_metrics"
        ][condition]

        for metric_name in structural_family[
            "metrics"
        ]:
            expected_metric_rows.append(
                {
                    "metric_row_index": str(
                        len(
                            expected_metric_rows
                        )
                    ),
                    "comparability_family": (
                        "structural_multi_relevance"
                    ),
                    "experiment_id": "E4",
                    "experiment_name": (
                        summaries["E4"][
                            "experiment_name"
                        ]
                    ),
                    "source_section": (
                        "structural_overall_metrics"
                    ),
                    "condition": (
                        condition
                    ),
                    "condition_label": (
                        CONDITION_LABELS[
                            condition
                        ]
                    ),
                    "metric_name": (
                        metric_name
                    ),
                    "metric_value": (
                        format_float(
                            record[
                                metric_name
                            ]
                        )
                    ),
                    "sample_count": str(
                        int(
                            record[
                                "group_count"
                            ]
                        )
                    ),
                    "sample_unit": (
                        SAMPLE_UNITS[
                            "structural_multi_relevance"
                        ]
                    ),
                    "relevance_type": (
                        structural_family[
                            "relevance_type"
                        ]
                    ),
                    "candidate_or_gallery_count": (
                        "56"
                    ),
                    "anchor_id": "",
                    "directly_comparable_within_family": (
                        str(
                            structural_family[
                                "directly_comparable_within_family"
                            ]
                        ).lower()
                    ),
                    "source_summary_path": (
                        summary_paths["E4"]
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                }
            )

    assert len(
        expected_metric_rows
    ) == 62

    compare_rows_exactly(
        metric_rows,
        expected_metric_rows,
        "metricas_maestras",
    )

    assert Counter(
        row["comparability_family"]
        for row in metric_rows
    ) == {
        "global_exact_retrieval": 30,
        "hard_negative_forced_choice": 8,
        "structural_multi_relevance": 24,
    }

    assert Counter(
        row["experiment_id"]
        for row in metric_rows
    ) == {
        "E1": 5,
        "E2": 8,
        "E3": 15,
        "E4": 34,
    }

    expected_comparison_rows: list[
        dict[str, str]
    ] = []

    e3_metrics = summaries["E3"][
        "overall_metrics"
    ]

    for (
        comparison_id,
        minuend,
        subtrahend,
    ) in E3_COMPARISON_ORDER:
        for metric_name in (
            E3_COMPARISON_METRICS
        ):
            value = (
                float(
                    e3_metrics[
                        minuend
                    ][metric_name]
                )
                - float(
                    e3_metrics[
                        subtrahend
                    ][metric_name]
                )
            )

            expected_comparison_rows.append(
                {
                    "comparison_row_index": str(
                        len(
                            expected_comparison_rows
                        )
                    ),
                    "comparability_family": (
                        "global_exact_retrieval"
                    ),
                    "experiment_id": "E3",
                    "experiment_name": (
                        summaries["E3"][
                            "experiment_name"
                        ]
                    ),
                    "source_section": (
                        "overall_metrics"
                    ),
                    "comparison_id": (
                        comparison_id
                    ),
                    "minuend": minuend,
                    "subtrahend": (
                        subtrahend
                    ),
                    "statistic_name": (
                        f"delta_{metric_name}"
                    ),
                    "statistic_value": (
                        format_float(value)
                    ),
                    "sample_count": "280",
                    "sample_unit": (
                        "caption_query"
                    ),
                    "source_artifact_path": (
                        comparison_source_paths[
                            "E3"
                        ]
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                }
            )

    for statistic_name in (
        "mean_paired_difference",
        "pairwise_win_rate",
    ):
        expected_comparison_rows.append(
            {
                "comparison_row_index": str(
                    len(
                        expected_comparison_rows
                    )
                ),
                "comparability_family": (
                    "hard_negative_forced_choice"
                ),
                "experiment_id": "E2",
                "experiment_name": (
                    summaries["E2"][
                        "experiment_name"
                    ]
                ),
                "source_section": (
                    "overall_metrics"
                ),
                "comparison_id": (
                    "positive_vs_hard_negatives"
                ),
                "minuend": (
                    "positive_candidate"
                ),
                "subtrahend": (
                    "hard_negative_candidate"
                ),
                "statistic_name": (
                    statistic_name
                ),
                "statistic_value": (
                    format_float(
                        e2_record[
                            statistic_name
                        ]
                    )
                ),
                "sample_count": str(
                    int(
                        summaries["E2"][
                            "counts"
                        ]["pairwise_rows"]
                    )
                ),
                "sample_unit": (
                    "positive_negative_pair"
                ),
                "source_artifact_path": (
                    summary_paths["E2"]
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
            }
        )

    e4_comparisons = summaries["E4"][
        "paired_overall_differences"
    ]

    for comparison_id in (
        E4_COMPARISON_ORDER
    ):
        record = e4_comparisons[
            comparison_id
        ]

        assert int(
            record["group_count"]
        ) == 40

        for statistic_name in (
            E4_DELTA_METRICS
        ):
            expected_comparison_rows.append(
                {
                    "comparison_row_index": str(
                        len(
                            expected_comparison_rows
                        )
                    ),
                    "comparability_family": (
                        "structural_multi_relevance"
                    ),
                    "experiment_id": "E4",
                    "experiment_name": (
                        summaries["E4"][
                            "experiment_name"
                        ]
                    ),
                    "source_section": (
                        "paired_overall_differences"
                    ),
                    "comparison_id": (
                        comparison_id
                    ),
                    "minuend": (
                        record["minuend"]
                    ),
                    "subtrahend": (
                        record["subtrahend"]
                    ),
                    "statistic_name": (
                        statistic_name
                    ),
                    "statistic_value": (
                        format_float(
                            record[
                                statistic_name
                            ]
                        )
                    ),
                    "sample_count": "40",
                    "sample_unit": (
                        "structural_caption_group"
                    ),
                    "source_artifact_path": (
                        comparison_source_paths[
                            "E4"
                        ]
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                }
            )

    assert len(
        expected_comparison_rows
    ) == 38

    compare_rows_exactly(
        comparison_rows,
        expected_comparison_rows,
        "comparaciones_maestras",
    )

    assert Counter(
        row["experiment_id"]
        for row in comparison_rows
    ) == {
        "E2": 2,
        "E3": 12,
        "E4": 24,
    }

    e3_source_by_condition = {
        row["condition"]: row
        for row in e3_source_rows
    }

    assert set(
        e3_source_by_condition
    ) == {
        "openclip",
        "random",
        "color_histogram",
    }

    for metric_name in (
        E3_COMPARISON_METRICS
    ):
        openclip_minus_random = (
            float(
                e3_metrics[
                    "openclip"
                ][metric_name]
            )
            - float(
                e3_metrics[
                    "random"
                ][metric_name]
            )
        )

        actual = float(
            e3_source_by_condition[
                "openclip"
            ][
                (
                    f"delta_{metric_name}"
                    "_vs_random"
                )
            ]
        )

        assert math.isclose(
            actual,
            openclip_minus_random,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    e4_source_overall = {
        row["comparison_id"]: row
        for row in e4_source_rows
        if (
            row["group_dimension"]
            == "overall"
            and row["group_value"]
            == "all"
        )
    }

    assert set(
        e4_source_overall
    ) == set(
        E4_COMPARISON_ORDER
    )

    for comparison_id in (
        E4_COMPARISON_ORDER
    ):
        summary_record = e4_comparisons[
            comparison_id
        ]

        csv_record = e4_source_overall[
            comparison_id
        ]

        for statistic_name in (
            E4_DELTA_METRICS
        ):
            assert math.isclose(
                float(
                    csv_record[
                        statistic_name
                    ]
                ),
                float(
                    summary_record[
                        statistic_name
                    ]
                ),
                rel_tol=0.0,
                abs_tol=1e-9,
            )

    metric_counts = Counter(
        row["experiment_id"]
        for row in metric_rows
    )

    comparison_counts = Counter(
        row["experiment_id"]
        for row in comparison_rows
    )

    experiment_families = {
        experiment_id: sorted(
            {
                row[
                    "comparability_family"
                ]
                for row in metric_rows
                if (
                    row["experiment_id"]
                    == experiment_id
                )
            }
        )
        for experiment_id in (
            "E1",
            "E2",
            "E3",
            "E4",
        )
    }

    expected_catalog_rows = []

    for row_index, experiment_id in enumerate(
        (
            "E1",
            "E2",
            "E3",
            "E4",
        )
    ):
        experiment_summary = (
            summaries[experiment_id]
        )

        experiment_summary_path = (
            summary_paths[experiment_id]
        )

        expected_catalog_rows.append(
            {
                "catalog_row_index": str(
                    row_index
                ),
                "experiment_id": (
                    experiment_id
                ),
                "experiment_name": (
                    experiment_summary[
                        "experiment_name"
                    ]
                ),
                "dataset_version": (
                    experiment_summary[
                        "dataset_version"
                    ]
                ),
                "evaluation_valid": (
                    str(
                        experiment_summary[
                            "evaluation_valid"
                        ]
                    ).lower()
                ),
                "comparability_families": (
                    "|".join(
                        experiment_families[
                            experiment_id
                        ]
                    )
                ),
                "summary_path": (
                    experiment_summary_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "summary_sha256": (
                    sha256_file(
                        experiment_summary_path
                    )
                ),
                "metric_row_count": str(
                    metric_counts[
                        experiment_id
                    ]
                ),
                "comparison_row_count": str(
                    comparison_counts[
                        experiment_id
                    ]
                ),
                "source_counts_json": (
                    compact_json(
                        experiment_summary[
                            "counts"
                        ]
                    )
                ),
            }
        )

    compare_rows_exactly(
        catalog_rows,
        expected_catalog_rows,
        "catalogo_experimentos",
    )

    anchor_rows = [
        row
        for row in metric_rows
        if (
            row["anchor_id"]
            == "original_openclip_exact"
        )
    ]

    assert len(anchor_rows) == 15

    anchor_values = {}

    for metric_name in (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
    ):
        values = [
            row["metric_value"]
            for row in anchor_rows
            if (
                row["metric_name"]
                == metric_name
            )
        ]

        assert len(values) == 3

        numeric_values = [
            float(value)
            for value in values
        ]

        reference_value = (
            numeric_values[0]
        )

        for record_index, value in enumerate(
            numeric_values[1:],
            start=1,
        ):
            assert math.isclose(
                value,
                reference_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ), (
                "Ancla maestra no equivalente "
                f"para {metric_name}, "
                f"registro {record_index}: "
                f"{value} frente a "
                f"{reference_value}."
            )

        anchor_values[
            metric_name
        ] = reference_value

    assert (
        summary["schema_version"]
        == "1.0"
    )

    assert (
        summary["contract_id"]
        == "TM1"
    )

    assert (
        summary["dataset_version"]
        == "v2"
    )

    assert (
        summary["generation_valid"]
        is True
    )

    assert summary["counts"] == {
        "experiment_catalog_rows": 4,
        "master_metric_rows": 62,
        "master_comparison_rows": 38,
    }

    assert summary[
        "metric_rows_by_family"
    ] == {
        "global_exact_retrieval": 30,
        "hard_negative_forced_choice": 8,
        "structural_multi_relevance": 24,
    }

    assert summary[
        "metric_rows_by_experiment"
    ] == {
        "E1": 5,
        "E2": 8,
        "E3": 15,
        "E4": 34,
    }

    assert summary[
        "comparison_rows_by_experiment"
    ] == {
        "E2": 2,
        "E3": 12,
        "E4": 24,
    }

    assert summary[
        "comparability_families"
    ] == [
        record["family_id"]
        for record in contract[
            "comparability_families"
        ]
    ]

    assert summary[
        "presentation_rules"
    ] == contract[
        "presentation_rules"
    ]

    anchor_summary = summary[
        "anchor_equivalence"
    ]

    assert (
        anchor_summary["anchor_id"]
        == "original_openclip_exact"
    )

    assert (
        anchor_summary["records"]
        == 3
    )

    assert (
        anchor_summary["valid"]
        is True
    )

    for metric_name, value in (
        anchor_values.items()
    ):
        assert math.isclose(
            float(
                anchor_summary[
                    "metrics"
                ][metric_name]
            ),
            value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    expected_input_paths = {
        "contract": CONTRACT_PATH,
        "E1_summary": (
            summary_paths["E1"]
        ),
        "E2_summary": (
            summary_paths["E2"]
        ),
        "E3_summary": (
            summary_paths["E3"]
        ),
        "E4_summary": (
            summary_paths["E4"]
        ),
        "E3_comparison_table": (
            comparison_source_paths[
                "E3"
            ]
        ),
        "E4_comparison_table": (
            comparison_source_paths[
                "E4"
            ]
        ),
    }

    assert set(
        summary["input_artifacts"]
    ) == set(
        expected_input_paths
    )

    for key, path in (
        expected_input_paths.items()
    ):
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

    expected_outputs = {
        "experiment_catalog": (
            CATALOG_PATH,
            4,
        ),
        "master_metrics": (
            METRICS_PATH,
            62,
        ),
        "master_comparisons": (
            COMPARISONS_PATH,
            38,
        ),
    }

    assert set(
        summary["output_artifacts"]
    ) == set(expected_outputs)

    for key, (
        path,
        row_count,
    ) in expected_outputs.items():
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
            == row_count
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )

    expected_names = {
        CATALOG_PATH.name,
        METRICS_PATH.name,
        COMPARISONS_PATH.name,
        SUMMARY_PATH.name,
    }

    actual_names = {
        path.name
        for path in OUTPUT_DIRECTORY.iterdir()
        if path.is_file()
    }

    assert actual_names == expected_names

    print("=" * 92)
    print(
        "VALIDACIÓN INDEPENDIENTE DE "
        "TABLAS MAESTRAS SUPERADA"
    )
    print("=" * 92)

    print(
        "Filas del catálogo verificadas:",
        len(catalog_rows),
    )

    print(
        "Filas de métricas reconstruidas:",
        len(metric_rows),
    )

    print(
        "Filas de comparaciones reconstruidas:",
        len(comparison_rows),
    )

    print()
    print("Métricas por familia:")

    for family_id, count in sorted(
        Counter(
            row["comparability_family"]
            for row in metric_rows
        ).items()
    ):
        print(
            f"- {family_id}: {count}"
        )

    print()
    print("Métricas por experimento:")

    for experiment_id in (
        "E1",
        "E2",
        "E3",
        "E4",
    ):
        print(
            f"- {experiment_id}:",
            metric_counts[
                experiment_id
            ],
        )

    print()
    print("Comparaciones por experimento:")

    for experiment_id in (
        "E1",
        "E2",
        "E3",
        "E4",
    ):
        print(
            f"- {experiment_id}:",
            comparison_counts[
                experiment_id
            ],
        )

    print()
    print(
        "Ancla E1 = E3 = E4:",
        True,
    )

    for metric_name, value in (
        anchor_values.items()
    ):
        print(
            f"- {metric_name}:",
            format_float(value),
        )

    print()
    print(
        "Hashes de entradas:",
        "válidos",
    )

    print(
        "Hashes de salidas:",
        "válidos",
    )

    print(
        "Resumen maestro:",
        "válido",
    )

    print(
        "Tablas maestras válidas:",
        True,
    )


if __name__ == "__main__":
    main()
