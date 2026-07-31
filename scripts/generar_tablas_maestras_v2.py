"""Genera tablas maestras auditables para los experimentos E1 a E4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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

TEMPORARY_DIRECTORY = (
    OUTPUT_DIRECTORY.with_name(
        OUTPUT_DIRECTORY.name + ".tmp"
    )
)

CATALOG_FILENAME = (
    "catalogo_experimentos_v2.csv"
)

METRICS_FILENAME = (
    "metricas_maestras_v2.csv"
)

COMPARISONS_FILENAME = (
    "comparaciones_maestras_v2.csv"
)

SUMMARY_FILENAME = (
    "resumen_tablas_maestras_v2.json"
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

FAMILY_SAMPLE_UNITS = {
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


def append_metric_rows(
    destination: list[dict[str, Any]],
    *,
    family_id: str,
    family: dict[str, Any],
    experiment_id: str,
    summary: dict[str, Any],
    summary_path: Path,
    section: str,
    condition: str,
    metric_record: dict[str, Any],
    metric_names: Iterable[str],
    sample_count: int,
    candidate_or_gallery_count: int,
    anchor_id: str,
) -> None:
    for metric_name in metric_names:
        assert metric_name in metric_record

        destination.append(
            {
                "metric_row_index": (
                    len(destination)
                ),
                "comparability_family": (
                    family_id
                ),
                "experiment_id": (
                    experiment_id
                ),
                "experiment_name": (
                    summary[
                        "experiment_name"
                    ]
                ),
                "source_section": section,
                "condition": condition,
                "condition_label": (
                    CONDITION_LABELS[
                        condition
                    ]
                ),
                "metric_name": metric_name,
                "metric_value": format_float(
                    metric_record[
                        metric_name
                    ]
                ),
                "sample_count": (
                    sample_count
                ),
                "sample_unit": (
                    FAMILY_SAMPLE_UNITS[
                        family_id
                    ]
                ),
                "relevance_type": (
                    family[
                        "relevance_type"
                    ]
                ),
                "candidate_or_gallery_count": (
                    candidate_or_gallery_count
                ),
                "anchor_id": anchor_id,
                "directly_comparable_within_family": (
                    str(
                        family[
                            "directly_comparable_within_family"
                        ]
                    ).lower()
                ),
                "source_summary_path": (
                    summary_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
            }
        )


def append_comparison_metric(
    destination: list[dict[str, Any]],
    *,
    family_id: str,
    experiment_id: str,
    experiment_name: str,
    source_section: str,
    comparison_id: str,
    minuend: str,
    subtrahend: str,
    statistic_name: str,
    statistic_value: float,
    sample_count: int,
    sample_unit: str,
    source_path: Path,
) -> None:
    destination.append(
        {
            "comparison_row_index": (
                len(destination)
            ),
            "comparability_family": (
                family_id
            ),
            "experiment_id": (
                experiment_id
            ),
            "experiment_name": (
                experiment_name
            ),
            "source_section": (
                source_section
            ),
            "comparison_id": (
                comparison_id
            ),
            "minuend": minuend,
            "subtrahend": subtrahend,
            "statistic_name": (
                statistic_name
            ),
            "statistic_value": (
                format_float(
                    statistic_value
                )
            ),
            "sample_count": (
                sample_count
            ),
            "sample_unit": (
                sample_unit
            ),
            "source_artifact_path": (
                source_path
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix()
            ),
        }
    )


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
    )

    assert (
        contract["contract_id"]
        == "TM1"
    )

    assert (
        contract["dataset_version"]
        == "v2"
    )

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
        assert (
            summaries[
                experiment_id
            ]["experiment_id"]
            == experiment_id
        )

        assert (
            summaries[
                experiment_id
            ]["evaluation_valid"]
            is True
        )

    comparison_paths = {
        experiment_id: (
            PROJECT_ROOT
            / relative_path
        )
        for experiment_id, relative_path
        in contract[
            "input_comparison_tables"
        ].items()
    }

    _, e3_comparison_rows = load_csv(
        comparison_paths["E3"]
    )

    _, e4_comparison_rows = load_csv(
        comparison_paths["E4"]
    )

    families = {
        record["family_id"]: record
        for record in contract[
            "comparability_families"
        ]
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

    metric_rows: list[
        dict[str, Any]
    ] = []

    global_family = families[
        "global_exact_retrieval"
    ]

    global_records = (
        (
            "E1",
            "overall_metrics",
            "openclip",
            summaries["E1"][
                "overall_metrics"
            ],
        ),
        (
            "E3",
            "overall_metrics",
            "openclip",
            summaries["E3"][
                "overall_metrics"
            ]["openclip"],
        ),
        (
            "E3",
            "overall_metrics",
            "random",
            summaries["E3"][
                "overall_metrics"
            ]["random"],
        ),
        (
            "E3",
            "overall_metrics",
            "color_histogram",
            summaries["E3"][
                "overall_metrics"
            ]["color_histogram"],
        ),
        (
            "E4",
            "exact_overall_metrics",
            (
                "original_image_"
                "full_caption"
            ),
            summaries["E4"][
                "exact_overall_metrics"
            ][
                "original_image_"
                "full_caption"
            ],
        ),
        (
            "E4",
            "exact_overall_metrics",
            (
                "grayscale_image_"
                "full_caption"
            ),
            summaries["E4"][
                "exact_overall_metrics"
            ][
                "grayscale_image_"
                "full_caption"
            ],
        ),
    )

    for (
        experiment_id,
        section,
        condition,
        metric_record,
    ) in global_records:
        sample_count = int(
            round(
                float(
                    metric_record[
                        "query_count"
                    ]
                )
            )
        )

        append_metric_rows(
            metric_rows,
            family_id=(
                "global_exact_retrieval"
            ),
            family=global_family,
            experiment_id=experiment_id,
            summary=summaries[
                experiment_id
            ],
            summary_path=summary_paths[
                experiment_id
            ],
            section=section,
            condition=condition,
            metric_record=metric_record,
            metric_names=global_family[
                "metrics"
            ],
            sample_count=sample_count,
            candidate_or_gallery_count=56,
            anchor_id=anchor_map.get(
                (
                    experiment_id,
                    section,
                    condition,
                ),
                "",
            ),
        )

    hard_family = families[
        "hard_negative_forced_choice"
    ]

    e2_metrics = summaries["E2"][
        "overall_metrics"
    ]

    append_metric_rows(
        metric_rows,
        family_id=(
            "hard_negative_forced_choice"
        ),
        family=hard_family,
        experiment_id="E2",
        summary=summaries["E2"],
        summary_path=summary_paths["E2"],
        section="overall_metrics",
        condition=(
            "openclip_hard_negatives"
        ),
        metric_record=e2_metrics,
        metric_names=hard_family[
            "metrics"
        ],
        sample_count=int(
            round(
                float(
                    e2_metrics[
                        "query_count"
                    ]
                )
            )
        ),
        candidate_or_gallery_count=5,
        anchor_id="",
    )

    structural_family = families[
        "structural_multi_relevance"
    ]

    for condition in (
        "original_image_full_caption",
        "grayscale_image_full_caption",
        (
            "original_image_caption_"
            "without_color"
        ),
        (
            "grayscale_image_caption_"
            "without_color"
        ),
    ):
        metric_record = summaries["E4"][
            "structural_overall_metrics"
        ][condition]

        append_metric_rows(
            metric_rows,
            family_id=(
                "structural_multi_relevance"
            ),
            family=structural_family,
            experiment_id="E4",
            summary=summaries["E4"],
            summary_path=summary_paths["E4"],
            section=(
                "structural_overall_metrics"
            ),
            condition=condition,
            metric_record=metric_record,
            metric_names=structural_family[
                "metrics"
            ],
            sample_count=int(
                metric_record[
                    "group_count"
                ]
            ),
            candidate_or_gallery_count=56,
            anchor_id="",
        )

    assert len(metric_rows) == 62

    assert Counter(
        row["comparability_family"]
        for row in metric_rows
    ) == {
        "global_exact_retrieval": 30,
        "hard_negative_forced_choice": 8,
        "structural_multi_relevance": 24,
    }

    comparison_rows: list[
        dict[str, Any]
    ] = []

    e3_metrics = summaries["E3"][
        "overall_metrics"
    ]

    e3_comparisons = (
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
            (
                "openclip_minus_"
                "color_histogram"
            ),
            "openclip",
            "color_histogram",
        ),
    )

    e3_metric_names = (
        "recall_at_1",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
    )

    for (
        comparison_id,
        minuend,
        subtrahend,
    ) in e3_comparisons:
        for metric_name in e3_metric_names:
            delta = (
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

            append_comparison_metric(
                comparison_rows,
                family_id=(
                    "global_exact_retrieval"
                ),
                experiment_id="E3",
                experiment_name=(
                    summaries["E3"][
                        "experiment_name"
                    ]
                ),
                source_section=(
                    "overall_metrics"
                ),
                comparison_id=(
                    comparison_id
                ),
                minuend=minuend,
                subtrahend=subtrahend,
                statistic_name=(
                    f"delta_{metric_name}"
                ),
                statistic_value=delta,
                sample_count=280,
                sample_unit=(
                    "caption_query"
                ),
                source_path=(
                    comparison_paths["E3"]
                ),
            )

    assert len(comparison_rows) == 12

    e3_csv_by_condition = {
        row["condition"]: row
        for row in e3_comparison_rows
    }

    assert set(
        e3_csv_by_condition
    ) == {
        "openclip",
        "random",
        "color_histogram",
    }

    for metric_name in e3_metric_names:
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

        assert_close(
            float(
                e3_csv_by_condition[
                    "openclip"
                ][
                    (
                        f"delta_{metric_name}"
                        "_vs_random"
                    )
                ]
            ),
            openclip_minus_random,
            (
                "E3 CSV openclip-random "
                f"{metric_name}"
            ),
        )

        color_minus_random = (
            float(
                e3_metrics[
                    "color_histogram"
                ][metric_name]
            )
            - float(
                e3_metrics[
                    "random"
                ][metric_name]
            )
        )

        assert_close(
            float(
                e3_csv_by_condition[
                    "color_histogram"
                ][
                    (
                        f"delta_{metric_name}"
                        "_vs_random"
                    )
                ]
            ),
            color_minus_random,
            (
                "E3 CSV color-random "
                f"{metric_name}"
            ),
        )

        color_minus_openclip = (
            float(
                e3_metrics[
                    "color_histogram"
                ][metric_name]
            )
            - float(
                e3_metrics[
                    "openclip"
                ][metric_name]
            )
        )

        assert_close(
            float(
                e3_csv_by_condition[
                    "color_histogram"
                ][
                    (
                        f"delta_{metric_name}"
                        "_vs_openclip"
                    )
                ]
            ),
            color_minus_openclip,
            (
                "E3 CSV color-openclip "
                f"{metric_name}"
            ),
        )

    for statistic_name in (
        "mean_paired_difference",
        "pairwise_win_rate",
    ):
        append_comparison_metric(
            comparison_rows,
            family_id=(
                "hard_negative_forced_choice"
            ),
            experiment_id="E2",
            experiment_name=(
                summaries["E2"][
                    "experiment_name"
                ]
            ),
            source_section=(
                "overall_metrics"
            ),
            comparison_id=(
                "positive_vs_hard_negatives"
            ),
            minuend=(
                "positive_candidate"
            ),
            subtrahend=(
                "hard_negative_candidate"
            ),
            statistic_name=(
                statistic_name
            ),
            statistic_value=float(
                e2_metrics[
                    statistic_name
                ]
            ),
            sample_count=int(
                summaries["E2"][
                    "counts"
                ]["pairwise_rows"]
            ),
            sample_unit=(
                "positive_negative_pair"
            ),
            source_path=(
                summary_paths["E2"]
            ),
        )

    assert len(comparison_rows) == 14

    e4_overall_csv = {
        row["comparison_id"]: row
        for row in e4_comparison_rows
        if (
            row["group_dimension"]
            == "overall"
            and row["group_value"]
            == "all"
        )
    }

    e4_summary_comparisons = (
        summaries["E4"][
            "paired_overall_differences"
        ]
    )

    assert set(
        e4_overall_csv
    ) == set(
        e4_summary_comparisons
    )

    e4_delta_metrics = (
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

    for comparison in contract[
        "master_comparison_rows"
    ]["E4"].get(
        "comparison_order",
        list(
            e4_summary_comparisons
        ),
    ):
        if comparison not in (
            e4_summary_comparisons
        ):
            raise AssertionError(
                "Comparación E4 desconocida: "
                f"{comparison}"
            )

    e4_comparison_order = [
        record["comparison_id"]
        for record in contract[
            "comparability_families"
        ][2]["members"]
        if False
    ]

    del e4_comparison_order

    for comparison_id in (
        "grayscale_effect_with_full_caption",
        "remove_text_color_with_original_image",
        "remove_text_color_with_grayscale_image",
        "remove_visual_and_text_color",
    ):
        record = e4_summary_comparisons[
            comparison_id
        ]

        assert int(
            record["group_count"]
        ) == 40

        csv_record = e4_overall_csv[
            comparison_id
        ]

        for statistic_name in (
            e4_delta_metrics
        ):
            assert_close(
                float(
                    csv_record[
                        statistic_name
                    ]
                ),
                float(
                    record[
                        statistic_name
                    ]
                ),
                (
                    f"E4 {comparison_id} "
                    f"{statistic_name}"
                ),
            )

            append_comparison_metric(
                comparison_rows,
                family_id=(
                    "structural_multi_relevance"
                ),
                experiment_id="E4",
                experiment_name=(
                    summaries["E4"][
                        "experiment_name"
                    ]
                ),
                source_section=(
                    "paired_overall_differences"
                ),
                comparison_id=(
                    comparison_id
                ),
                minuend=record[
                    "minuend"
                ],
                subtrahend=record[
                    "subtrahend"
                ],
                statistic_name=(
                    statistic_name
                ),
                statistic_value=float(
                    record[
                        statistic_name
                    ]
                ),
                sample_count=40,
                sample_unit=(
                    "structural_caption_group"
                ),
                source_path=(
                    comparison_paths["E4"]
                ),
            )

    assert len(comparison_rows) == 38

    assert Counter(
        row["experiment_id"]
        for row in comparison_rows
    ) == {
        "E2": 2,
        "E3": 12,
        "E4": 24,
    }

    metric_counts_by_experiment = Counter(
        row["experiment_id"]
        for row in metric_rows
    )

    comparison_counts_by_experiment = (
        Counter(
            row["experiment_id"]
            for row in comparison_rows
        )
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

    catalog_rows = []

    for catalog_row_index, experiment_id in enumerate(
        (
            "E1",
            "E2",
            "E3",
            "E4",
        )
    ):
        summary = summaries[
            experiment_id
        ]

        summary_path = summary_paths[
            experiment_id
        ]

        catalog_rows.append(
            {
                "catalog_row_index": (
                    catalog_row_index
                ),
                "experiment_id": (
                    experiment_id
                ),
                "experiment_name": (
                    summary[
                        "experiment_name"
                    ]
                ),
                "dataset_version": (
                    summary[
                        "dataset_version"
                    ]
                ),
                "evaluation_valid": (
                    str(
                        summary[
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
                    summary_path
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "summary_sha256": (
                    sha256_file(
                        summary_path
                    )
                ),
                "metric_row_count": (
                    metric_counts_by_experiment[
                        experiment_id
                    ]
                ),
                "comparison_row_count": (
                    comparison_counts_by_experiment[
                        experiment_id
                    ]
                ),
                "source_counts_json": (
                    compact_json(
                        summary["counts"]
                    )
                ),
            }
        )

    assert len(catalog_rows) == 4

    assert {
        row["experiment_id"]: int(
            row["metric_row_count"]
        )
        for row in catalog_rows
    } == {
        "E1": 5,
        "E2": 8,
        "E3": 15,
        "E4": 34,
    }

    assert {
        row["experiment_id"]: int(
            row[
                "comparison_row_count"
            ]
        )
        for row in catalog_rows
    } == {
        "E1": 0,
        "E2": 2,
        "E3": 12,
        "E4": 24,
    }

    if TEMPORARY_DIRECTORY.exists():
        shutil.rmtree(
            TEMPORARY_DIRECTORY
        )

    TEMPORARY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    catalog_path = (
        TEMPORARY_DIRECTORY
        / CATALOG_FILENAME
    )

    metrics_path = (
        TEMPORARY_DIRECTORY
        / METRICS_FILENAME
    )

    comparisons_path = (
        TEMPORARY_DIRECTORY
        / COMPARISONS_FILENAME
    )

    summary_path = (
        TEMPORARY_DIRECTORY
        / SUMMARY_FILENAME
    )

    try:
        write_csv(
            catalog_path,
            CATALOG_FIELDS,
            catalog_rows,
        )

        write_csv(
            metrics_path,
            METRIC_FIELDS,
            metric_rows,
        )

        write_csv(
            comparisons_path,
            COMPARISON_FIELDS,
            comparison_rows,
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
                float(
                    row["metric_value"]
                )
                for row in anchor_rows
                if (
                    row["metric_name"]
                    == metric_name
                )
            ]

            assert len(values) == 3

            for value in values[1:]:
                assert_close(
                    value,
                    values[0],
                    (
                        "Ancla maestra "
                        f"{metric_name}"
                    ),
                    tolerance=1e-12,
                )

            anchor_values[
                metric_name
            ] = values[0]

        input_artifacts = {
            "contract": {
                "path": (
                    CONTRACT_PATH
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "sha256": sha256_file(
                    CONTRACT_PATH
                ),
            }
        }

        for experiment_id in (
            "E1",
            "E2",
            "E3",
            "E4",
        ):
            path = summary_paths[
                experiment_id
            ]

            input_artifacts[
                f"{experiment_id}_summary"
            ] = {
                "path": (
                    path.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
                "sha256": (
                    sha256_file(path)
                ),
            }

        for experiment_id in (
            "E3",
            "E4",
        ):
            path = comparison_paths[
                experiment_id
            ]

            input_artifacts[
                (
                    f"{experiment_id}_"
                    "comparison_table"
                )
            ] = {
                "path": (
                    path.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
                "sha256": (
                    sha256_file(path)
                ),
            }

        output_artifacts = {
            "experiment_catalog": {
                "path": (
                    OUTPUT_DIRECTORY
                    / CATALOG_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 4,
                "sha256": sha256_file(
                    catalog_path
                ),
            },
            "master_metrics": {
                "path": (
                    OUTPUT_DIRECTORY
                    / METRICS_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 62,
                "sha256": sha256_file(
                    metrics_path
                ),
            },
            "master_comparisons": {
                "path": (
                    OUTPUT_DIRECTORY
                    / COMPARISONS_FILENAME
                )
                .relative_to(
                    PROJECT_ROOT
                )
                .as_posix(),
                "rows": 38,
                "sha256": sha256_file(
                    comparisons_path
                ),
            },
        }

        summary_payload = {
            "schema_version": "1.0",
            "contract_id": "TM1",
            "dataset_version": "v2",
            "generation_valid": True,
            "counts": {
                "experiment_catalog_rows": 4,
                "master_metric_rows": 62,
                "master_comparison_rows": 38,
            },
            "metric_rows_by_family": dict(
                sorted(
                    Counter(
                        row[
                            "comparability_family"
                        ]
                        for row in metric_rows
                    ).items()
                )
            ),
            "metric_rows_by_experiment": dict(
                sorted(
                    metric_counts_by_experiment.items()
                )
            ),
            "comparison_rows_by_experiment": dict(
                sorted(
                    comparison_counts_by_experiment.items()
                )
            ),
            "anchor_equivalence": {
                "anchor_id": (
                    "original_openclip_exact"
                ),
                "records": 3,
                "metrics": (
                    anchor_values
                ),
                "valid": True,
            },
            "comparability_families": [
                record["family_id"]
                for record in contract[
                    "comparability_families"
                ]
            ],
            "presentation_rules": (
                contract[
                    "presentation_rules"
                ]
            ),
            "input_artifacts": (
                input_artifacts
            ),
            "output_artifacts": (
                output_artifacts
            ),
        }

        write_json(
            summary_path,
            summary_payload,
        )

        expected_names = {
            CATALOG_FILENAME,
            METRICS_FILENAME,
            COMPARISONS_FILENAME,
            SUMMARY_FILENAME,
        }

        actual_names = {
            path.name
            for path
            in TEMPORARY_DIRECTORY.iterdir()
            if path.is_file()
        }

        assert (
            actual_names
            == expected_names
        )

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

    print("=" * 92)
    print(
        "GENERACIÓN DE TABLAS MAESTRAS "
        "COMPLETADA"
    )
    print("=" * 92)

    print(
        "Catálogo experimental:",
        len(catalog_rows),
    )

    print(
        "Filas de métricas:",
        len(metric_rows),
    )

    print(
        "Filas de comparaciones:",
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
            metric_counts_by_experiment[
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
            comparison_counts_by_experiment[
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
    print("Artefactos generados: 4")

    print(
        "Directorio:",
        OUTPUT_DIRECTORY
        .relative_to(
            PROJECT_ROOT
        )
        .as_posix(),
    )

    print(
        "Generación válida:",
        True,
    )


if __name__ == "__main__":
    main()
