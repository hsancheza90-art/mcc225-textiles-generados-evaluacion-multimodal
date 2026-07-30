"""Valida el contrato de consolidación de E1 a E4."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "tablas_maestras_v2.json"
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


def assert_close(
    actual: float,
    expected: float,
    context: str,
    tolerance: float,
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


def main() -> None:
    contract = load_json(
        CONFIG_PATH
    )

    assert (
        contract["schema_version"]
        == "1.0"
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
            PROJECT_ROOT / relative_path
        )
        for experiment_id, relative_path
        in contract[
            "input_summaries"
        ].items()
    }

    assert set(summary_paths) == {
        "E1",
        "E2",
        "E3",
        "E4",
    }

    summaries = {
        experiment_id: load_json(path)
        for experiment_id, path
        in summary_paths.items()
    }

    for experiment_id, summary in (
        summaries.items()
    ):
        assert (
            summary["experiment_id"]
            == experiment_id
        )

        assert (
            summary["dataset_version"]
            == "v2"
        )

        assert (
            summary["evaluation_valid"]
            is True
        )

    assert summaries["E1"]["counts"][
        "queries"
    ] == 280

    assert summaries["E1"]["counts"][
        "gallery_images"
    ] == 56

    assert summaries["E2"]["counts"][
        "queries"
    ] == 56

    assert summaries["E2"]["counts"][
        "candidate_rows"
    ] == 280

    assert summaries["E2"]["counts"][
        "pairwise_rows"
    ] == 224

    assert summaries["E3"]["counts"][
        "query_rows"
    ] == 840

    assert summaries["E3"]["counts"][
        "comparison_rows"
    ] == 3

    assert summaries["E4"]["counts"] == {
        "raw_structural_results": 640,
        "structural_ranking_rows": 35840,
        "structural_group_results": 160,
        "structural_aggregate_rows": 40,
        "paired_group_rows": 160,
        "paired_aggregate_rows": 12,
        "exact_results": 560,
        "exact_aggregate_rows": 14,
    }

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

    global_family = families[
        "global_exact_retrieval"
    ]

    assert global_family[
        "gallery_size"
    ] == 56

    assert global_family[
        "relevance_type"
    ] == "single_exact_image"

    assert global_family["metrics"] == [
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
    ]

    hard_family = families[
        "hard_negative_forced_choice"
    ]

    assert hard_family[
        "candidate_count"
    ] == 5

    assert len(
        hard_family["metrics"]
    ) == 8

    structural_family = families[
        "structural_multi_relevance"
    ]

    assert structural_family[
        "gallery_size"
    ] == 56

    assert structural_family[
        "relevant_count"
    ] == 7

    assert len(
        structural_family["metrics"]
    ) == 6

    assert set(
        summaries["E3"][
            "overall_metrics"
        ]
    ) == {
        "openclip",
        "random",
        "color_histogram",
    }

    assert set(
        summaries["E4"][
            "structural_overall_metrics"
        ]
    ) == {
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
    }

    assert set(
        summaries["E4"][
            "exact_overall_metrics"
        ]
    ) == {
        "original_image_full_caption",
        "grayscale_image_full_caption",
    }

    e2_required_metrics = {
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
        "positive_margin",
        "hard_negative_accuracy",
        "mean_paired_difference",
        "pairwise_win_rate",
    }

    assert e2_required_metrics.issubset(
        summaries["E2"][
            "overall_metrics"
        ]
    )

    anchor = contract[
        "anchor_equivalences"
    ][0]

    assert (
        anchor["anchor_id"]
        == "original_openclip_exact"
    )

    tolerance = float(
        anchor["absolute_tolerance"]
    )

    anchor_records = [
        summaries["E1"][
            "overall_metrics"
        ],
        summaries["E3"][
            "overall_metrics"
        ]["openclip"],
        summaries["E4"][
            "exact_overall_metrics"
        ][
            "original_image_full_caption"
        ],
    ]

    for metric in anchor["metrics"]:
        reference = float(
            anchor_records[0][metric]
        )

        for record_index, record in enumerate(
            anchor_records[1:],
            start=1,
        ):
            assert_close(
                float(record[metric]),
                reference,
                (
                    f"anchor.{metric}."
                    f"record_{record_index}"
                ),
                tolerance,
            )

    comparison_paths = {
        experiment_id: (
            PROJECT_ROOT / relative_path
        )
        for experiment_id, relative_path
        in contract[
            "input_comparison_tables"
        ].items()
    }

    e3_fields, e3_rows = load_csv(
        comparison_paths["E3"]
    )

    e4_fields, e4_rows = load_csv(
        comparison_paths["E4"]
    )

    assert len(e3_rows) == 3
    assert len(e4_rows) == 12

    assert {
        "condition",
        "delta_recall_at_1_vs_openclip",
        "delta_mrr_vs_openclip",
        "delta_ndcg_at_10_vs_openclip",
        "delta_positive_margin_vs_openclip",
        "delta_recall_at_1_vs_random",
        "delta_mrr_vs_random",
        "delta_ndcg_at_10_vs_random",
        "delta_positive_margin_vs_random",
    }.issubset(e3_fields)

    assert {
        "comparison_id",
        "group_dimension",
        "group_value",
        "group_count",
        "delta_structural_hit_at_1",
        "delta_structural_hit_at_5",
        (
            "delta_structural_fractional_"
            "recall_at_5"
        ),
        "delta_structural_mrr",
        "delta_structural_ndcg_at_10",
        "delta_best_relevant_margin",
    }.issubset(e4_fields)

    metric_counts = contract[
        "master_metric_rows"
    ]

    assert (
        metric_counts[
            "global_exact_retrieval"
        ]
        == 6 * 5
    )

    assert (
        metric_counts[
            "hard_negative_forced_choice"
        ]
        == 1 * 8
    )

    assert (
        metric_counts[
            "structural_multi_relevance"
        ]
        == 4 * 6
    )

    assert metric_counts["total"] == (
        metric_counts[
            "global_exact_retrieval"
        ]
        + metric_counts[
            "hard_negative_forced_choice"
        ]
        + metric_counts[
            "structural_multi_relevance"
        ]
    )

    comparison_counts = contract[
        "master_comparison_rows"
    ]

    assert comparison_counts["E3"][
        "rows"
    ] == 12

    assert comparison_counts["E2"][
        "rows"
    ] == 2

    assert comparison_counts["E4"][
        "rows"
    ] == 24

    assert comparison_counts[
        "total"
    ] == 38

    output = contract[
        "output_artifacts"
    ]

    assert output[
        "experiment_catalog"
    ]["rows"] == 4

    assert output[
        "master_metrics"
    ]["rows"] == 62

    assert output[
        "master_comparisons"
    ]["rows"] == 38

    assert len(
        contract["presentation_rules"]
    ) == 5

    print("=" * 92)
    print(
        "VALIDACIÓN DEL CONTRATO DE "
        "TABLAS MAESTRAS SUPERADA"
    )
    print("=" * 92)

    print(
        "Experimentos válidos:",
        sorted(summaries),
    )

    print(
        "Familias de comparabilidad:",
        sorted(families),
    )

    print(
        "Filas previstas del catálogo:",
        4,
    )

    print(
        "Filas previstas de métricas:",
        metric_counts["total"],
    )

    print(
        "Filas previstas de comparaciones:",
        comparison_counts["total"],
    )

    print()
    print(
        "Ancla exacta E1 = E3 OpenCLIP "
        "= E4 original:",
        True,
    )

    for metric in anchor["metrics"]:
        print(
            f"- {metric}:",
            format(
                float(
                    anchor_records[0][metric]
                ),
                ".12f",
            ),
        )

    print()
    print(
        "E2 permanece separado del "
        "retrieval global:",
        True,
    )

    print(
        "E4 estructural permanece separado "
        "de E4 exacto:",
        True,
    )

    print(
        "No se generaron tablas maestras."
    )

    print(
        "Contrato de tablas maestras válido:",
        True,
    )


if __name__ == "__main__":
    main()
