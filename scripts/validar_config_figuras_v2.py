"""Valida el contrato de las figuras principales v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "figuras_v2.json"
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


def metric_index(
    rows: list[dict[str, str]],
) -> dict[
    tuple[str, str, str, str],
    dict[str, str],
]:
    result = {}

    for row in rows:
        key = (
            row["experiment_id"],
            row["source_section"],
            row["condition"],
            row["metric_name"],
        )

        assert key not in result

        result[key] = row

    return result


def get_metric(
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    record: dict[str, str],
    metric_name: str,
) -> float:
    key = (
        record["experiment_id"],
        record["source_section"],
        record["condition"],
        metric_name,
    )

    assert key in index, (
        f"No existe la métrica: {key}."
    )

    return float(
        index[key]["metric_value"]
    )


def get_family(
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    record: dict[str, str],
    metric_name: str,
) -> str:
    key = (
        record["experiment_id"],
        record["source_section"],
        record["condition"],
        metric_name,
    )

    assert key in index

    return index[key][
        "comparability_family"
    ]


def main() -> None:
    contract = load_json(
        CONFIG_PATH
    )

    assert contract[
        "schema_version"
    ] == "1.0"

    assert (
        contract["contract_id"]
        == "FG1"
    )

    assert (
        contract["dataset_version"]
        == "v2"
    )

    input_paths = {
        key: (
            PROJECT_ROOT
            / record["path"]
        )
        for key, record in contract[
            "input_artifacts"
        ].items()
    }

    metric_fields, metric_rows = load_csv(
        input_paths["master_metrics"]
    )

    comparison_fields, comparison_rows = (
        load_csv(
            input_paths[
                "master_comparisons"
            ]
        )
    )

    master_summary = load_json(
        input_paths["master_summary"]
    )

    assert len(metric_rows) == 62
    assert len(comparison_rows) == 38

    assert (
        master_summary[
            "generation_valid"
        ]
        is True
    )

    assert master_summary["counts"] == {
        "experiment_catalog_rows": 4,
        "master_metric_rows": 62,
        "master_comparison_rows": 38,
    }

    assert {
        "experiment_id",
        "source_section",
        "condition",
        "metric_name",
        "metric_value",
        "sample_count",
        "comparability_family",
    }.issubset(metric_fields)

    assert {
        "experiment_id",
        "comparison_id",
        "statistic_name",
        "statistic_value",
    }.issubset(comparison_fields)

    summary_output_record = (
        master_summary[
            "output_artifacts"
        ]["master_metrics"]
    )

    assert (
        summary_output_record["path"]
        == contract[
            "input_artifacts"
        ]["master_metrics"]["path"]
    )

    assert (
        summary_output_record["rows"]
        == 62
    )

    assert (
        summary_output_record["sha256"]
        == sha256_file(
            input_paths[
                "master_metrics"
            ]
        )
    )

    metrics = metric_index(
        metric_rows
    )

    output = contract["output"]

    assert (
        output["figure_count"]
        == 5
    )

    assert (
        output["figure_file_count"]
        == 10
    )

    assert (
        output["manifest"]["rows"]
        == 10
    )

    assert (
        output["total_file_count"]
        == 12
    )

    assert output["formats"] == [
        "png",
        "svg",
    ]

    style = contract[
        "global_style"
    ]

    assert (
        style["backend"]
        == "Agg"
    )

    assert (
        style["font_family"]
        == "DejaVu Sans"
    )

    assert (
        style["dpi_png"]
        == 300
    )

    available_fonts = {
        font.name
        for font
        in font_manager.fontManager.ttflist
    }

    assert (
        style["font_family"]
        in available_fonts
    )

    assert tuple(
        contract["manifest_fields"]
    ) == (
        "manifest_row_index",
        "figure_id",
        "filename_stem",
        "file_format",
        "relative_path",
        "sha256",
        "size_bytes",
    )

    figures = contract["figures"]

    assert len(figures) == 5

    assert [
        figure["figure_id"]
        for figure in figures
    ] == [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    ]

    assert len(
        {
            figure["filename_stem"]
            for figure in figures
        }
    ) == 5

    figure_by_id = {
        figure["figure_id"]: figure
        for figure in figures
    }

    f1 = figure_by_id["F1"]

    assert (
        f1["comparability_family"]
        == "global_exact_retrieval"
    )

    assert (
        f1["chart_type"]
        == "grouped_bar"
    )

    assert (
        f1["expected_data_points"]
        == 16
    )

    assert len(f1["records"]) == 4
    assert len(f1["metrics"]) == 4

    f1_values = {}

    for record in f1["records"]:
        condition = record["condition"]

        f1_values[condition] = {}

        for metric_name in f1["metrics"]:
            assert (
                get_family(
                    metrics,
                    record,
                    metric_name,
                )
                == f1[
                    "comparability_family"
                ]
            )

            f1_values[
                condition
            ][metric_name] = get_metric(
                metrics,
                record,
                metric_name,
            )

    assert_close(
        f1_values["random"][
            "recall_at_1"
        ],
        0.014285714286,
        "F1 random Recall@1",
    )

    assert_close(
        f1_values["color_histogram"][
            "recall_at_5"
        ],
        0.625,
        "F1 HSV Recall@5",
    )

    assert_close(
        f1_values["openclip"]["mrr"],
        0.462076240384,
        "F1 OpenCLIP MRR",
    )

    assert_close(
        f1_values[
            "grayscale_image_full_caption"
        ]["ndcg_at_10"],
        0.251120969549,
        "F1 gris nDCG@10",
    )

    f2 = figure_by_id["F2"]

    assert (
        f2["comparability_family"]
        == "hard_negative_forced_choice"
    )

    assert len(
        f2["bar_metrics"]
    ) == 4

    assert f2[
        "annotation_metrics"
    ] == [
        "mean_paired_difference",
    ]

    f2_values = {}

    for metric_name in (
        f2["bar_metrics"]
        + f2["annotation_metrics"]
    ):
        assert (
            get_family(
                metrics,
                f2["record"],
                metric_name,
            )
            == f2[
                "comparability_family"
            ]
        )

        f2_values[metric_name] = (
            get_metric(
                metrics,
                f2["record"],
                metric_name,
            )
        )

    assert_close(
        f2_values[
            "hard_negative_accuracy"
        ],
        0.517857142857,
        "F2 exactitud",
    )

    assert_close(
        f2_values["pairwise_win_rate"],
        0.834821428571,
        "F2 tasa pareada",
    )

    assert_close(
        f2_values[
            "mean_paired_difference"
        ],
        0.025734035431,
        "F2 diferencia media",
    )

    f3 = figure_by_id["F3"]

    assert (
        f3["comparability_family"]
        == "structural_multi_relevance"
    )

    assert (
        f3["expected_data_points"]
        == 20
    )

    assert len(f3["records"]) == 4
    assert len(f3["metrics"]) == 5

    f3_values = {}

    for record in f3["records"]:
        condition = record["condition"]

        f3_values[condition] = {}

        for metric_name in f3["metrics"]:
            assert (
                get_family(
                    metrics,
                    record,
                    metric_name,
                )
                == f3[
                    "comparability_family"
                ]
            )

            f3_values[
                condition
            ][metric_name] = get_metric(
                metrics,
                record,
                metric_name,
            )

    assert_close(
        f3_values[
            "grayscale_image_caption_without_color"
        ]["structural_hit_at_1"],
        0.425,
        "F3 mejor Hit@1",
    )

    assert_close(
        f3_values[
            "original_image_full_caption"
        ]["structural_hit_at_5"],
        0.832142857143,
        "F3 mejor Hit@5",
    )

    f4 = figure_by_id["F4"]

    assert (
        f4["comparability_family"]
        == "global_exact_retrieval"
    )

    assert len(
        f4["bar_metrics"]
    ) == 4

    assert f4[
        "annotation_metrics"
    ] == [
        "positive_margin",
    ]

    f4_deltas = {}

    for metric_name in (
        f4["bar_metrics"]
        + f4["annotation_metrics"]
    ):
        minuend_family = get_family(
            metrics,
            f4["minuend"],
            metric_name,
        )

        subtrahend_family = get_family(
            metrics,
            f4["subtrahend"],
            metric_name,
        )

        assert (
            minuend_family
            == f4[
                "comparability_family"
            ]
        )

        assert (
            subtrahend_family
            == f4[
                "comparability_family"
            ]
        )

        f4_deltas[metric_name] = (
            get_metric(
                metrics,
                f4["minuend"],
                metric_name,
            )
            - get_metric(
                metrics,
                f4["subtrahend"],
                metric_name,
            )
        )

    expected_deltas = {
        "recall_at_1": -0.15,
        "recall_at_5": (
            -0.510714285714
        ),
        "mrr": -0.254995943338,
        "ndcg_at_10": (
            -0.330187501181
        ),
        "positive_margin": (
            -0.009412784077
        ),
    }

    for metric_name, expected in (
        expected_deltas.items()
    ):
        assert_close(
            f4_deltas[metric_name],
            expected,
            f"F4 {metric_name}",
        )

    assert all(
        value < 0.0
        for value in f4_deltas.values()
    )

    f5 = figure_by_id["F5"]

    assert (
        f5["comparability_family"]
        == "structural_multi_relevance"
    )

    assert (
        f5["expected_point_count"]
        == 4
    )

    assert len(f5["records"]) == 4

    f5_points = []

    for record in f5["records"]:
        x_value = get_metric(
            metrics,
            record,
            f5["x_metric"],
        )

        y_value = get_metric(
            metrics,
            record,
            f5["y_metric"],
        )

        assert (
            get_family(
                metrics,
                record,
                f5["x_metric"],
            )
            == f5[
                "comparability_family"
            ]
        )

        assert (
            get_family(
                metrics,
                record,
                f5["y_metric"],
            )
            == f5[
                "comparability_family"
            ]
        )

        f5_points.append(
            (
                record["condition"],
                x_value,
                y_value,
            )
        )

    assert len(f5_points) == 4

    assert (
        contract["protected_paths"]
        == [
            {
                "path": "figures",
                "rule": "do_not_modify",
            },
        ]
    )

    protected_path = (
        PROJECT_ROOT / "figures"
    )

    output_path = (
        PROJECT_ROOT
        / output["directory"]
    )

    assert (
        protected_path.resolve()
        != output_path.resolve()
    )

    assert len(
        contract["validation_rules"]
    ) == 7

    print("=" * 92)
    print(
        "VALIDACIÓN DEL CONTRATO DE "
        "FIGURAS SUPERADA"
    )
    print("=" * 92)

    print(
        "Matplotlib:",
        matplotlib.__version__,
    )

    print(
        "Fuente:",
        style["font_family"],
    )

    print(
        "Figuras previstas:",
        len(figures),
    )

    print(
        "Archivos gráficos previstos:",
        output[
            "figure_file_count"
        ],
    )

    print(
        "Archivos totales previstos:",
        output[
            "total_file_count"
        ],
    )

    print()
    print("F1 — recuperación exacta:")

    for condition, values in (
        f1_values.items()
    ):
        print(
            "-",
            condition,
            values,
        )

    print()
    print("F2 — negativos difíciles:")

    for metric_name, value in (
        f2_values.items()
    ):
        print(
            f"- {metric_name}:",
            format(value, ".12f"),
        )

    print()
    print(
        "F3 — puntos métricos:",
        20,
    )

    print(
        "F3 mejor Hit@1:",
        (
            "grayscale_image_caption_"
            "without_color"
        ),
        format(
            f3_values[
                "grayscale_image_caption_without_color"
            ]["structural_hit_at_1"],
            ".12f",
        ),
    )

    print(
        "F3 mejor Hit@5:",
        "original_image_full_caption",
        format(
            f3_values[
                "original_image_full_caption"
            ]["structural_hit_at_5"],
            ".12f",
        ),
    )

    print()
    print("F4 — deltas exactos:")

    for metric_name, value in (
        f4_deltas.items()
    ):
        print(
            f"- delta_{metric_name}:",
            format(value, ".12f"),
        )

    print()
    print("F5 — puntos Hit@1–Hit@5:")

    for condition, x_value, y_value in (
        f5_points
    ):
        print(
            "-",
            condition,
            "x=",
            format(x_value, ".12f"),
            "y=",
            format(y_value, ".12f"),
        )

    print()
    print(
        "Directorio protegido 'figures' "
        "sin modificar:",
        True,
    )

    print(
        "Directorio de salida:",
        output["directory"],
    )

    print(
        "No se generaron figuras."
    )

    print(
        "Contrato de figuras válido:",
        True,
    )


if __name__ == "__main__":
    main()
