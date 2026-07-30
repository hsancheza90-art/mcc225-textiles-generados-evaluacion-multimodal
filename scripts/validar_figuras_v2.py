"""Valida independientemente las figuras principales del protocolo v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "figuras_v2.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "figuras"
)

MANIFEST_PATH = (
    OUTPUT_DIRECTORY
    / "manifiesto_figuras_v2.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "resumen_figuras_v2.json"
)

MANIFEST_FIELDS = (
    "manifest_row_index",
    "figure_id",
    "filename_stem",
    "file_format",
    "relative_path",
    "sha256",
    "size_bytes",
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


def inventory_directory(
    directory: Path,
) -> dict[str, str]:
    if not directory.exists():
        return {}

    return {
        path.relative_to(
            directory
        ).as_posix(): sha256_file(path)
        for path in sorted(
            directory.rglob("*")
        )
        if path.is_file()
    }


def inventory_digest(
    inventory: dict[str, str],
) -> str:
    encoded = json.dumps(
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


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


def get_metric_row(
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    record: dict[str, str],
    metric_name: str,
) -> dict[str, str]:
    key = (
        record["experiment_id"],
        record["source_section"],
        record["condition"],
        metric_name,
    )

    assert key in index, (
        f"No existe la métrica {key}."
    )

    return index[key]


def get_metric_value(
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    record: dict[str, str],
    metric_name: str,
) -> float:
    return float(
        get_metric_row(
            index,
            record,
            metric_name,
        )["metric_value"]
    )


def assert_nested_close(
    actual: Any,
    expected: Any,
    context: str,
) -> None:
    if isinstance(expected, dict):
        assert isinstance(
            actual,
            dict,
        ), f"{context}: se esperaba un diccionario."

        assert set(actual) == set(expected), (
            f"{context}: claves distintas. "
            f"Actual={sorted(actual)}, "
            f"esperado={sorted(expected)}."
        )

        for key in expected:
            assert_nested_close(
                actual[key],
                expected[key],
                f"{context}.{key}",
            )

        return

    if isinstance(expected, list):
        assert isinstance(
            actual,
            list,
        ), f"{context}: se esperaba una lista."

        assert len(actual) == len(expected), (
            f"{context}: longitudes distintas."
        )

        for index, (
            actual_item,
            expected_item,
        ) in enumerate(
            zip(
                actual,
                expected,
                strict=True,
            )
        ):
            assert_nested_close(
                actual_item,
                expected_item,
                f"{context}[{index}]",
            )

        return

    numeric_expected = (
        isinstance(
            expected,
            (int, float),
        )
        and not isinstance(
            expected,
            bool,
        )
    )

    numeric_actual = (
        isinstance(
            actual,
            (int, float),
        )
        and not isinstance(
            actual,
            bool,
        )
    )

    if (
        numeric_expected
        and numeric_actual
    ):
        assert math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ), (
            f"{context}: actual={actual}, "
            f"esperado={expected}."
        )

        return

    assert actual == expected, (
        f"{context}: actual={actual!r}, "
        f"esperado={expected!r}."
    )


def expected_figure_data(
    specification: dict[str, Any],
    metrics: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
) -> dict[str, Any]:
    chart_type = specification[
        "chart_type"
    ]

    if chart_type == "grouped_bar":
        records = []

        for record in specification[
            "records"
        ]:
            values = {}

            for metric_name in specification[
                "metrics"
            ]:
                source_row = get_metric_row(
                    metrics,
                    record,
                    metric_name,
                )

                assert (
                    source_row[
                        "comparability_family"
                    ]
                    == specification[
                        "comparability_family"
                    ]
                )

                values[metric_name] = float(
                    source_row[
                        "metric_value"
                    ]
                )

            records.append(
                {
                    "experiment_id": (
                        record[
                            "experiment_id"
                        ]
                    ),
                    "source_section": (
                        record[
                            "source_section"
                        ]
                    ),
                    "condition": (
                        record[
                            "condition"
                        ]
                    ),
                    "values": values,
                }
            )

        return {
            "records": records,
            "metrics": list(
                specification["metrics"]
            ),
            "data_point_count": (
                len(
                    specification[
                        "records"
                    ]
                )
                * len(
                    specification[
                        "metrics"
                    ]
                )
            ),
        }

    if chart_type == "horizontal_bar":
        record = specification[
            "record"
        ]

        bar_values = {}
        annotation_values = {}

        for metric_name in specification[
            "bar_metrics"
        ]:
            source_row = get_metric_row(
                metrics,
                record,
                metric_name,
            )

            assert (
                source_row[
                    "comparability_family"
                ]
                == specification[
                    "comparability_family"
                ]
            )

            bar_values[metric_name] = float(
                source_row["metric_value"]
            )

        for metric_name in specification[
            "annotation_metrics"
        ]:
            source_row = get_metric_row(
                metrics,
                record,
                metric_name,
            )

            assert (
                source_row[
                    "comparability_family"
                ]
                == specification[
                    "comparability_family"
                ]
            )

            annotation_values[
                metric_name
            ] = float(
                source_row["metric_value"]
            )

        return {
            "record": record,
            "bar_values": bar_values,
            "annotation_values": (
                annotation_values
            ),
            "bar_count": len(
                specification[
                    "bar_metrics"
                ]
            ),
        }

    if chart_type == "delta_bar":
        all_metrics = (
            specification["bar_metrics"]
            + specification[
                "annotation_metrics"
            ]
        )

        deltas = {}

        for metric_name in all_metrics:
            minuend_row = get_metric_row(
                metrics,
                specification[
                    "minuend"
                ],
                metric_name,
            )

            subtrahend_row = get_metric_row(
                metrics,
                specification[
                    "subtrahend"
                ],
                metric_name,
            )

            assert (
                minuend_row[
                    "comparability_family"
                ]
                == specification[
                    "comparability_family"
                ]
            )

            assert (
                subtrahend_row[
                    "comparability_family"
                ]
                == specification[
                    "comparability_family"
                ]
            )

            deltas[metric_name] = (
                float(
                    minuend_row[
                        "metric_value"
                    ]
                )
                - float(
                    subtrahend_row[
                        "metric_value"
                    ]
                )
            )

        return {
            "minuend": (
                specification["minuend"]
            ),
            "subtrahend": (
                specification[
                    "subtrahend"
                ]
            ),
            "bar_deltas": {
                metric_name: deltas[
                    metric_name
                ]
                for metric_name
                in specification[
                    "bar_metrics"
                ]
            },
            "annotation_deltas": {
                metric_name: deltas[
                    metric_name
                ]
                for metric_name
                in specification[
                    "annotation_metrics"
                ]
            },
            "bar_count": len(
                specification[
                    "bar_metrics"
                ]
            ),
        }

    if chart_type == "scatter":
        points = []

        for record in specification[
            "records"
        ]:
            x_row = get_metric_row(
                metrics,
                record,
                specification[
                    "x_metric"
                ],
            )

            y_row = get_metric_row(
                metrics,
                record,
                specification[
                    "y_metric"
                ],
            )

            assert (
                x_row[
                    "comparability_family"
                ]
                == specification[
                    "comparability_family"
                ]
            )

            assert (
                y_row[
                    "comparability_family"
                ]
                == specification[
                    "comparability_family"
                ]
            )

            points.append(
                {
                    "condition": (
                        record[
                            "condition"
                        ]
                    ),
                    "x": float(
                        x_row[
                            "metric_value"
                        ]
                    ),
                    "y": float(
                        y_row[
                            "metric_value"
                        ]
                    ),
                }
            )

        return {
            "x_metric": (
                specification[
                    "x_metric"
                ]
            ),
            "y_metric": (
                specification[
                    "y_metric"
                ]
            ),
            "points": points,
            "point_count": len(points),
        }

    raise AssertionError(
        "Tipo de figura desconocido: "
        f"{chart_type}"
    )


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
    )

    assert (
        contract["schema_version"]
        == "1.0"
    )

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

    _, comparison_rows = load_csv(
        input_paths[
            "master_comparisons"
        ]
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

    assert {
        "experiment_id",
        "source_section",
        "condition",
        "metric_name",
        "metric_value",
        "comparability_family",
    }.issubset(metric_fields)

    metrics = metric_index(
        metric_rows
    )

    manifest_fields, manifest_rows = (
        load_csv(
            MANIFEST_PATH
        )
    )

    summary = load_json(
        SUMMARY_PATH
    )

    assert tuple(
        manifest_fields
    ) == MANIFEST_FIELDS

    assert len(manifest_rows) == 10

    assert [
        int(
            row[
                "manifest_row_index"
            ]
        )
        for row in manifest_rows
    ] == list(range(10))

    assert (
        summary["schema_version"]
        == "1.0"
    )

    assert (
        summary["contract_id"]
        == "FG1"
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
        "figures": 5,
        "graphic_files": 10,
        "manifest_rows": 10,
        "total_files": 12,
    }

    assert (
        summary["formats"]
        == contract["output"][
            "formats"
        ]
    )

    assert (
        summary["style"]
        == contract[
            "global_style"
        ]
    )

    assert (
        summary["matplotlib"]
        == matplotlib.__version__
    )

    expected_names = {
        MANIFEST_PATH.name,
        SUMMARY_PATH.name,
    }

    expected_manifest_order = []

    for specification in contract[
        "figures"
    ]:
        for file_format in contract[
            "output"
        ]["formats"]:
            filename = (
                f"{specification['filename_stem']}"
                f".{file_format}"
            )

            expected_names.add(
                filename
            )

            expected_manifest_order.append(
                (
                    specification[
                        "figure_id"
                    ],
                    specification[
                        "filename_stem"
                    ],
                    file_format,
                    (
                        OUTPUT_DIRECTORY
                        / filename
                    )
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix(),
                )
            )

    actual_names = {
        path.name
        for path
        in OUTPUT_DIRECTORY.iterdir()
        if path.is_file()
    }

    assert actual_names == expected_names
    assert len(actual_names) == 12

    for row_index, (
        manifest_row,
        expected_record,
    ) in enumerate(
        zip(
            manifest_rows,
            expected_manifest_order,
            strict=True,
        )
    ):
        (
            expected_figure_id,
            expected_stem,
            expected_format,
            expected_relative_path,
        ) = expected_record

        assert (
            manifest_row[
                "manifest_row_index"
            ]
            == str(row_index)
        )

        assert (
            manifest_row["figure_id"]
            == expected_figure_id
        )

        assert (
            manifest_row[
                "filename_stem"
            ]
            == expected_stem
        )

        assert (
            manifest_row[
                "file_format"
            ]
            == expected_format
        )

        assert (
            manifest_row[
                "relative_path"
            ]
            == expected_relative_path
        )

        artifact_path = (
            PROJECT_ROOT
            / manifest_row[
                "relative_path"
            ]
        )

        assert artifact_path.exists()
        assert artifact_path.is_file()

        assert (
            manifest_row["sha256"]
            == sha256_file(
                artifact_path
            )
        )

        assert (
            int(
                manifest_row[
                    "size_bytes"
                ]
            )
            == artifact_path.stat().st_size
        )

        assert artifact_path.stat().st_size > 0

        if expected_format == "png":
            with Image.open(
                artifact_path
            ) as image:
                image.verify()

            with Image.open(
                artifact_path
            ) as image:
                assert (
                    image.format
                    == "PNG"
                )

                assert image.width >= 1800
                assert image.height >= 1200

                assert image.mode in {
                    "RGB",
                    "RGBA",
                }

        if expected_format == "svg":
            svg_raw = (
                artifact_path.read_bytes()
            )

            assert not svg_raw.startswith(
                b"\xef\xbb\xbf"
            )

            assert b"\r\n" not in svg_raw

            svg_text = svg_raw.decode(
                "utf-8"
            )

            trailing_whitespace_lines = [
                line_number
                for line_number, line
                in enumerate(
                    svg_text.splitlines(),
                    start=1,
                )
                if line != line.rstrip()
            ]

            assert not (
                trailing_whitespace_lines
            ), (
                f"{artifact_path}: contiene "
                "espacios finales en las líneas "
                f"{trailing_whitespace_lines[:10]}."
            )

            root = ET.parse(
                artifact_path
            ).getroot()

            assert root.tag.endswith(
                "svg"
            )

    summary_figures = summary[
        "figures"
    ]

    assert len(summary_figures) == 5

    assert [
        record["figure_id"]
        for record in summary_figures
    ] == [
        record["figure_id"]
        for record in contract[
            "figures"
        ]
    ]

    manifest_by_key = {
        (
            row["figure_id"],
            row["file_format"],
        ): row
        for row in manifest_rows
    }

    assert len(manifest_by_key) == 10

    for specification, figure_summary in zip(
        contract["figures"],
        summary_figures,
        strict=True,
    ):
        assert (
            figure_summary[
                "figure_id"
            ]
            == specification[
                "figure_id"
            ]
        )

        assert (
            figure_summary[
                "filename_stem"
            ]
            == specification[
                "filename_stem"
            ]
        )

        assert (
            figure_summary[
                "comparability_family"
            ]
            == specification[
                "comparability_family"
            ]
        )

        assert (
            figure_summary[
                "chart_type"
            ]
            == specification[
                "chart_type"
            ]
        )

        assert (
            figure_summary["title"]
            == specification["title"]
        )

        expected_data = (
            expected_figure_data(
                specification,
                metrics,
            )
        )

        assert_nested_close(
            figure_summary["data"],
            expected_data,
            (
                "summary.figures."
                f"{specification['figure_id']}.data"
            ),
        )

        assert len(
            figure_summary["files"]
        ) == 2

        for file_record in (
            figure_summary["files"]
        ):
            key = (
                file_record[
                    "figure_id"
                ],
                file_record[
                    "file_format"
                ],
            )

            assert key in manifest_by_key

            manifest_record = (
                manifest_by_key[key]
            )

            assert (
                file_record[
                    "filename_stem"
                ]
                == manifest_record[
                    "filename_stem"
                ]
            )

            assert (
                file_record[
                    "relative_path"
                ]
                == manifest_record[
                    "relative_path"
                ]
            )

            assert (
                file_record["sha256"]
                == manifest_record[
                    "sha256"
                ]
            )

            assert (
                int(
                    file_record[
                        "size_bytes"
                    ]
                )
                == int(
                    manifest_record[
                        "size_bytes"
                    ]
                )
            )

    assert set(
        summary["input_artifacts"]
    ) == set(input_paths)

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

    output_artifacts = summary[
        "output_artifacts"
    ]

    assert (
        output_artifacts[
            "manifest"
        ]["path"]
        == MANIFEST_PATH
        .relative_to(
            PROJECT_ROOT
        )
        .as_posix()
    )

    assert (
        output_artifacts[
            "manifest"
        ]["rows"]
        == 10
    )

    assert (
        output_artifacts[
            "manifest"
        ]["sha256"]
        == sha256_file(
            MANIFEST_PATH
        )
    )

    graphic_outputs = (
        output_artifacts[
            "graphic_files"
        ]
    )

    assert len(
        graphic_outputs
    ) == 10

    for graphic_record, manifest_record in zip(
        graphic_outputs,
        manifest_rows,
        strict=True,
    ):
        assert (
            graphic_record[
                "figure_id"
            ]
            == manifest_record[
                "figure_id"
            ]
        )

        assert (
            graphic_record[
                "file_format"
            ]
            == manifest_record[
                "file_format"
            ]
        )

        assert (
            graphic_record["path"]
            == manifest_record[
                "relative_path"
            ]
        )

        assert (
            graphic_record[
                "sha256"
            ]
            == manifest_record[
                "sha256"
            ]
        )

        assert (
            int(
                graphic_record[
                    "size_bytes"
                ]
            )
            == int(
                manifest_record[
                    "size_bytes"
                ]
            )
        )

    protected_record = summary[
        "protected_path"
    ]

    protected_path = (
        PROJECT_ROOT
        / protected_record["path"]
    )

    protected_inventory = (
        inventory_directory(
            protected_path
        )
    )

    protected_digest = (
        inventory_digest(
            protected_inventory
        )
    )

    assert (
        protected_record["path"]
        == "figures"
    )

    assert (
        protected_record[
            "file_count"
        ]
        == len(
            protected_inventory
        )
    )

    assert (
        protected_record[
            "inventory_sha256_before"
        ]
        == protected_digest
    )

    assert (
        protected_record[
            "inventory_sha256_after"
        ]
        == protected_digest
    )

    assert (
        protected_record[
            "unchanged"
        ]
        is True
    )

    temporary_paths = (
        OUTPUT_DIRECTORY.with_name(
            OUTPUT_DIRECTORY.name
            + ".tmp"
        ),
        OUTPUT_DIRECTORY.with_name(
            OUTPUT_DIRECTORY.name
            + ".previous"
        ),
    )

    for path in temporary_paths:
        assert not path.exists()

    figure_counts = Counter(
        row["figure_id"]
        for row in manifest_rows
    )

    format_counts = Counter(
        row["file_format"]
        for row in manifest_rows
    )

    assert figure_counts == {
        "F1": 2,
        "F2": 2,
        "F3": 2,
        "F4": 2,
        "F5": 2,
    }

    assert format_counts == {
        "png": 5,
        "svg": 5,
    }

    print("=" * 92)
    print(
        "VALIDACIÓN INDEPENDIENTE DE "
        "FIGURAS SUPERADA"
    )
    print("=" * 92)

    print(
        "Figuras verificadas:",
        len(summary_figures),
    )

    print(
        "Archivos gráficos verificados:",
        len(manifest_rows),
    )

    print(
        "PNG válidos:",
        format_counts["png"],
    )

    print(
        "SVG válidos:",
        format_counts["svg"],
    )

    print()
    print("Archivos por figura:")

    for figure_id in (
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    ):
        print(
            f"- {figure_id}:",
            figure_counts[
                figure_id
            ],
        )

    print()
    print(
        "Datos gráficos reconstruidos "
        "desde tablas maestras:",
        True,
    )

    print(
        "Hashes del manifiesto:",
        "válidos",
    )

    print(
        "Hashes del resumen:",
        "válidos",
    )

    print(
        "Directorio protegido 'figures':",
        "sin cambios",
    )

    print(
        "Directorios temporales residuales:",
        False,
    )

    print(
        "Figuras válidas:",
        True,
    )


if __name__ == "__main__":
    main()
