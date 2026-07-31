"""Genera las cinco figuras principales del protocolo v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import textwrap
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLBACKEND",
    "Agg",
)

import matplotlib

matplotlib.use(
    "Agg",
    force=True,
)

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "figuras_v2.json"
)

PROTECTED_DIRECTORY = (
    PROJECT_ROOT
    / "figures"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "figuras"
)

TEMPORARY_DIRECTORY = (
    OUTPUT_DIRECTORY.with_name(
        OUTPUT_DIRECTORY.name + ".tmp"
    )
)

MANIFEST_FILENAME = (
    "manifiesto_figuras_v2.csv"
)

SUMMARY_FILENAME = (
    "resumen_figuras_v2.json"
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

SHORT_CONDITION_LABELS = {
    "random": "Aleatorio",
    "color_histogram": "Histograma HSV",
    "openclip": "OpenCLIP",
    "grayscale_image_full_caption": (
        "Imagen gris\n+ caption completo"
    ),
    "original_image_full_caption": (
        "Imagen original\n+ caption completo"
    ),
    "original_image_caption_without_color": (
        "Imagen original\n+ caption sin color"
    ),
    "grayscale_image_caption_without_color": (
        "Imagen gris\n+ caption sin color"
    ),
}

SCATTER_LABELS = {
    "original_image_full_caption": (
        "Original +\ncompleto"
    ),
    "grayscale_image_full_caption": (
        "Gris +\ncompleto"
    ),
    "original_image_caption_without_color": (
        "Original +\nsin color"
    ),
    "grayscale_image_caption_without_color": (
        "Gris +\nsin color"
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


def normalize_svg_file(
    path: Path,
) -> None:
    """Normaliza un SVG a UTF-8, LF y sin espacios finales."""

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    text = raw.decode("utf-8")

    normalized = (
        "\n".join(
            line.rstrip()
            for line in text.splitlines()
        )
        + "\n"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(normalized)

    normalized_raw = path.read_bytes()

    assert not normalized_raw.startswith(
        b"\xef\xbb\xbf"
    )

    assert b"\r\n" not in normalized_raw

    assert all(
        line == line.rstrip()
        for line
        in normalized.splitlines()
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


def format_float(value: float) -> str:
    numeric = float(value)

    assert math.isfinite(numeric)

    return format(
        numeric,
        ".12f",
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


def metric_row(
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


def metric_value(
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    record: dict[str, str],
    metric_name: str,
) -> float:
    return float(
        metric_row(
            index,
            record,
            metric_name,
        )["metric_value"]
    )


def configure_matplotlib(
    style: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": (
                style["font_family"]
            ),
            "font.size": (
                style["tick_fontsize"]
            ),
            "axes.titlesize": (
                style["title_fontsize"]
            ),
            "axes.labelsize": (
                style["label_fontsize"]
            ),
            "xtick.labelsize": (
                style["tick_fontsize"]
            ),
            "ytick.labelsize": (
                style["tick_fontsize"]
            ),
            "legend.fontsize": (
                style["legend_fontsize"]
            ),
            "figure.facecolor": (
                style["figure_facecolor"]
            ),
            "axes.facecolor": (
                style["axes_facecolor"]
            ),
            "svg.fonttype": (
                style["svg_fonttype"]
            ),
            "svg.hashsalt": (
                "mcc225-figuras-v2"
            ),
        }
    )


def add_titles(
    figure: plt.Figure,
    axes: plt.Axes,
    specification: dict[str, Any],
    style: dict[str, Any],
) -> None:
    figure.suptitle(
        specification["title"],
        fontsize=style[
            "title_fontsize"
        ],
        fontweight="semibold",
        y=0.985,
    )

    axes.set_title(
        specification["subtitle"],
        fontsize=style[
            "label_fontsize"
        ],
        pad=10,
    )


def add_source_note(
    figure: plt.Figure,
) -> None:
    figure.text(
        0.99,
        0.012,
        (
            "Fuente: tablas maestras "
            "auditables E1–E4, versión 2."
        ),
        ha="right",
        va="bottom",
        fontsize=7,
    )


def condition_label(
    condition: str,
    fallback: str,
) -> str:
    return SHORT_CONDITION_LABELS.get(
        condition,
        textwrap.fill(
            fallback,
            width=18,
        ),
    )


def annotate_bar_container(
    axes: plt.Axes,
    container: Any,
    values: list[float],
    decimals: int,
) -> None:
    labels = [
        format(
            value,
            f".{decimals}f",
        )
        for value in values
    ]

    axes.bar_label(
        container,
        labels=labels,
        padding=2,
        fontsize=7,
        rotation=90,
    )


def save_figure(
    figure: plt.Figure,
    specification: dict[str, Any],
    style: dict[str, Any],
    formats: list[str],
    temporary_directory: Path,
) -> list[dict[str, Any]]:
    records = []

    for file_format in formats:
        filename = (
            f"{specification['filename_stem']}"
            f".{file_format}"
        )

        output_path = (
            temporary_directory
            / filename
        )

        if file_format == "png":
            metadata = {
                "Software": (
                    "matplotlib "
                    f"{matplotlib.__version__}"
                ),
                "Title": (
                    specification["title"]
                ),
                "Description": (
                    specification[
                        "subtitle"
                    ]
                ),
            }

            figure.savefig(
                output_path,
                format="png",
                dpi=style["dpi_png"],
                bbox_inches=style[
                    "bbox_inches"
                ],
                facecolor=style[
                    "figure_facecolor"
                ],
                metadata=metadata,
            )

        elif file_format == "svg":
            metadata = {
                "Creator": (
                    "mcc225-textiles-generados-"
                    "evaluacion-multimodal"
                ),
                "Date": None,
                "Title": (
                    specification["title"]
                ),
                "Description": (
                    specification[
                        "subtitle"
                    ]
                ),
            }

            figure.savefig(
                output_path,
                format="svg",
                bbox_inches=style[
                    "bbox_inches"
                ],
                facecolor=style[
                    "figure_facecolor"
                ],
                metadata=metadata,
            )

            normalize_svg_file(
                output_path
            )

        else:
            raise AssertionError(
                "Formato gráfico desconocido: "
                f"{file_format}"
            )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        records.append(
            {
                "figure_id": (
                    specification[
                        "figure_id"
                    ]
                ),
                "filename_stem": (
                    specification[
                        "filename_stem"
                    ]
                ),
                "file_format": (
                    file_format
                ),
                "relative_path": (
                    (
                        OUTPUT_DIRECTORY
                        / filename
                    )
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "sha256": (
                    sha256_file(
                        output_path
                    )
                ),
                "size_bytes": (
                    output_path.stat().st_size
                ),
            }
        )

    if style["close_after_save"]:
        plt.close(figure)

    return records


def render_grouped_bar(
    specification: dict[str, Any],
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    metric_labels: dict[str, str],
    style: dict[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    records = specification["records"]
    metrics = specification["metrics"]

    values = np.asarray(
        [
            [
                metric_value(
                    index,
                    record,
                    metric_name,
                )
                for metric_name in metrics
            ]
            for record in records
        ],
        dtype=np.float64,
    )

    assert values.shape == (
        len(records),
        len(metrics),
    )

    figure, axes = plt.subplots(
        figsize=tuple(
            specification[
                "figure_size_inches"
            ]
        )
    )

    add_titles(
        figure,
        axes,
        specification,
        style,
    )

    x_positions = np.arange(
        len(records),
        dtype=np.float64,
    )

    width = min(
        0.8 / len(metrics),
        0.18,
    )

    for metric_index, metric_name in enumerate(
        metrics
    ):
        offset = (
            metric_index
            - (len(metrics) - 1) / 2.0
        ) * width

        metric_values = values[
            :,
            metric_index,
        ].tolist()

        container = axes.bar(
            x_positions + offset,
            metric_values,
            width=width,
            label=metric_labels[
                metric_name
            ],
        )

        if specification[
            "show_value_labels"
        ]:
            annotate_bar_container(
                axes,
                container,
                metric_values,
                style[
                    "annotation_decimals"
                ],
            )

    labels = []

    for record in records:
        reference = metric_row(
            index,
            record,
            metrics[0],
        )

        labels.append(
            condition_label(
                record["condition"],
                reference[
                    "condition_label"
                ],
            )
        )

    axes.set_xticks(
        x_positions,
        labels,
    )

    axes.set_xlabel(
        specification["x_label"]
    )

    axes.set_ylabel(
        specification["y_label"]
    )

    axes.set_ylim(
        *specification["y_limits"]
    )

    axes.grid(
        axis="y",
        alpha=style["grid_alpha"],
    )

    axes.set_axisbelow(True)

    legend_location = (
        "upper right"
        if specification[
            "figure_id"
        ] == "F3"
        else "upper left"
    )

    axes.legend(
        loc=legend_location,
        frameon=False,
        ncol=2,
    )

    add_source_note(
        figure
    )

    figure.tight_layout(
        rect=(
            0.02,
            0.05,
            0.98,
            0.92,
        )
    )

    payload = {
        "records": [
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
                    record["condition"]
                ),
                "values": {
                    metric_name: float(
                        values[
                            record_index,
                            metric_index,
                        ]
                    )
                    for metric_index, metric_name
                    in enumerate(metrics)
                },
            }
            for record_index, record
            in enumerate(records)
        ],
        "metrics": list(metrics),
        "data_point_count": int(
            values.size
        ),
    }

    return figure, payload


def render_horizontal_bar(
    specification: dict[str, Any],
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    metric_labels: dict[str, str],
    style: dict[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    record = specification["record"]

    bar_metrics = specification[
        "bar_metrics"
    ]

    values = [
        metric_value(
            index,
            record,
            metric_name,
        )
        for metric_name in bar_metrics
    ]

    annotation_values = {
        metric_name: metric_value(
            index,
            record,
            metric_name,
        )
        for metric_name
        in specification[
            "annotation_metrics"
        ]
    }

    figure, axes = plt.subplots(
        figsize=tuple(
            specification[
                "figure_size_inches"
            ]
        )
    )

    add_titles(
        figure,
        axes,
        specification,
        style,
    )

    y_positions = np.arange(
        len(values),
        dtype=np.float64,
    )

    container = axes.barh(
        y_positions,
        values,
    )

    axes.set_yticks(
        y_positions,
        [
            metric_labels[
                metric_name
            ]
            for metric_name
            in bar_metrics
        ],
    )

    axes.invert_yaxis()

    axes.set_xlabel(
        specification["x_label"]
    )

    axes.set_ylabel(
        specification["y_label"]
    )

    axes.set_xlim(
        *specification["x_limits"]
    )

    axes.grid(
        axis="x",
        alpha=style["grid_alpha"],
    )

    axes.set_axisbelow(True)

    for rectangle, value in zip(
        container,
        values,
        strict=True,
    ):
        axes.text(
            value + 0.015,
            (
                rectangle.get_y()
                + rectangle.get_height()
                / 2.0
            ),
            format(
                value,
                (
                    f".{style['annotation_decimals']}"
                    "f"
                ),
            ),
            ha="left",
            va="center",
            fontsize=style[
                "annotation_fontsize"
            ],
        )

    annotation_text = " · ".join(
        (
            f"{metric_labels[metric_name]}: "
            f"{value:.3f}"
        )
        for metric_name, value
        in annotation_values.items()
    )

    figure.text(
        0.5,
        0.055,
        annotation_text,
        ha="center",
        va="bottom",
        fontsize=style[
            "annotation_fontsize"
        ],
    )

    add_source_note(
        figure
    )

    figure.tight_layout(
        rect=(
            0.03,
            0.11,
            0.98,
            0.92,
        )
    )

    payload = {
        "record": record,
        "bar_values": {
            metric_name: value
            for metric_name, value
            in zip(
                bar_metrics,
                values,
                strict=True,
            )
        },
        "annotation_values": (
            annotation_values
        ),
        "bar_count": len(values),
    }

    return figure, payload


def render_delta_bar(
    specification: dict[str, Any],
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    metric_labels: dict[str, str],
    style: dict[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    bar_metrics = specification[
        "bar_metrics"
    ]

    all_metrics = (
        bar_metrics
        + specification[
            "annotation_metrics"
        ]
    )

    deltas = {
        metric_name: (
            metric_value(
                index,
                specification[
                    "minuend"
                ],
                metric_name,
            )
            - metric_value(
                index,
                specification[
                    "subtrahend"
                ],
                metric_name,
            )
        )
        for metric_name in all_metrics
    }

    bar_values = [
        deltas[metric_name]
        for metric_name in bar_metrics
    ]

    figure, axes = plt.subplots(
        figsize=tuple(
            specification[
                "figure_size_inches"
            ]
        )
    )

    add_titles(
        figure,
        axes,
        specification,
        style,
    )

    x_positions = np.arange(
        len(bar_metrics),
        dtype=np.float64,
    )

    container = axes.bar(
        x_positions,
        bar_values,
    )

    axes.set_xticks(
        x_positions,
        [
            metric_labels[
                metric_name
            ]
            for metric_name
            in bar_metrics
        ],
    )

    axes.set_xlabel(
        specification["x_label"]
    )

    axes.set_ylabel(
        specification["y_label"]
    )

    axes.set_ylim(
        *specification["y_limits"]
    )

    if specification[
        "zero_reference_line"
    ]:
        axes.axhline(
            0.0,
            linewidth=0.9,
        )

    axes.grid(
        axis="y",
        alpha=style["grid_alpha"],
    )

    axes.set_axisbelow(True)

    for rectangle, value in zip(
        container,
        bar_values,
        strict=True,
    ):
        axes.text(
            (
                rectangle.get_x()
                + rectangle.get_width()
                / 2.0
            ),
            value - 0.012,
            format(
                value,
                (
                    f".{style['annotation_decimals']}"
                    "f"
                ),
            ),
            ha="center",
            va="top",
            fontsize=style[
                "annotation_fontsize"
            ],
        )

    annotation_text = " · ".join(
        (
            f"Δ {metric_labels[metric_name]}: "
            f"{deltas[metric_name]:.3f}"
        )
        for metric_name
        in specification[
            "annotation_metrics"
        ]
    )

    figure.text(
        0.5,
        0.055,
        annotation_text,
        ha="center",
        va="bottom",
        fontsize=style[
            "annotation_fontsize"
        ],
    )

    add_source_note(
        figure
    )

    figure.tight_layout(
        rect=(
            0.03,
            0.11,
            0.98,
            0.92,
        )
    )

    payload = {
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
            for metric_name in bar_metrics
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
        "bar_count": len(bar_values),
    }

    return figure, payload


def render_scatter(
    specification: dict[str, Any],
    index: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ],
    style: dict[str, Any],
) -> tuple[plt.Figure, dict[str, Any]]:
    records = specification["records"]

    points = [
        {
            "condition": (
                record["condition"]
            ),
            "x": metric_value(
                index,
                record,
                specification[
                    "x_metric"
                ],
            ),
            "y": metric_value(
                index,
                record,
                specification[
                    "y_metric"
                ],
            ),
        }
        for record in records
    ]

    figure, axes = plt.subplots(
        figsize=tuple(
            specification[
                "figure_size_inches"
            ]
        )
    )

    add_titles(
        figure,
        axes,
        specification,
        style,
    )

    offsets = {
        "original_image_full_caption": (
            8,
            -4,
        ),
        "grayscale_image_full_caption": (
            8,
            12,
        ),
        "original_image_caption_without_color": (
            -8,
            -4,
        ),
        "grayscale_image_caption_without_color": (
            -8,
            12,
        ),
    }

    alignments = {
        "original_image_full_caption": (
            "left",
            "center",
        ),
        "grayscale_image_full_caption": (
            "left",
            "bottom",
        ),
        "original_image_caption_without_color": (
            "right",
            "top",
        ),
        "grayscale_image_caption_without_color": (
            "right",
            "bottom",
        ),
    }

    connector_conditions = {
        "original_image_caption_without_color",
        "grayscale_image_caption_without_color",
    }

    for point in points:
        axes.scatter(
            [point["x"]],
            [point["y"]],
            s=72,
        )

        if specification[
            "show_point_labels"
        ]:
            (
                horizontal_alignment,
                vertical_alignment,
            ) = alignments[
                point["condition"]
            ]

            arrow_properties = None

            if (
                point["condition"]
                in connector_conditions
            ):
                arrow_properties = {
                    "arrowstyle": "-",
                    "linewidth": 0.7,
                    "shrinkA": 2,
                    "shrinkB": 5,
                }

            axes.annotate(
                SCATTER_LABELS[
                    point["condition"]
                ],
                (
                    point["x"],
                    point["y"],
                ),
                xytext=offsets[
                    point["condition"]
                ],
                textcoords=(
                    "offset points"
                ),
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=style[
                    "annotation_fontsize"
                ],
                arrowprops=arrow_properties,
            )

    axes.set_xlabel(
        specification["x_label"]
    )

    axes.set_ylabel(
        specification["y_label"]
    )

    axes.set_xlim(
        *specification["x_limits"]
    )

    axes.set_ylim(
        *specification["y_limits"]
    )

    axes.grid(
        axis="both",
        alpha=style["grid_alpha"],
    )

    axes.set_axisbelow(True)

    add_source_note(
        figure
    )

    figure.tight_layout(
        rect=(
            0.03,
            0.05,
            0.98,
            0.92,
        )
    )

    payload = {
        "x_metric": (
            specification["x_metric"]
        ),
        "y_metric": (
            specification["y_metric"]
        ),
        "points": points,
        "point_count": len(points),
    }

    return figure, payload


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


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
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

    assert {
        "experiment_id",
        "source_section",
        "condition",
        "metric_name",
        "metric_value",
        "condition_label",
        "comparability_family",
    }.issubset(metric_fields)

    assert (
        master_summary[
            "generation_valid"
        ]
        is True
    )

    protected_before = inventory_directory(
        PROTECTED_DIRECTORY
    )

    protected_before_digest = (
        inventory_digest(
            protected_before
        )
    )

    style = contract[
        "global_style"
    ]

    configure_matplotlib(
        style
    )

    index = metric_index(
        metric_rows
    )

    metric_labels = contract[
        "metric_labels"
    ]

    formats = contract[
        "output"
    ]["formats"]

    if TEMPORARY_DIRECTORY.exists():
        shutil.rmtree(
            TEMPORARY_DIRECTORY
        )

    TEMPORARY_DIRECTORY.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest_rows = []
    figure_summaries = []

    try:
        for specification in (
            contract["figures"]
        ):
            chart_type = (
                specification[
                    "chart_type"
                ]
            )

            if chart_type == (
                "grouped_bar"
            ):
                figure, data_payload = (
                    render_grouped_bar(
                        specification,
                        index,
                        metric_labels,
                        style,
                    )
                )

            elif chart_type == (
                "horizontal_bar"
            ):
                figure, data_payload = (
                    render_horizontal_bar(
                        specification,
                        index,
                        metric_labels,
                        style,
                    )
                )

            elif chart_type == (
                "delta_bar"
            ):
                figure, data_payload = (
                    render_delta_bar(
                        specification,
                        index,
                        metric_labels,
                        style,
                    )
                )

            elif chart_type == "scatter":
                figure, data_payload = (
                    render_scatter(
                        specification,
                        index,
                        style,
                    )
                )

            else:
                raise AssertionError(
                    "Tipo de figura "
                    f"desconocido: {chart_type}"
                )

            file_records = save_figure(
                figure,
                specification,
                style,
                formats,
                TEMPORARY_DIRECTORY,
            )

            for record in file_records:
                manifest_rows.append(
                    {
                        "manifest_row_index": (
                            len(
                                manifest_rows
                            )
                        ),
                        **record,
                    }
                )

            figure_summaries.append(
                {
                    "figure_id": (
                        specification[
                            "figure_id"
                        ]
                    ),
                    "filename_stem": (
                        specification[
                            "filename_stem"
                        ]
                    ),
                    "comparability_family": (
                        specification[
                            "comparability_family"
                        ]
                    ),
                    "chart_type": (
                        chart_type
                    ),
                    "title": (
                        specification[
                            "title"
                        ]
                    ),
                    "data": data_payload,
                    "files": file_records,
                }
            )

        assert len(
            figure_summaries
        ) == 5

        assert len(
            manifest_rows
        ) == 10

        manifest_path = (
            TEMPORARY_DIRECTORY
            / MANIFEST_FILENAME
        )

        write_csv(
            manifest_path,
            tuple(
                contract[
                    "manifest_fields"
                ]
            ),
            manifest_rows,
        )

        protected_after = (
            inventory_directory(
                PROTECTED_DIRECTORY
            )
        )

        protected_after_digest = (
            inventory_digest(
                protected_after
            )
        )

        assert (
            protected_after
            == protected_before
        )

        summary_payload = {
            "schema_version": "1.0",
            "contract_id": "FG1",
            "dataset_version": "v2",
            "generation_valid": True,
            "counts": {
                "figures": 5,
                "graphic_files": 10,
                "manifest_rows": 10,
                "total_files": 12,
            },
            "formats": list(formats),
            "matplotlib": (
                matplotlib.__version__
            ),
            "style": style,
            "protected_path": {
                "path": (
                    PROTECTED_DIRECTORY
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "file_count": len(
                    protected_before
                ),
                "inventory_sha256_before": (
                    protected_before_digest
                ),
                "inventory_sha256_after": (
                    protected_after_digest
                ),
                "unchanged": True,
            },
            "input_artifacts": {
                key: {
                    "path": (
                        path.relative_to(
                            PROJECT_ROOT
                        ).as_posix()
                    ),
                    "sha256": (
                        sha256_file(path)
                    ),
                }
                for key, path
                in input_paths.items()
            },
            "figures": figure_summaries,
            "output_artifacts": {
                "manifest": {
                    "path": (
                        (
                            OUTPUT_DIRECTORY
                            / MANIFEST_FILENAME
                        )
                        .relative_to(
                            PROJECT_ROOT
                        )
                        .as_posix()
                    ),
                    "rows": 10,
                    "sha256": (
                        sha256_file(
                            manifest_path
                        )
                    ),
                },
                "graphic_files": [
                    {
                        "figure_id": (
                            row["figure_id"]
                        ),
                        "file_format": (
                            row["file_format"]
                        ),
                        "path": (
                            row["relative_path"]
                        ),
                        "sha256": (
                            row["sha256"]
                        ),
                        "size_bytes": (
                            row["size_bytes"]
                        ),
                    }
                    for row in manifest_rows
                ],
            },
        }

        summary_path = (
            TEMPORARY_DIRECTORY
            / SUMMARY_FILENAME
        )

        write_json(
            summary_path,
            summary_payload,
        )

        expected_names = {
            MANIFEST_FILENAME,
            SUMMARY_FILENAME,
        }

        for specification in (
            contract["figures"]
        ):
            for file_format in formats:
                expected_names.add(
                    (
                        f"{specification['filename_stem']}"
                        f".{file_format}"
                    )
                )

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

        assert len(
            actual_names
        ) == 12

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
        "GENERACIÓN DE FIGURAS "
        "PRINCIPALES COMPLETADA"
    )
    print("=" * 92)

    print(
        "Figuras:",
        len(figure_summaries),
    )

    print(
        "Archivos gráficos:",
        len(manifest_rows),
    )

    print(
        "Archivos totales:",
        12,
    )

    print()
    print("Figuras generadas:")

    for figure in figure_summaries:
        print(
            "-",
            figure["figure_id"],
            figure["filename_stem"],
            figure[
                "comparability_family"
            ],
        )

    print()
    print(
        "Directorio protegido 'figures' "
        "sin modificar:",
        True,
    )

    print(
        "Directorio de salida:",
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
