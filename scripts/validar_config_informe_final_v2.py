"""Valida el contrato documental del informe final v2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "informe_final_v2.json"
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
) -> list[dict[str, str]]:
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
        return list(
            csv.DictReader(file)
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
        == "RF2"
    )

    assert (
        contract["dataset_version"]
        == "v2"
    )

    assert (
        contract["narrative_strategy"]
        == (
            "quantitative_v2_with_"
            "legacy_qualitative_v1"
        )
    )

    primary_task = contract[
        "primary_task"
    ]

    assert (
        primary_task["direction"]
        == "text_to_image"
    )

    assert (
        primary_task["image_count"]
        == 56
    )

    assert (
        primary_task[
            "positive_query_count"
        ]
        == 280
    )

    legacy_block = contract[
        "legacy_qualitative_block"
    ]

    assert (
        legacy_block["dataset_version"]
        == "v1"
    )

    assert (
        legacy_block["direction"]
        == "image_to_text"
    )

    assert (
        legacy_block[
            "must_not_be_presented_as_v2"
        ]
        is True
    )

    sections = contract["sections"]

    assert len(sections) == 10

    assert [
        section["section_number"]
        for section in sections
    ] == list(range(1, 11))

    figure_records = contract[
        "required_v2_figures"
    ]

    assert len(figure_records) == 5

    assert [
        record["figure_id"]
        for record in figure_records
    ] == [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
    ]

    checked_paths = []

    path_groups = (
        contract[
            "required_source_artifacts"
        ],
        legacy_block["artifacts"],
        [
            record["path"]
            for record in figure_records
        ],
    )

    for path_group in path_groups:
        for relative_path in path_group:
            path = (
                PROJECT_ROOT
                / relative_path
            )

            assert path.exists(), (
                f"No existe: {relative_path}"
            )

            assert path.is_file(), (
                f"No es archivo: {relative_path}"
            )

            assert path.stat().st_size > 0

            checked_paths.append(
                relative_path
            )

    assert len(checked_paths) == len(
        set(checked_paths)
    )

    metrics_path = (
        PROJECT_ROOT
        / "results"
        / "v2"
        / "tablas_maestras"
        / "metricas_maestras_v2.csv"
    )

    metric_rows = load_csv(
        metrics_path
    )

    assert len(metric_rows) == 62

    metric_index = {}

    for row in metric_rows:
        key = (
            row["experiment_id"],
            row["source_section"],
            row["condition"],
            row["metric_name"],
        )

        assert key not in metric_index

        metric_index[key] = float(
            row["metric_value"]
        )

    anchors = contract[
        "required_metric_anchors"
    ]

    assert len(anchors) == 11

    for anchor in anchors:
        key = (
            anchor["experiment_id"],
            anchor["source_section"],
            anchor["condition"],
            anchor["metric_name"],
        )

        assert key in metric_index, (
            f"No existe el ancla {key}."
        )

        actual = metric_index[key]

        expected = float(
            anchor["expected_value"]
        )

        assert math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ), (
            f"Ancla distinta {key}: "
            f"actual={actual}, "
            f"esperada={expected}."
        )

    assert len(
        contract[
            "required_interpretations"
        ]
    ) == 6

    assert len(
        contract[
            "required_limitations"
        ]
    ) == 6

    assert len(
        contract[
            "forbidden_unqualified_claims"
        ]
    ) == 6

    encoding_policy = contract[
        "encoding_policy"
    ]

    assert (
        encoding_policy[
            "canonical_encoding"
        ]
        == "utf-8"
    )

    assert (
        encoding_policy["allow_bom"]
        is False
    )

    assert (
        encoding_policy[
            "line_endings"
        ]
        == "lf"
    )

    assert contract["update_order"] == [
        "validate_contract",
        "rewrite_canonical_report",
        "validate_report",
        "update_readme",
        "validate_readme",
        "commit_and_push",
    ]

    print("=" * 92)
    print(
        "VALIDACIÓN DEL CONTRATO DEL "
        "INFORME FINAL SUPERADA"
    )
    print("=" * 92)

    print(
        "Contrato:",
        contract["contract_id"],
    )

    print(
        "Secciones:",
        len(sections),
    )

    print(
        "Figuras v2:",
        len(figure_records),
    )

    print(
        "Anclas métricas:",
        len(anchors),
    )

    print(
        "Artefactos verificados:",
        len(checked_paths),
    )

    print(
        "Dirección principal:",
        primary_task["direction"],
    )

    print(
        "Bloque cualitativo heredado:",
        legacy_block["dataset_version"],
    )

    print(
        "Contrato válido:",
        True,
    )


if __name__ == "__main__":
    main()
