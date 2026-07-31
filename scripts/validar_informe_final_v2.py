"""Valida independientemente el informe final v2."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "informe_final_v2.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reporte_evaluacion_responsable.md"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "tablas_maestras"
    / "metricas_maestras_v2.csv"
)

CASES_PATH = (
    PROJECT_ROOT
    / "results"
    / "casos_analizados.csv"
)

RELIABILITY_PATH = (
    PROJECT_ROOT
    / "results"
    / "pruebas_confiabilidad.csv"
)

EXPLAINABILITY_PATH = (
    PROJECT_ROOT
    / "results"
    / "explicabilidad.csv"
)

PROTECTED_TRACKED_PATHS = (
    "README.md",
    "results/conclusion_tecnica.md",
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

    raw.decode("utf-8-sig")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def build_metric_index(
    rows: list[dict[str, str]],
) -> dict[
    tuple[str, str, str, str],
    float,
]:
    index = {}

    for row in rows:
        key = (
            row["experiment_id"],
            row["source_section"],
            row["condition"],
            row["metric_name"],
        )

        assert key not in index, (
            f"Métrica duplicada: {key}"
        )

        index[key] = float(
            row["metric_value"]
        )

    return index


def get_metric(
    index: dict[
        tuple[str, str, str, str],
        float,
    ],
    experiment_id: str,
    source_section: str,
    condition: str,
    metric_name: str,
) -> float:
    key = (
        experiment_id,
        source_section,
        condition,
        metric_name,
    )

    assert key in index, (
        f"No existe la métrica {key}."
    )

    return index[key]


def format_metric(
    value: float,
) -> str:
    return f"{value:.3f}"


def markdown_row(
    cells: list[str],
) -> str:
    return (
        "| "
        + " | ".join(cells)
        + " |"
    )


def git_head_bytes(
    relative_path: str,
) -> bytes:
    process = subprocess.run(
        [
            "git",
            "show",
            f"HEAD:{relative_path}",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return process.stdout


def normalize_tracked_text(
    data: bytes,
) -> str:
    """Normaliza BOM y finales de línea sin alterar el contenido."""

    text = data.decode(
        "utf-8-sig"
    )

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def git_path_has_diff(
    relative_path: str,
) -> bool:
    process = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "--",
            relative_path,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )

    assert process.returncode in {
        0,
        1,
    }, (
        "git diff falló para "
        f"{relative_path}: "
        f"código {process.returncode}"
    )

    return process.returncode == 1


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
    )

    report_raw = REPORT_PATH.read_bytes()

    assert not report_raw.startswith(
        b"\xef\xbb\xbf"
    ), "El informe contiene BOM."

    assert b"\r\n" not in report_raw, (
        "El informe contiene CRLF."
    )

    report = report_raw.decode(
        "utf-8"
    )

    assert report.endswith("\n")

    assert (
        contract["contract_id"]
        == "RF2"
    )

    assert (
        contract["dataset_version"]
        == "v2"
    )

    metric_rows = load_csv(
        METRICS_PATH
    )

    assert len(metric_rows) == 62

    metrics = build_metric_index(
        metric_rows
    )

    cases = load_csv(
        CASES_PATH
    )

    reliability = load_csv(
        RELIABILITY_PATH
    )

    explainability = load_csv(
        EXPLAINABILITY_PATH
    )

    assert len(cases) == 5
    assert len(reliability) == 10
    assert len(explainability) == 2

    title_lines = re.findall(
        r"^# (.+)$",
        report,
        flags=re.MULTILINE,
    )

    assert title_lines == [
        (
            "Evaluación responsable de "
            "recuperación de texto a imagen sobre "
            "patrones textiles generados"
        )
    ]

    numbered_headings = re.findall(
        r"^## (\d+)\. (.+)$",
        report,
        flags=re.MULTILINE,
    )

    expected_headings = [
        (
            str(section["section_number"]),
            section["title"],
        )
        for section in contract[
            "sections"
        ]
    ]

    assert numbered_headings == (
        expected_headings
    ), (
        "Las secciones del informe no "
        "coinciden con el contrato."
    )

    assert len(numbered_headings) == 10

    image_links = re.findall(
        r"!\[[^\]]*\]\(([^)]+)\)",
        report,
    )

    assert len(image_links) == 7

    assert len(image_links) == len(
        set(image_links)
    )

    required_v2_images = {
        record["path"]
        for record in contract[
            "required_v2_figures"
        ]
    }

    actual_v2_images = {
        link
        for link in image_links
        if link.startswith(
            "results/v2/figuras/"
        )
    }

    assert (
        actual_v2_images
        == required_v2_images
    )

    expected_legacy_images = {
        "figures/ejemplos_evaluados.png",
        "figures/explicabilidad_casos.png",
    }

    actual_legacy_images = (
        set(image_links)
        - actual_v2_images
    )

    assert (
        actual_legacy_images
        == expected_legacy_images
    )

    for image_link in image_links:
        image_path = (
            PROJECT_ROOT
            / image_link
        )

        assert image_path.exists(), (
            f"No existe la imagen "
            f"{image_link}."
        )

        assert image_path.is_file()

        assert (
            image_path.stat().st_size
            > 0
        )

    legacy_quantitative_markers = (
        "0.325",
        "0.600",
        "0.481256",
        "0.026175",
        "0.119300",
    )

    legacy_findings = {
        marker: report.count(marker)
        for marker
        in legacy_quantitative_markers
        if marker in report
    }

    assert not legacy_findings, (
        "Persisten métricas v1: "
        f"{legacy_findings}"
    )

    for claim in contract[
        "forbidden_unqualified_claims"
    ]:
        assert (
            claim.casefold()
            not in report.casefold()
        ), (
            "Afirmación no permitida: "
            f"{claim}"
        )

    for limitation in contract[
        "required_limitations"
    ]:
        expected_line = (
            f"- {limitation}"
        )

        assert expected_line in report, (
            "Falta la limitación: "
            f"{limitation}"
        )

    required_narrative_markers = (
        (
            "OpenCLIP supera los "
            "baselines de azar y color",
            (
                "El resultado supera tanto "
                "al baseline aleatorio como "
                "al histograma HSV."
            ),
        ),
        (
            "Discriminación local",
            (
                "Estos resultados describen "
                "discriminación local"
            ),
        ),
        (
            "Escala de grises perjudica "
            "la recuperación exacta",
            (
                "Eliminar el color visual "
                "reduce R@1"
            ),
        ),
        (
            "Métricas exactas y "
            "estructurales distintas",
            (
                "Las métricas responden "
                "preguntas diferentes."
            ),
        ),
        (
            "Sin superioridad universal",
            (
                "no es válido afirmar que "
                "retirar el color sea "
                "universalmente mejor"
            ),
        ),
        (
            "v1 no es evidencia de E1-E4",
            (
                "no constituyen evidencia "
                "directa de E1 a E4"
            ),
        ),
    )

    for label, marker in (
        required_narrative_markers
    ):
        assert marker in report, (
            f"Falta la interpretación: "
            f"{label}"
        )

    openclip_record = {
        "experiment_id": "E3",
        "source_section": (
            "overall_metrics"
        ),
        "condition": "openclip",
    }

    random_record = {
        "experiment_id": "E3",
        "source_section": (
            "overall_metrics"
        ),
        "condition": "random",
    }

    hsv_record = {
        "experiment_id": "E3",
        "source_section": (
            "overall_metrics"
        ),
        "condition": "color_histogram",
    }

    gray_record = {
        "experiment_id": "E4",
        "source_section": (
            "exact_overall_metrics"
        ),
        "condition": (
            "grayscale_image_full_caption"
        ),
    }

    exact_metrics = (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
    )

    def exact_values(
        record: dict[str, str],
    ) -> list[str]:
        return [
            format_metric(
                get_metric(
                    metrics,
                    record["experiment_id"],
                    record["source_section"],
                    record["condition"],
                    metric_name,
                )
            )
            for metric_name in exact_metrics
        ]

    expected_exact_rows = (
        markdown_row(
            [
                "Aleatorio",
                *exact_values(
                    random_record
                ),
            ]
        ),
        markdown_row(
            [
                "Histograma HSV",
                *exact_values(
                    hsv_record
                ),
            ]
        ),
        markdown_row(
            [
                "OpenCLIP",
                *exact_values(
                    openclip_record
                ),
            ]
        ),
        markdown_row(
            [
                (
                    "OpenCLIP con "
                    "imágenes grises"
                ),
                *exact_values(
                    gray_record
                ),
            ]
        ),
    )

    for expected_row in (
        expected_exact_rows
    ):
        assert expected_row in report, (
            "Falta una fila exacta: "
            f"{expected_row}"
        )

    hard_record = {
        "experiment_id": "E2",
        "source_section": (
            "overall_metrics"
        ),
        "condition": (
            "openclip_hard_negatives"
        ),
    }

    hard_metric_labels = (
        (
            "Exactitud ante "
            "negativos difíciles",
            "hard_negative_accuracy",
        ),
        (
            "MRR",
            "mrr",
        ),
        (
            "nDCG@10",
            "ndcg_at_10",
        ),
        (
            "Victorias pareadas",
            "pairwise_win_rate",
        ),
        (
            (
                "Diferencia pareada "
                "media"
            ),
            "mean_paired_difference",
        ),
    )

    for label, metric_name in (
        hard_metric_labels
    ):
        value = format_metric(
            get_metric(
                metrics,
                hard_record[
                    "experiment_id"
                ],
                hard_record[
                    "source_section"
                ],
                hard_record[
                    "condition"
                ],
                metric_name,
            )
        )

        row_pattern = re.compile(
            r"^\| "
            + re.escape(label)
            + r" \| "
            + re.escape(value)
            + r" \|",
            flags=re.MULTILINE,
        )

        assert row_pattern.search(
            report
        ), (
            "Falta la métrica E2: "
            f"{label}={value}"
        )

    structural_conditions = (
        (
            "original_image_full_caption",
            (
                "Imagen original + "
                "caption completo"
            ),
        ),
        (
            "grayscale_image_full_caption",
            (
                "Imagen gris + "
                "caption completo"
            ),
        ),
        (
            (
                "original_image_"
                "caption_without_color"
            ),
            (
                "Imagen original + "
                "caption sin color"
            ),
        ),
        (
            (
                "grayscale_image_"
                "caption_without_color"
            ),
            (
                "Imagen gris + "
                "caption sin color"
            ),
        ),
    )

    structural_metrics = (
        "structural_hit_at_1",
        "structural_hit_at_5",
        (
            "structural_"
            "fractional_recall_at_5"
        ),
        "structural_mrr",
        "structural_ndcg_at_10",
    )

    for condition, label in (
        structural_conditions
    ):
        values = [
            format_metric(
                get_metric(
                    metrics,
                    "E4",
                    (
                        "structural_"
                        "overall_metrics"
                    ),
                    condition,
                    metric_name,
                )
            )
            for metric_name
            in structural_metrics
        ]

        expected_row = markdown_row(
            [
                label,
                *values,
            ]
        )

        assert expected_row in report, (
            "Falta la fila estructural: "
            f"{condition}"
        )

    metric_anchors = contract[
        "required_metric_anchors"
    ]

    assert len(metric_anchors) == 11

    for anchor in metric_anchors:
        actual_value = get_metric(
            metrics,
            anchor["experiment_id"],
            anchor["source_section"],
            anchor["condition"],
            anchor["metric_name"],
        )

        expected_value = float(
            anchor["expected_value"]
        )

        assert math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    case_ids = [
        row["caso_id"]
        for row in cases
    ]

    assert case_ids == [
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
    ]

    for row in cases:
        assert row["caso_id"] in report
        assert row["image_id"] in report

    reliability_counts = Counter(
        row["prueba"]
        for row in reliability
    )

    assert sum(
        reliability_counts.values()
    ) == 10

    assert len(
        reliability_counts
    ) == 2

    assert (
        "Sensibilidad al texto"
        in report
    )

    assert (
        "Degradación visual"
        in report
    )

    explainability_case_ids = {
        row["caso_id"]
        for row in explainability
    }

    assert explainability_case_ids == {
        "C01",
        "C03",
    }

    traceability_paths = (
        "docs/especificacion_experimental_v2.md",
        "docs/auditoria_visual_patrones_v2.md",
        "docs/diseno_captions_positivos_v2.md",
        "docs/diseno_negativos_dificiles_v2.md",
        "docs/entorno_reproducible_v2.md",
        (
            "results/v2/tablas_maestras/"
            "metricas_maestras_v2.csv"
        ),
        (
            "results/v2/tablas_maestras/"
            "comparaciones_maestras_v2.csv"
        ),
        "results/casos_analizados.csv",
        "results/pruebas_confiabilidad.csv",
        "results/explicabilidad.csv",
    )

    for relative_path in (
        traceability_paths
    ):
        assert relative_path in report

        path = (
            PROJECT_ROOT
            / relative_path
        )

        assert path.exists()
        assert path.is_file()

    word_count = len(
        re.findall(
            r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]+\b",
            report,
        )
    )

    assert 2500 <= word_count
    assert word_count <= 4500

    table_headers = re.findall(
        r"^\| .+ \|$",
        report,
        flags=re.MULTILINE,
    )

    assert len(table_headers) >= 30

    for relative_path in (
        PROTECTED_TRACKED_PATHS
    ):
        working_bytes = (
            PROJECT_ROOT
            / relative_path
        ).read_bytes()

        committed_bytes = git_head_bytes(
            relative_path
        )

        has_git_diff = (
            git_path_has_diff(
                relative_path
            )
        )

        assert not has_git_diff, (
            "Git detecta cambios en el "
            "archivo protegido: "
            f"{relative_path}"
        )

        working_text = (
            normalize_tracked_text(
                working_bytes
            )
        )

        committed_text = (
            normalize_tracked_text(
                committed_bytes
            )
        )

        assert (
            working_text
            == committed_text
        ), (
            "El contenido textual del "
            "archivo protegido cambió: "
            f"{relative_path}"
        )

    print("=" * 92)
    print(
        "VALIDACIÓN INDEPENDIENTE DEL "
        "INFORME FINAL V2 SUPERADA"
    )
    print("=" * 92)

    print(
        "Secciones verificadas:",
        len(numbered_headings),
    )

    print(
        "Figuras v2 verificadas:",
        len(actual_v2_images),
    )

    print(
        "Figuras cualitativas v1:",
        len(actual_legacy_images),
    )

    print(
        "Filas exactas verificadas:",
        len(expected_exact_rows),
    )

    print(
        "Métricas de negativos "
        "difíciles:",
        len(hard_metric_labels),
    )

    print(
        "Condiciones estructurales:",
        len(structural_conditions),
    )

    print(
        "Anclas métricas:",
        len(metric_anchors),
    )

    print(
        "Casos cualitativos:",
        len(cases),
    )

    print(
        "Pruebas de confiabilidad:",
        len(reliability),
    )

    print(
        "Casos de explicabilidad:",
        len(explainability),
    )

    print(
        "Palabras aproximadas:",
        word_count,
    )

    print(
        "Métricas cuantitativas v1:",
        0,
    )

    print(
        "README protegido:",
        True,
    )

    print(
        "Conclusión heredada protegida:",
        True,
    )

    print(
        "Informe válido:",
        True,
    )


if __name__ == "__main__":
    main()
