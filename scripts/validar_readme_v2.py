"""Valida independientemente el README principal del benchmark v2."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

README_PATH = PROJECT_ROOT / "README.md"
CONTRACT_PATH = PROJECT_ROOT / "config" / "informe_final_v2.json"
METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "tablas_maestras"
    / "metricas_maestras_v2.csv"
)

PROTECTED_TRACKED_PATHS = (
    "reporte_evaluacion_responsable.md",
    "results/conclusion_tecnica.md",
)

README_HITO_CHANGED_PATHS = frozenset(
    {
        "README.md",
        "scripts/generar_readme_v2.py",
        "scripts/validar_readme_v2.py",
    }
)

AUDIT_HITO_CHANGED_PATHS = frozenset(
    {
        "scripts/validar_ablaciones_v2.py",
        "scripts/validar_global_openclip_v2.py",
        "scripts/validar_readme_v2.py",
    }
)

DELIVERY_HITO_CHANGED_PATHS = frozenset(
    {
        "config/entrega_final_v2.json",
        "results/v2/manifiesto_entrega_v2.json",
        "scripts/generar_manifiesto_entrega_v2.py",
        "scripts/validar_ablaciones_v2.py",
        "scripts/validar_global_openclip_v2.py",
        "scripts/validar_manifiesto_entrega_v2.py",
        "scripts/validar_readme_v2.py",
    }
)

CLEAN_DELIVERY_CHANGED_PATHS = frozenset()

ALLOWED_CHANGED_PATH_SETS = (
    README_HITO_CHANGED_PATHS,
    AUDIT_HITO_CHANGED_PATHS,
    DELIVERY_HITO_CHANGED_PATHS,
    CLEAN_DELIVERY_CHANGED_PATHS,
)


def assert_allowed_changed_paths(
    actual_changed_paths: set[str],
) -> None:
    normalized_paths = frozenset(
        actual_changed_paths
    )

    assert (
        normalized_paths
        in ALLOWED_CHANGED_PATH_SETS
    ), (
        "El conjunto de cambios no corresponde "
        "a un hito permitido para validar el README. "
        f"Actual={sorted(normalized_paths)}, "
        "permitidos="
        f"{[sorted(paths) for paths in ALLOWED_CHANGED_PATH_SETS]}."
    )


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path}: contiene BOM."
    assert b"\r\n" not in raw, f"{path}: contiene CRLF."

    return json.loads(raw.decode("utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path}: contiene BOM."
    assert b"\r\n" not in raw, f"{path}: contiene CRLF."

    raw.decode("utf-8")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_metric_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], float]:
    index: dict[tuple[str, str, str, str], float] = {}

    for row in rows:
        key = (
            row["experiment_id"],
            row["source_section"],
            row["condition"],
            row["metric_name"],
        )

        assert key not in index, f"Métrica duplicada: {key}"
        index[key] = float(row["metric_value"])

    return index


def get_metric(
    index: dict[tuple[str, str, str, str], float],
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

    assert key in index, f"No existe la métrica {key}."
    return index[key]


def fmt(value: float) -> str:
    return f"{value:.3f}"


def normalize_tracked_text(data: bytes) -> str:
    text = data.decode("utf-8-sig")

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def git_head_bytes(relative_path: str) -> bytes:
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


def git_path_has_diff(relative_path: str) -> bool:
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

    assert process.returncode in {0, 1}, (
        f"git diff falló para {relative_path}: "
        f"código {process.returncode}"
    )

    return process.returncode == 1


def main() -> None:
    # Evita signos que no se usarán en la redacción del README.
    readme_style_text = Path(README_PATH).read_text(
        encoding="utf-8"
    )
    forbidden_readme_characters = {
        chr(0x2192): "flecha derecha",
        chr(0x2013): "guion mediano",
        chr(0x2014): "guion largo",
    }
    forbidden_readme_counts = {}

    for character, label in forbidden_readme_characters.items():
        count = readme_style_text.count(character)

        if count > 0:
            forbidden_readme_counts[label] = count

    assert not forbidden_readme_counts, (
        "El README contiene signos de estilo "
        "no permitidos: "
        f"{forbidden_readme_counts}"
    )

    contract = load_json(CONTRACT_PATH)
    metric_rows = load_csv(METRICS_PATH)
    metrics = build_metric_index(metric_rows)

    raw = README_PATH.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf"), "README.md contiene BOM."
    assert b"\r\n" not in raw, "README.md contiene CRLF."

    text = raw.decode("utf-8")

    assert text.endswith("\n")
    assert contract["contract_id"] == "RF2"
    assert contract["dataset_version"] == "v2"
    assert len(metric_rows) == 62

    title_lines = re.findall(
        r"^# (.+)$",
        text,
        flags=re.MULTILINE,
    )

    assert title_lines == [
        "Evaluación multimodal de patrones textiles generados"
    ]

    actual_sections = re.findall(
        r"^## (\d+)\. (.+)$",
        text,
        flags=re.MULTILINE,
    )

    expected_titles = (
        "Objetivo",
        "Tarea multimodal",
        "Dataset v2",
        "Experimentos",
        "Resultados principales",
        "Informe final",
        "Tablas maestras",
        "Estructura principal",
        "Entorno reproducible",
        "Validación",
        "Evidencia cualitativa heredada",
        "Limitaciones",
        "Uso responsable",
        "Estado del proyecto",
        "Alcance de las conclusiones",
        "Repositorio",
    )

    expected_sections = [
        (str(index), title)
        for index, title in enumerate(
            expected_titles,
            start=1,
        )
    ]

    assert actual_sections == expected_sections, (
        "Las secciones del README no coinciden "
        "con la estructura prevista."
    )

    legacy_metrics = (
        "0.325",
        "0.600",
        "0.481256",
        "0.026175",
        "0.119300",
    )

    legacy_findings = {
        marker: text.count(marker)
        for marker in legacy_metrics
        if marker in text
    }

    assert not legacy_findings, (
        f"Persisten métricas cuantitativas v1: {legacy_findings}"
    )

    for claim in contract["forbidden_unqualified_claims"]:
        assert claim.casefold() not in text.casefold(), (
            f"Afirmación no permitida en README: {claim}"
        )

    exact_metric_names = (
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "ndcg_at_10",
    )

    conditions = (
        ("random", "Aleatorio"),
        ("color_histogram", "Histograma HSV"),
        ("openclip", "OpenCLIP"),
    )

    for condition, label in conditions:
        values = [
            fmt(
                get_metric(
                    metrics,
                    "E3",
                    "overall_metrics",
                    condition,
                    metric_name,
                )
            )
            for metric_name in exact_metric_names
        ]

        if condition == "openclip":
            expected_row = (
                f"| {label} | "
                f"**{values[0]}** | "
                f"**{values[1]}** | "
                f"**{values[2]}** | "
                f"**{values[3]}** |"
            )
        else:
            expected_row = (
                f"| {label} | "
                + " | ".join(values)
                + " |"
            )

        assert expected_row in text, (
            f"Falta la fila de resultados para {condition}."
        )

    hard_metrics = (
        ("Exactitud", "hard_negative_accuracy"),
        ("MRR", "mrr"),
        ("nDCG@10", "ndcg_at_10"),
        ("Tasa de victorias pareadas", "pairwise_win_rate"),
    )

    for label, metric_name in hard_metrics:
        value = fmt(
            get_metric(
                metrics,
                "E2",
                "overall_metrics",
                "openclip_hard_negatives",
                metric_name,
            )
        )

        expected_row = f"| {label} | {value} |"
        assert expected_row in text, (
            f"Falta la métrica de negativos difíciles: "
            f"{label}={value}"
        )

    openclip_r1 = fmt(
        get_metric(
            metrics,
            "E3",
            "overall_metrics",
            "openclip",
            "recall_at_1",
        )
    )

    gray_r1 = fmt(
        get_metric(
            metrics,
            "E4",
            "exact_overall_metrics",
            "grayscale_image_full_caption",
            "recall_at_1",
        )
    )

    structural_hit_1 = fmt(
        get_metric(
            metrics,
            "E4",
            "structural_overall_metrics",
            "grayscale_image_caption_without_color",
            "structural_hit_at_1",
        )
    )

    assert (
        f"reduce el R@1 exacto de **{openclip_r1}** "
        f"a **{gray_r1}**"
    ) in text

    assert (
        f"alcanza Hit@1 = **{structural_hit_1}**"
    ) in text

    required_markers = (
        "recuperación **de texto a imagen**",
        "56 imágenes sintéticas",
        "Consultas positivas | 280",
        "protocolo inicial v1",
        "no se presentan como resultados cuantitativos de v2",
        "reporte_evaluacion_responsable.md",
        "results/v2/tablas_maestras/",
        "results/v2/figuras/",
        "henry/examen-final-mcc225",
    )

    for marker in required_markers:
        assert marker in text, f"Falta el marcador: {marker}"

    required_paths = (
        "docs/especificacion_experimental_v2.md",
        "docs/auditoria_visual_patrones_v2.md",
        "docs/diseno_captions_positivos_v2.md",
        "docs/diseno_negativos_dificiles_v2.md",
        "docs/entorno_reproducible_v2.md",
        "results/v2/entorno_cpu_pip_freeze.txt",
        "reporte_evaluacion_responsable.md",
        "results/v2/figuras/f1_recuperacion_exacta_v2.png",
        "results/v2/figuras/f2_negativos_dificiles_v2.png",
        "results/v2/figuras/f3_ablaciones_estructurales_v2.png",
        "results/v2/figuras/f4_efecto_grises_exacto_v2.png",
        "results/v2/figuras/f5_compromiso_hit1_hit5_v2.png",
    )

    for relative_path in required_paths:
        assert relative_path in text, (
            f"README no referencia el artefacto: {relative_path}"
        )

        path = PROJECT_ROOT / relative_path

        assert path.exists(), f"No existe: {relative_path}"
        assert path.is_file(), f"No es archivo: {relative_path}"
        assert path.stat().st_size > 0

    image_links = re.findall(
        r"!\[[^\]]*\]\(([^)]+)\)",
        text,
    )

    assert image_links == [
        "results/v2/figuras/f1_recuperacion_exacta_v2.png"
    ]

    markdown_links = re.findall(
        r"(?<!!)\[[^\]]+\]\(([^)]+)\)",
        text,
    )

    expected_links = {
        "reporte_evaluacion_responsable.md",
        "results/v2/figuras/f1_recuperacion_exacta_v2.png",
        "results/v2/figuras/f2_negativos_dificiles_v2.png",
        "results/v2/figuras/f3_ablaciones_estructurales_v2.png",
        "results/v2/figuras/f4_efecto_grises_exacto_v2.png",
        "results/v2/figuras/f5_compromiso_hit1_hit5_v2.png",
    }

    assert set(markdown_links) == expected_links

    for link in markdown_links:
        path = PROJECT_ROOT / link

        assert path.exists(), f"Enlace local roto: {link}"
        assert path.is_file(), f"Enlace no apunta a archivo: {link}"

    assert text.count("```") % 2 == 0
    assert text.count("```") >= 8

    tree_markers = (
        "├── config/",
        "├── data/",
        "├── docs/",
        "├── figures/",
        "├── notebooks/",
        "├── results/",
        "├── scripts/",
        "├── tests/",
    )

    for marker in tree_markers:
        assert marker in text, f"Falta en el árbol: {marker}"

    status_rows = (
        "| Dataset v2 | Completo y auditado |",
        "| Embeddings | Congelados |",
        "| E1 a E4 | Evaluados |",
        "| Tablas maestras | Validadas |",
        "| Figuras F1 a F5 | Validadas |",
        "| Informe final v2 | Validado |",
    )

    for row in status_rows:
        assert row in text, f"Falta la fila de estado: {row}"

    word_count = len(
        re.findall(
            r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]+\b",
            text,
        )
    )

    assert 650 <= word_count <= 1100

    status_lines = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()

    changed_paths = []

    for status_line in status_lines:
        assert len(status_line) >= 4, (
            "Línea inesperada de git status: "
            f"{status_line!r}"
        )

        path_text = (
            status_line[3:]
            .strip()
            .replace("\\", "/")
        )

        if " -> " in path_text:
            path_text = path_text.split(
                " -> ",
                maxsplit=1,
            )[1]

        changed_paths.append(
            path_text
        )

    actual_changed_paths = set(
        changed_paths
    )

    assert_allowed_changed_paths(
        actual_changed_paths
    )

    for relative_path in PROTECTED_TRACKED_PATHS:
        assert not git_path_has_diff(relative_path), (
            f"Git detecta cambios en el archivo protegido: "
            f"{relative_path}"
        )

        working = normalize_tracked_text(
            (PROJECT_ROOT / relative_path).read_bytes()
        )
        committed = normalize_tracked_text(
            git_head_bytes(relative_path)
        )

        assert working == committed, (
            f"Cambió el contenido del archivo protegido: "
            f"{relative_path}"
        )

    print("=" * 92)
    print("VALIDACIÓN INDEPENDIENTE DEL README V2 SUPERADA")
    print("=" * 92)
    print("Secciones verificadas:", len(actual_sections))
    print("Filas exactas verificadas:", len(conditions))
    print("Métricas de negativos difíciles:", len(hard_metrics))
    print("Figura embebida:", len(image_links))
    print("Enlaces locales verificados:", len(markdown_links))
    print("Artefactos referenciados:", len(required_paths))
    print("Palabras aproximadas:", word_count)
    print("Métricas cuantitativas v1:", 0)
    print("BOM:", False)
    print("CRLF:", False)
    print("Informe final protegido:", True)
    print("Conclusión heredada protegida:", True)
    print("README válido:", True)


if __name__ == "__main__":
    main()
