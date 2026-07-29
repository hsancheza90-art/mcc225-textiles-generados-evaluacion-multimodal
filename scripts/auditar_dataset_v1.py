"""Auditoría reproducible del dataset sintético v1 de MCC225.

El script no modifica el dataset. Verifica su integridad y genera evidencia
auditable sobre duplicados visuales, firmas semánticas, repetición textual y
posibles falsos negativos producidos por usar image_id como única relevancia.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifest_textiles_generados.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

SUMMARY_PATH = RESULTS_DIR / "auditoria_dataset_v1.json"
DUPLICATES_PATH = RESULTS_DIR / "duplicados_semanticos_v1.csv"
SHARED_CAPTIONS_PATH = RESULTS_DIR / "captions_compartidos_v1.csv"
REPORT_PATH = DOCS_DIR / "auditoria_dataset_v1.md"

CAPTION_COLUMNS = (
    "caption_1",
    "caption_2",
    "caption_3",
    "caption_4",
    "caption_5",
)

SEMANTIC_COLUMNS = (
    "paleta",
    "composicion",
    "motivo",
    "simetria",
    "nivel_ambiguedad",
)

CONFIGURATION_COLUMNS = (
    "composicion",
    "motivo",
    "simetria",
    "nivel_ambiguedad",
)

REQUIRED_COLUMNS = (
    "image_id",
    "image_path",
    *CAPTION_COLUMNS,
    *SEMANTIC_COLUMNS,
    "observacion",
)


def clean(value: Any) -> str:
    """Normaliza un valor leído del CSV."""

    return str(value).strip()


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Carga y valida la estructura básica del manifiesto."""

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el manifiesto: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(fieldnames)
    )

    if missing_columns:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(missing_columns)
        )

    return rows


def semantic_signature(
    row: dict[str, str],
) -> tuple[str, ...]:
    """Construye la firma semántica completa de una imagen."""

    return tuple(
        clean(row[column])
        for column in SEMANTIC_COLUMNS
    )


def configuration_signature(
    row: dict[str, str],
) -> tuple[str, ...]:
    """Construye la firma visual sin incluir la paleta."""

    return tuple(
        clean(row[column])
        for column in CONFIGURATION_COLUMNS
    )


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def resolve_image_path(raw_path: str) -> Path:
    """Resuelve una ruta del manifiesto respecto de la raíz."""

    path = Path(clean(raw_path))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Exporta un CSV auditable en UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_audit() -> dict[str, Any]:
    """Ejecuta la auditoría completa y genera sus artefactos."""

    rows = load_manifest(MANIFEST_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    image_ids = [clean(row["image_id"]) for row in rows]

    duplicate_image_ids = sorted(
        image_id
        for image_id, count
        in Counter(image_ids).items()
        if count > 1
    )

    missing_caption_records: list[dict[str, str]] = []
    missing_image_records: list[dict[str, str]] = []

    image_hash_by_id: dict[str, str] = {}

    for row in rows:
        image_id = clean(row["image_id"])

        for caption_column in CAPTION_COLUMNS:
            if not clean(row[caption_column]):
                missing_caption_records.append(
                    {
                        "image_id": image_id,
                        "caption_column": caption_column,
                    }
                )

        image_path = resolve_image_path(row["image_path"])

        if not image_path.exists():
            missing_image_records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                }
            )
        else:
            image_hash_by_id[image_id] = sha256_file(
                image_path
            )

    signature_groups: defaultdict[
        tuple[str, ...],
        list[dict[str, str]],
    ] = defaultdict(list)

    configuration_groups: defaultdict[
        tuple[str, ...],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        signature_groups[
            semantic_signature(row)
        ].append(row)

        configuration_groups[
            configuration_signature(row)
        ].append(row)

    duplicate_rows: list[dict[str, Any]] = []
    affected_query_ids: list[str] = []

    duplicate_group_number = 0

    for signature, group in sorted(
        signature_groups.items(),
        key=lambda item: min(
            clean(row["image_id"])
            for row in item[1]
        ),
    ):
        if len(group) <= 1:
            continue

        duplicate_group_number += 1

        group_image_ids = [
            clean(row["image_id"])
            for row in group
        ]

        hashes = [
            image_hash_by_id[image_id]
            for image_id in group_image_ids
            if image_id in image_hash_by_id
        ]

        same_exact_image = (
            len(hashes) == len(group_image_ids)
            and len(set(hashes)) == 1
        )

        affected_query_ids.extend(group_image_ids)

        for row in group:
            duplicate_rows.append(
                {
                    "semantic_group": (
                        f"SG{duplicate_group_number:03d}"
                    ),
                    "image_id": clean(row["image_id"]),
                    "image_path": clean(row["image_path"]),
                    "paleta": clean(row["paleta"]),
                    "composicion": clean(
                        row["composicion"]
                    ),
                    "motivo": clean(row["motivo"]),
                    "simetria": clean(row["simetria"]),
                    "nivel_ambiguedad": clean(
                        row["nivel_ambiguedad"]
                    ),
                    "group_size": len(group),
                    "same_exact_image": same_exact_image,
                    "sha256": image_hash_by_id.get(
                        clean(row["image_id"]),
                        "",
                    ),
                }
            )

    caption_records: list[dict[str, Any]] = []

    for row in rows:
        image_id = clean(row["image_id"])
        signature = semantic_signature(row)

        for caption_column in CAPTION_COLUMNS:
            caption_records.append(
                {
                    "image_id": image_id,
                    "caption_column": caption_column,
                    "caption": clean(row[caption_column]),
                    "semantic_signature": signature,
                }
            )

    caption_groups: defaultdict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in caption_records:
        caption_groups[record["caption"]].append(record)

    shared_caption_rows: list[dict[str, Any]] = []

    for caption, group in sorted(
        caption_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    ):
        image_id_group = sorted(
            {
                record["image_id"]
                for record in group
            }
        )

        if len(image_id_group) <= 1:
            continue

        signatures = {
            record["semantic_signature"]
            for record in group
        }

        columns = sorted(
            {
                record["caption_column"]
                for record in group
            }
        )

        shared_caption_rows.append(
            {
                "caption": caption,
                "occurrences": len(group),
                "image_count": len(image_id_group),
                "semantic_signature_count": len(signatures),
                "crosses_semantic_signatures": (
                    len(signatures) > 1
                ),
                "caption_columns": "|".join(columns),
                "image_ids": "|".join(image_id_group),
            }
        )

    hash_groups: defaultdict[str, list[str]] = defaultdict(list)

    for image_id, digest in image_hash_by_id.items():
        hash_groups[digest].append(image_id)

    exact_duplicate_groups = [
        sorted(group)
        for group in hash_groups.values()
        if len(group) > 1
    ]

    unique_palettes = sorted(
        {
            clean(row["paleta"])
            for row in rows
        }
    )

    unique_configurations = sorted(
        configuration_groups.keys()
    )

    expected_cycle_length = math.lcm(
        len(unique_palettes),
        len(unique_configurations),
    )

    caption_unique_by_column = {
        column: len(
            {
                clean(row[column])
                for row in rows
            }
        )
        for column in CAPTION_COLUMNS
    }

    methodological_warnings: list[str] = []

    if exact_duplicate_groups:
        methodological_warnings.append(
            "Se detectaron im?genes exactamente duplicadas."
        )

    if any(
        len(group) > 1
        for group in signature_groups.values()
    ):
        methodological_warnings.append(
            "Se detectaron firmas sem?nticas repetidas."
        )

    if any(
        bool(row["crosses_semantic_signatures"])
        for row in shared_caption_rows
    ):
        methodological_warnings.append(
            "Existen captions id?nticos que describen "
            "varias firmas sem?nticas."
        )

    if len(rows) > expected_cycle_length:
        methodological_warnings.append(
            "El n?mero de registros supera la longitud "
            "del ciclo conjunto de configuraciones y paletas."
        )

    summary: dict[str, Any] = {
        "dataset_version": "v1",
        "manifest_path": (
            MANIFEST_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "manifest_rows": len(rows),
        "image_ids_unique": len(set(image_ids)),
        "duplicate_image_ids": duplicate_image_ids,
        "caption_columns": list(CAPTION_COLUMNS),
        "captions_expected_per_image": len(
            CAPTION_COLUMNS
        ),
        "missing_caption_count": len(
            missing_caption_records
        ),
        "missing_image_count": len(
            missing_image_records
        ),
        "caption_slots_total": len(caption_records),
        "caption_texts_unique_global": len(
            caption_groups
        ),
        "caption_slots_repeated": (
            len(caption_records) - len(caption_groups)
        ),
        "caption_unique_by_column": (
            caption_unique_by_column
        ),
        "shared_caption_text_count": len(
            shared_caption_rows
        ),
        "shared_caption_cross_signature_count": sum(
            bool(row["crosses_semantic_signatures"])
            for row in shared_caption_rows
        ),
        "semantic_signatures_unique": len(
            signature_groups
        ),
        "semantic_signatures_repeated": len(
            [
                group
                for group in signature_groups.values()
                if len(group) > 1
            ]
        ),
        "queries_affected_by_semantic_duplicates": len(
            set(affected_query_ids)
        ),
        "positive_captions_current_per_query": 5,
        "positive_captions_semantic_duplicate_group": 10,
        "image_files_hashed": len(image_hash_by_id),
        "image_hashes_unique": len(hash_groups),
        "exact_duplicate_image_groups": len(
            exact_duplicate_groups
        ),
        "exact_duplicate_image_pairs": (
            exact_duplicate_groups
        ),
        "palette_count": len(unique_palettes),
        "configuration_count": len(
            unique_configurations
        ),
        "expected_joint_cycle_length": (
            expected_cycle_length
        ),
        "records_after_first_complete_cycle": max(
            0,
            len(rows) - expected_cycle_length,
        ),
        "structural_integrity_ok": (
            not duplicate_image_ids
            and not missing_caption_records
            and not missing_image_records
        ),
        "methodological_validity_ok": (
            not methodological_warnings
        ),
        "methodological_warnings": (
            methodological_warnings
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(
        DUPLICATES_PATH,
        [
            "semantic_group",
            "image_id",
            "image_path",
            "paleta",
            "composicion",
            "motivo",
            "simetria",
            "nivel_ambiguedad",
            "group_size",
            "same_exact_image",
            "sha256",
        ],
        duplicate_rows,
    )

    write_csv(
        SHARED_CAPTIONS_PATH,
        [
            "caption",
            "occurrences",
            "image_count",
            "semantic_signature_count",
            "crosses_semantic_signatures",
            "caption_columns",
            "image_ids",
        ],
        shared_caption_rows,
    )

    return summary


def build_markdown_report(
    summary: dict[str, Any],
) -> str:
    """Construye el informe breve de auditoría."""

    duplicate_pairs = "\n".join(
        f"- `{', '.join(group)}`"
        for group in summary[
            "exact_duplicate_image_pairs"
        ]
    )

    return f"""# Auditoría del dataset sintético v1

## Propósito

Esta auditoría revisa la integridad, unicidad y unidad de relevancia
del dataset heredado de la Actividad 5. No modifica imágenes ni
captions.

## Resultados principales

| Indicador | Resultado |
|---|---:|
| Registros del manifiesto | {summary["manifest_rows"]} |
| Identificadores únicos | {summary["image_ids_unique"]} |
| Captions almacenados | {summary["caption_slots_total"]} |
| Textos de caption únicos | {summary["caption_texts_unique_global"]} |
| Firmas semánticas únicas | {summary["semantic_signatures_unique"]} |
| Firmas semánticas repetidas | {summary["semantic_signatures_repeated"]} |
| Imágenes únicas por SHA-256 | {summary["image_hashes_unique"]} |
| Grupos de imágenes exactas | {summary["exact_duplicate_image_groups"]} |
| Consultas afectadas por duplicación semántica | {summary["queries_affected_by_semantic_duplicates"]} |
| Longitud esperada del ciclo conjunto | {summary["expected_joint_cycle_length"]} |

## Origen del ciclo

El generador combina {summary["configuration_count"]} configuraciones
y {summary["palette_count"]} paletas mediante índices modulares
independientes. La combinación completa se repite cada
{summary["expected_joint_cycle_length"]} registros.

Como el dataset contiene {summary["manifest_rows"]} registros,
los últimos {summary["records_after_first_complete_cycle"]} vuelven
a utilizar combinaciones ya observadas.

## Duplicados exactos

{duplicate_pairs}

## Problema de relevancia

La evaluación original asigna cinco captions positivos a cada
`image_id`. Sin embargo, {summary["queries_affected_by_semantic_duplicates"]}
consultas pertenecen a firmas repetidas. En esos casos, otros cinco
captions corresponden a la misma firma semántica y pueden ser
contabilizados como negativos por el protocolo basado solamente en
`image_id`.

## Repetición textual

De {summary["caption_slots_total"]} slots de captions, únicamente
{summary["caption_texts_unique_global"]} textos son únicos. Además,
{summary["shared_caption_cross_signature_count"]} textos compartidos
describen más de una firma semántica.

Esto muestra que algunas plantillas son demasiado generales para
evaluar discriminación visual fina.

## Decisión metodológica

El dataset v1 se conserva como línea base histórica. La evaluación
final utilizará un dataset v2 con:

- unidad semántica explícita;
- control de duplicados;
- captions positivos discriminativos;
- negativos que modifiquen un único atributo;
- métricas por patrón y nivel de ambigüedad;
- prueba con configuraciones no vistas.

## Artefactos

- `results/auditoria_dataset_v1.json`
- `results/duplicados_semanticos_v1.csv`
- `results/captions_compartidos_v1.csv`
"""

    
def main() -> None:
    """Punto de entrada."""

    summary = build_audit()

    report = build_markdown_report(summary)

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("=" * 72)
    print("AUDITORÍA DEL DATASET V1 COMPLETADA")
    print("=" * 72)

    print(
        f"Registros:                 "
        f"{summary['manifest_rows']}"
    )
    print(
        f"Firmas semánticas únicas:  "
        f"{summary['semantic_signatures_unique']}"
    )
    print(
        f"Hashes de imagen únicos:   "
        f"{summary['image_hashes_unique']}"
    )
    print(
        f"Textos de caption únicos:  "
        f"{summary['caption_texts_unique_global']}"
    )
    print(
        f"Consultas afectadas:       "
        f"{summary['queries_affected_by_semantic_duplicates']}"
    )
    print(
        f"Ciclo conjunto esperado:   "
        f"{summary['expected_joint_cycle_length']}"
    )

    print("\nArtefactos generados:")
    print(
        f"- {SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"- {DUPLICATES_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"- {SHARED_CAPTIONS_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"- {REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
