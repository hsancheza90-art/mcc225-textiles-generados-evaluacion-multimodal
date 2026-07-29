"""Construye el plan determinista de registros del dataset v2.

Esta etapa no genera imágenes ni captions. Expande explícitamente el
producto cartesiano de patrones y paletas definido en la configuración,
asigna identificadores, semillas y splits, y exporta artefactos auditables.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "experimento_v2.json"

DATA_DIR = PROJECT_ROOT / "data" / "v2"
RESULTS_DIR = PROJECT_ROOT / "results" / "v2"

PLAN_CSV_PATH = DATA_DIR / "plan_registros_v2.csv"
PLAN_JSON_PATH = DATA_DIR / "plan_registros_v2.json"
SUMMARY_PATH = RESULTS_DIR / "resumen_plan_dataset_v2.json"

SPLIT_ORDER = (
    "id",
    "ood_palette",
    "ood_pattern",
    "ood_both",
)

SEMANTIC_FIELDS = (
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
)

CSV_FIELDS = (
    "image_id",
    "semantic_id",
    "semantic_signature",
    "pattern_id",
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "ambiguity_level",
    "split",
    "seed",
    "image_width",
    "image_height",
    "image_path",
    "generator_version",
    "source",
    "usage_restriction",
)


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON UTF-8."""

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def build_semantic_signature(
    record: dict[str, Any],
) -> str:
    """Construye la firma canónica de los cinco atributos."""

    return "|".join(
        f"{field}={record[field]}"
        for field in SEMANTIC_FIELDS
    )


def build_semantic_id(signature: str) -> str:
    """Construye un identificador estable a partir de la firma."""

    digest = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:16].upper()

    return f"SEM-{digest}"


def build_records(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expande explícitamente los splits configurados."""

    dataset = config["dataset_v2"]
    seed_base = int(config["reproducibility"]["seed"])

    width = int(dataset["image_width"])
    height = int(dataset["image_height"])

    records: list[dict[str, Any]] = []
    record_index = 0

    for split_name in SPLIT_ORDER:
        split_config = dataset["splits"][split_name]

        pattern_source = split_config["pattern_source"]
        palette_source = split_config["palette_source"]

        patterns = dataset[pattern_source]
        palettes = dataset[palette_source]

        split_start = len(records)

        for pattern in patterns:
            for palette_id in palettes:
                record_index += 1

                image_id = f"V2_{record_index:03d}"

                record: dict[str, Any] = {
                    "image_id": image_id,
                    "pattern_id": pattern["pattern_id"],
                    "palette_id": palette_id,
                    "motif": pattern["motif"],
                    "orientation": pattern["orientation"],
                    "composition": pattern["composition"],
                    "symmetry": pattern["symmetry"],
                    "ambiguity_level": (
                        pattern["ambiguity_level"]
                    ),
                    "split": split_name,
                    "seed": seed_base + record_index,
                    "image_width": width,
                    "image_height": height,
                    "image_path": (
                        f"data/v2/images/{image_id}.png"
                    ),
                    "generator_version": "2.0",
                    "source": "synthetic_rule_based",
                    "usage_restriction": (
                        "Solo evaluación académica controlada; "
                        "no usar para identificación cultural "
                        "ni validación patrimonial."
                    ),
                }

                signature = build_semantic_signature(record)

                record["semantic_signature"] = signature
                record["semantic_id"] = build_semantic_id(
                    signature
                )

                ordered_record = {
                    field: record[field]
                    for field in CSV_FIELDS
                }

                records.append(ordered_record)

        generated_in_split = len(records) - split_start
        expected_in_split = int(
            split_config["expected_images"]
        )

        if generated_in_split != expected_in_split:
            raise AssertionError(
                f"{split_name}: se generaron "
                f"{generated_in_split} registros, pero "
                f"se esperaban {expected_in_split}."
            )

    expected_total = int(dataset["expected_images"])

    if len(records) != expected_total:
        raise AssertionError(
            f"Se generaron {len(records)} registros, "
            f"pero se esperaban {expected_total}."
        )

    return records


def validate_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Valida unicidad, balance y correspondencia de splits."""

    dataset = config["dataset_v2"]

    image_ids = [record["image_id"] for record in records]
    semantic_ids = [
        record["semantic_id"]
        for record in records
    ]
    signatures = [
        record["semantic_signature"]
        for record in records
    ]
    image_paths = [
        record["image_path"]
        for record in records
    ]
    seeds = [record["seed"] for record in records]

    pattern_palette_pairs = [
        (
            record["pattern_id"],
            record["palette_id"],
        )
        for record in records
    ]

    assert len(image_ids) == len(set(image_ids))
    assert len(semantic_ids) == len(set(semantic_ids))
    assert len(signatures) == len(set(signatures))
    assert len(image_paths) == len(set(image_paths))
    assert len(seeds) == len(set(seeds))

    assert len(pattern_palette_pairs) == len(
        set(pattern_palette_pairs)
    )

    counts_by_split = Counter(
        record["split"]
        for record in records
    )

    counts_by_pattern = Counter(
        record["pattern_id"]
        for record in records
    )

    counts_by_palette = Counter(
        record["palette_id"]
        for record in records
    )

    expected_split_counts = {
        split_name: int(
            dataset["splits"][split_name][
                "expected_images"
            ]
        )
        for split_name in SPLIT_ORDER
    }

    assert dict(counts_by_split) == expected_split_counts

    pattern_counts = set(counts_by_pattern.values())
    palette_counts = set(counts_by_palette.values())

    assert len(counts_by_pattern) == 8, (
        f"Se esperaban 8 patrones, pero se encontraron "
        f"{len(counts_by_pattern)}."
    )

    assert len(counts_by_palette) == 7, (
        f"Se esperaban 7 paletas, pero se encontraron "
        f"{len(counts_by_palette)}."
    )

    assert pattern_counts == {7}, (
        "Cada patrón debe aparecer exactamente "
        "con las siete paletas."
    )

    assert palette_counts == {8}, (
        "Cada paleta debe aparecer exactamente "
        "con los ocho patrones."
    )

    return {
        "counts_by_split": dict(counts_by_split),
        "counts_by_pattern": dict(
            sorted(counts_by_pattern.items())
        ),
        "counts_by_palette": dict(
            sorted(counts_by_palette.items())
        ),
        "image_ids_unique": (
            len(image_ids) == len(set(image_ids))
        ),
        "semantic_ids_unique": (
            len(semantic_ids) == len(set(semantic_ids))
        ),
        "semantic_signatures_unique": (
            len(signatures) == len(set(signatures))
        ),
        "image_paths_unique": (
            len(image_paths) == len(set(image_paths))
        ),
        "seeds_unique": (
            len(seeds) == len(set(seeds))
        ),
        "pattern_palette_pairs_unique": (
            len(pattern_palette_pairs)
            == len(set(pattern_palette_pairs))
        ),
        "patterns_balanced": pattern_counts == {7},
        "palettes_balanced": palette_counts == {8},
    }


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Exporta el plan en formato tabular."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(CSV_FIELDS),
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Exporta JSON estándar UTF-8 sin BOM."""

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Genera y valida el plan del dataset v2."""

    config = load_json(CONFIG_PATH)
    records = build_records(config)
    validation = validate_records(records, config)

    config_hash = sha256_file(CONFIG_PATH)

    plan_payload = {
        "schema_version": "1.0",
        "protocol_version": config["protocol_version"],
        "config_path": (
            CONFIG_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "config_sha256": config_hash,
        "record_count": len(records),
        "records": records,
    }

    write_csv(PLAN_CSV_PATH, records)
    write_json(PLAN_JSON_PATH, plan_payload)

    summary = {
        "schema_version": "1.0",
        "dataset_version": "v2",
        "stage": "record_plan",
        "config_sha256": config_hash,
        "record_count": len(records),
        "expected_positive_captions": (
            len(records)
            * int(
                config["dataset_v2"][
                    "positive_captions_per_image"
                ]
            )
        ),
        "expected_hard_negatives": (
            len(records)
            * int(
                config["dataset_v2"][
                    "hard_negatives_per_query"
                ]
            )
        ),
        **validation,
        "plan_csv_path": (
            PLAN_CSV_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "plan_json_path": (
            PLAN_JSON_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "plan_csv_sha256": sha256_file(
            PLAN_CSV_PATH
        ),
        "plan_json_sha256": sha256_file(
            PLAN_JSON_PATH
        ),
        "plan_valid": all(
            (
                validation["image_ids_unique"],
                validation["semantic_ids_unique"],
                validation[
                    "semantic_signatures_unique"
                ],
                validation["image_paths_unique"],
                validation["seeds_unique"],
                validation[
                    "pattern_palette_pairs_unique"
                ],
                validation["patterns_balanced"],
                validation["palettes_balanced"],
            )
        ),
    }

    write_json(SUMMARY_PATH, summary)

    print("=" * 76)
    print("PLAN DEL DATASET V2 GENERADO")
    print("=" * 76)
    print(f"Registros:              {len(records)}")
    print(
        f"Splits:                 "
        f"{validation['counts_by_split']}"
    )
    print(
        f"Patrones representados: "
        f"{len(validation['counts_by_pattern'])}"
    )
    print(
        f"Paletas representadas:  "
        f"{len(validation['counts_by_palette'])}"
    )
    print("Registros por patrón:   7")
    print("Registros por paleta:   8")
    print(
        "Pares patrón-paleta:    "
        "todos únicos"
    )
    print(
        "Captions previstos:     "
        f"{summary['expected_positive_captions']}"
    )
    print(
        "Negativos previstos:    "
        f"{summary['expected_hard_negatives']}"
    )
    print(f"Plan válido:            {summary['plan_valid']}")

    print("\nArtefactos generados:")
    print(
        "- "
        + PLAN_CSV_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print(
        "- "
        + PLAN_JSON_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print(
        "- "
        + SUMMARY_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )


if __name__ == "__main__":
    main()