"""Valida los artefactos del plan de registros del dataset v2."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "experimento_v2.json"
PLAN_CSV_PATH = (
    PROJECT_ROOT / "data" / "v2" / "plan_registros_v2.csv"
)
PLAN_JSON_PATH = (
    PROJECT_ROOT / "data" / "v2" / "plan_registros_v2.json"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "resumen_plan_dataset_v2.json"
)

SEMANTIC_FIELDS = (
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
)

INTEGER_FIELDS = (
    "seed",
    "image_width",
    "image_height",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        normalized = dict(row)

        for field in INTEGER_FIELDS:
            normalized[field] = int(normalized[field])

        normalized_rows.append(normalized)

    return normalized_rows


def semantic_signature(
    record: dict[str, Any],
) -> str:
    return "|".join(
        f"{field}={record[field]}"
        for field in SEMANTIC_FIELDS
    )


def semantic_id(signature: str) -> str:
    digest = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:16].upper()

    return f"SEM-{digest}"


def main() -> None:
    config = load_json(CONFIG_PATH)
    plan_json = load_json(PLAN_JSON_PATH)
    summary = load_json(SUMMARY_PATH)
    csv_records = load_csv(PLAN_CSV_PATH)

    json_records = plan_json["records"]

    assert csv_records == json_records, (
        "Los registros del CSV y del JSON no coinciden."
    )

    assert len(csv_records) == 56
    assert plan_json["record_count"] == 56
    assert summary["record_count"] == 56

    assert plan_json["config_sha256"] == sha256_file(
        CONFIG_PATH
    )
    assert summary["config_sha256"] == sha256_file(
        CONFIG_PATH
    )

    assert summary["plan_csv_sha256"] == sha256_file(
        PLAN_CSV_PATH
    )
    assert summary["plan_json_sha256"] == sha256_file(
        PLAN_JSON_PATH
    )

    image_ids = [
        record["image_id"]
        for record in csv_records
    ]
    semantic_ids = [
        record["semantic_id"]
        for record in csv_records
    ]
    signatures = [
        record["semantic_signature"]
        for record in csv_records
    ]
    image_paths = [
        record["image_path"]
        for record in csv_records
    ]
    seeds = [
        record["seed"]
        for record in csv_records
    ]
    pairs = [
        (
            record["pattern_id"],
            record["palette_id"],
        )
        for record in csv_records
    ]

    assert len(set(image_ids)) == 56
    assert len(set(semantic_ids)) == 56
    assert len(set(signatures)) == 56
    assert len(set(image_paths)) == 56
    assert len(set(seeds)) == 56
    assert len(set(pairs)) == 56

    assert image_ids == [
        f"V2_{index:03d}"
        for index in range(1, 57)
    ]

    expected_seed_start = (
        int(config["reproducibility"]["seed"]) + 1
    )

    assert seeds == list(
        range(
            expected_seed_start,
            expected_seed_start + 56,
        )
    )

    base_pattern_ids = {
        pattern["pattern_id"]
        for pattern in config["dataset_v2"][
            "base_patterns"
        ]
    }

    heldout_pattern_ids = {
        pattern["pattern_id"]
        for pattern in config["dataset_v2"][
            "heldout_patterns"
        ]
    }

    base_palette_ids = set(
        config["dataset_v2"]["base_palettes"]
    )

    heldout_palette_ids = set(
        config["dataset_v2"]["heldout_palettes"]
    )

    expected_split_counts = {
        "id": 30,
        "ood_palette": 12,
        "ood_pattern": 10,
        "ood_both": 4,
    }

    split_counts = Counter(
        record["split"]
        for record in csv_records
    )

    assert dict(split_counts) == expected_split_counts

    pattern_counts = Counter(
        record["pattern_id"]
        for record in csv_records
    )

    palette_counts = Counter(
        record["palette_id"]
        for record in csv_records
    )

    assert len(pattern_counts) == 8, (
        f"Se esperaban 8 patrones, pero se encontraron "
        f"{len(pattern_counts)}."
    )
    assert len(palette_counts) == 7, (
        f"Se esperaban 7 paletas, pero se encontraron "
        f"{len(palette_counts)}."
    )

    assert set(pattern_counts.values()) == {7}
    assert set(palette_counts.values()) == {8}

    for record in csv_records:
        rebuilt_signature = semantic_signature(record)
        rebuilt_semantic_id = semantic_id(
            rebuilt_signature
        )

        assert (
            record["semantic_signature"]
            == rebuilt_signature
        )
        assert record["semantic_id"] == rebuilt_semantic_id

        assert record["image_path"] == (
            f"data/v2/images/{record['image_id']}.png"
        )

        pattern_id = record["pattern_id"]
        palette_id = record["palette_id"]
        split_name = record["split"]

        if split_name == "id":
            assert pattern_id in base_pattern_ids
            assert palette_id in base_palette_ids

        elif split_name == "ood_palette":
            assert pattern_id in base_pattern_ids
            assert palette_id in heldout_palette_ids

        elif split_name == "ood_pattern":
            assert pattern_id in heldout_pattern_ids
            assert palette_id in base_palette_ids

        elif split_name == "ood_both":
            assert pattern_id in heldout_pattern_ids
            assert palette_id in heldout_palette_ids

        else:
            raise AssertionError(
                f"Split desconocido: {split_name}"
            )

    assert summary["expected_positive_captions"] == 280
    assert summary["expected_hard_negatives"] == 224

    assert summary["image_ids_unique"] is True
    assert summary["semantic_ids_unique"] is True
    assert (
        summary["semantic_signatures_unique"]
        is True
    )
    assert summary["image_paths_unique"] is True
    assert summary["seeds_unique"] is True
    assert (
        summary["pattern_palette_pairs_unique"]
        is True
    )
    assert summary["patterns_balanced"] is True
    assert summary["palettes_balanced"] is True
    assert summary["plan_valid"] is True

    print("=" * 76)
    print("VALIDACIÓN DEL PLAN V2 SUPERADA")
    print("=" * 76)
    print("Registros CSV/JSON:      56 y coincidentes")
    print("Identificadores únicos:  56")
    print("Firmas semánticas:       56 únicas")
    print("Pares patrón-paleta:     56 únicos")
    print("Patrones:                8, con 7 registros cada uno")
    print("Paletas:                 7, con 8 registros cada una")
    print(
        "Splits:                 "
        f"{dict(split_counts)}"
    )
    print("Semillas:                226 a 281")
    print("Captions previstos:      280")
    print("Negativos previstos:     224")
    print("Plan válido:             True")


if __name__ == "__main__":
    main()