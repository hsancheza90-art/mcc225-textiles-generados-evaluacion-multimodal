"""Valida las 56 imágenes finales del dataset v2."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcc225_textiles.rendering import render_pattern  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "experimento_v2.json"

PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "plan_registros_v2.json"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "manifest_imagenes_v2.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "resumen_imagenes_v2.json"
)

IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "images"
)

INTEGER_FIELDS = (
    "seed",
    "image_width",
    "image_height",
    "file_size_bytes",
    "unique_color_count",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM UTF-8."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
    )

    raw.decode("utf-8")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    normalized: list[dict[str, Any]] = []

    for row in rows:
        record: dict[str, Any] = dict(row)

        for field in INTEGER_FIELDS:
            record[field] = int(record[field])

        normalized.append(record)

    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_pixels(image: Image.Image) -> str:
    return hashlib.sha256(
        image.convert("RGB").tobytes()
    ).hexdigest()


def count_unique_colors(image: Image.Image) -> int:
    rgb_image = image.convert("RGB")

    colors = rgb_image.getcolors(
        maxcolors=rgb_image.width * rgb_image.height
    )

    if colors is None:
        raise AssertionError(
            "No fue posible contabilizar los colores."
        )

    return len(colors)


def build_palette_map(
    config: dict[str, Any],
) -> dict[str, list[list[int]]]:
    dataset = config["dataset_v2"]

    return {
        **dataset["base_palettes"],
        **dataset["heldout_palettes"],
    }


def main() -> None:
    config = load_json(CONFIG_PATH)
    plan = load_json(PLAN_PATH)
    summary = load_json(SUMMARY_PATH)
    manifest = load_csv(MANIFEST_PATH)

    plan_records = plan["records"]

    assert len(plan_records) == 56
    assert len(manifest) == 56
    assert summary["image_count"] == 56

    expected_ids = [
        f"V2_{index:03d}"
        for index in range(1, 57)
    ]

    manifest_ids = [
        record["image_id"]
        for record in manifest
    ]

    assert manifest_ids == expected_ids

    plan_by_id = {
        record["image_id"]: record
        for record in plan_records
    }

    palette_map = build_palette_map(config)

    expected_paths = {
        (
            PROJECT_ROOT
            / record["image_path"]
        ).resolve()
        for record in plan_records
    }

    actual_paths = {
        path.resolve()
        for path in IMAGES_DIR.glob("*.png")
    }

    assert actual_paths == expected_paths
    assert len(actual_paths) == 56

    file_hashes: list[str] = []
    pixel_hashes: list[str] = []

    for record in manifest:
        image_id = record["image_id"]
        plan_record = plan_by_id[image_id]

        for field in (
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
        ):
            assert record[field] == plan_record[field], (
                f"{image_id}: diferencia en {field}."
            )

        image_path = PROJECT_ROOT / record["image_path"]

        assert image_path.exists()
        assert image_path.stat().st_size == record[
            "file_size_bytes"
        ]

        actual_file_hash = sha256_file(image_path)

        assert actual_file_hash == record["file_sha256"]

        with Image.open(image_path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (
                record["image_width"],
                record["image_height"],
            )

            actual_pixel_hash = sha256_pixels(image)
            actual_color_count = count_unique_colors(
                image
            )

        assert (
            actual_pixel_hash
            == record["pixel_sha256"]
        )
        assert (
            actual_color_count
            == record["unique_color_count"]
        )
        assert actual_color_count >= 3

        rendered_again = render_pattern(
            pattern_id=record["pattern_id"],
            raw_palette=palette_map[
                record["palette_id"]
            ],
            seed=record["seed"],
            width=record["image_width"],
            height=record["image_height"],
        )

        rerendered_hash = sha256_pixels(
            rendered_again
        )

        assert rerendered_hash == actual_pixel_hash, (
            f"{image_id}: el renderizado no es "
            "determinista."
        )

        file_hashes.append(actual_file_hash)
        pixel_hashes.append(actual_pixel_hash)

    assert len(set(file_hashes)) == 56
    assert len(set(pixel_hashes)) == 56

    split_counts = Counter(
        record["split"]
        for record in manifest
    )

    pattern_counts = Counter(
        record["pattern_id"]
        for record in manifest
    )

    palette_counts = Counter(
        record["palette_id"]
        for record in manifest
    )

    assert dict(split_counts) == {
        "id": 30,
        "ood_palette": 12,
        "ood_pattern": 10,
        "ood_both": 4,
    }

    assert len(pattern_counts) == 8
    assert len(palette_counts) == 7
    assert set(pattern_counts.values()) == {7}
    assert set(palette_counts.values()) == {8}

    inventory_material = "\n".join(
        (
            f"{record['image_id']}:"
            f"{record['file_sha256']}:"
            f"{record['pixel_sha256']}"
        )
        for record in manifest
    )

    assert (
        summary["inventory_sha256"]
        == sha256_text(inventory_material)
    )

    assert (
        summary["manifest_sha256"]
        == sha256_file(MANIFEST_PATH)
    )

    assert (
        summary["config_sha256"]
        == sha256_file(CONFIG_PATH)
    )

    assert (
        summary["plan_sha256"]
        == sha256_file(PLAN_PATH)
    )

    assert summary["all_file_hashes_unique"] is True
    assert summary["all_pixel_hashes_unique"] is True
    assert (
        summary[
            "all_images_have_at_least_three_colors"
        ]
        is True
    )
    assert summary["generation_valid"] is True

    print("=" * 76)
    print("VALIDACIÓN DE LAS 56 IMÁGENES V2 SUPERADA")
    print("=" * 76)
    print("Imágenes previstas:       56")
    print("Imágenes encontradas:     56")
    print("Registros del manifiesto: 56")
    print("Hashes de archivo únicos: 56")
    print("Hashes de píxeles únicos: 56")
    print("Renderizado determinista: confirmado")
    print("Patrones balanceados:     8 × 7")
    print("Paletas balanceadas:      7 × 8")
    print(
        "Splits:                  "
        f"{dict(split_counts)}"
    )
    print("Inventario válido:        True")


if __name__ == "__main__":
    main()
