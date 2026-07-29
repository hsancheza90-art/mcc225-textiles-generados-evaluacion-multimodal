"""Valida la previsualización de los ocho patrones del dataset v2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcc225_textiles.rendering import render_pattern  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "experimento_v2.json"

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "resumen_previsualizacion_patrones_v2.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_pixels(image: Image.Image) -> str:
    return hashlib.sha256(
        image.convert("RGB").tobytes()
    ).hexdigest()


def main() -> None:
    config = load_json(CONFIG_PATH)
    summary = load_json(SUMMARY_PATH)

    dataset = config["dataset_v2"]

    expected_patterns = {
        pattern["pattern_id"]
        for pattern in (
            [
                *dataset["base_patterns"],
                *dataset["heldout_patterns"],
            ]
        )
    }

    assert len(expected_patterns) == 8
    assert summary["pattern_count"] == 8
    assert summary["base_pattern_count"] == 6
    assert summary["heldout_pattern_count"] == 2

    palette_id = summary["palette_id"]
    palette = dataset["base_palettes"][palette_id]

    width = int(dataset["image_width"])
    height = int(dataset["image_height"])

    actual_patterns = {
        record["pattern_id"]
        for record in summary["previews"]
    }

    assert actual_patterns == expected_patterns

    file_hashes: list[str] = []
    pixel_hashes: list[str] = []

    for record in summary["previews"]:
        path = PROJECT_ROOT / record["image_path"]

        assert path.exists(), (
            f"No existe la imagen: {path}"
        )

        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (width, height)

            actual_pixel_hash = sha256_pixels(image)

        actual_file_hash = sha256_file(path)

        assert actual_file_hash == record["file_sha256"]
        assert actual_pixel_hash == record["pixel_sha256"]

        rendered_again = render_pattern(
            pattern_id=record["pattern_id"],
            raw_palette=palette,
            seed=int(record["seed"]),
            width=width,
            height=height,
        )

        rerendered_pixel_hash = sha256_pixels(
            rendered_again
        )

        assert (
            rerendered_pixel_hash
            == record["pixel_sha256"]
        ), (
            f"Renderizado no determinista para "
            f"{record['pattern_id']}."
        )

        file_hashes.append(actual_file_hash)
        pixel_hashes.append(actual_pixel_hash)

    assert len(set(file_hashes)) == 8
    assert len(set(pixel_hashes)) == 8

    assert summary["all_file_hashes_unique"] is True
    assert summary["all_pixel_hashes_unique"] is True

    contact_sheet_path = (
        PROJECT_ROOT / summary["contact_sheet_path"]
    )

    assert contact_sheet_path.exists()
    assert (
        sha256_file(contact_sheet_path)
        == summary["contact_sheet_sha256"]
    )

    with Image.open(contact_sheet_path) as sheet:
        assert sheet.format == "PNG"
        assert sheet.mode == "RGB"
        assert sheet.width == 1024
        assert sheet.height == 584

    print("=" * 76)
    print("VALIDACIÓN DE PREVISUALIZACIÓN SUPERADA")
    print("=" * 76)
    print("Patrones:                 8")
    print("Patrones base:            6")
    print("Patrones OOD:             2")
    print("Imágenes PNG:             8")
    print("Dimensiones individuales: 512 x 512")
    print("Hashes únicos:            8")
    print("Renderizado determinista: confirmado")
    print("Lámina de contacto:       1024 x 584")


if __name__ == "__main__":
    main()