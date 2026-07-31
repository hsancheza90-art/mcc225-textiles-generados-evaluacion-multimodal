"""Genera una previsualización de los ocho patrones del dataset v2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcc225_textiles.rendering import (  # noqa: E402
    render_pattern,
    save_contact_sheet,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "experimento_v2.json"

PREVIEW_DIR = PROJECT_ROOT / "data" / "v2" / "previews"

CONTACT_SHEET_PATH = (
    PROJECT_ROOT
    / "figures"
    / "v2"
    / "previsualizacion_patrones_v2.png"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "resumen_previsualizacion_patrones_v2.json"
)

PREVIEW_PALETTE_ID = "blanco_negro_gris"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_pixels(image: Any) -> str:
    return hashlib.sha256(
        image.convert("RGB").tobytes()
    ).hexdigest()


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
    )


def main() -> None:
    config = load_json(CONFIG_PATH)
    dataset = config["dataset_v2"]

    patterns = [
        *dataset["base_patterns"],
        *dataset["heldout_patterns"],
    ]

    pattern_ids = [
        pattern["pattern_id"]
        for pattern in patterns
    ]

    if len(pattern_ids) != 8:
        raise AssertionError(
            f"Se esperaban 8 patrones y se encontraron "
            f"{len(pattern_ids)}."
        )

    if len(set(pattern_ids)) != 8:
        raise AssertionError(
            "Los pattern_id deben ser únicos."
        )

    palette = dataset["base_palettes"][
        PREVIEW_PALETTE_ID
    ]

    width = int(dataset["image_width"])
    height = int(dataset["image_height"])
    seed_base = int(
        config["reproducibility"]["seed"]
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_records: list[dict[str, Any]] = []
    image_paths: list[Path] = []

    for index, pattern in enumerate(
        patterns,
        start=1,
    ):
        pattern_id = pattern["pattern_id"]
        preview_seed = seed_base * 1000 + index

        image = render_pattern(
            pattern_id=pattern_id,
            raw_palette=palette,
            seed=preview_seed,
            width=width,
            height=height,
        )

        image_path = (
            PREVIEW_DIR / f"{pattern_id}.png"
        )

        image.save(
            image_path,
            format="PNG",
        )

        image_paths.append(image_path)

        preview_records.append(
            {
                "pattern_id": pattern_id,
                "motif": pattern["motif"],
                "orientation": pattern["orientation"],
                "composition": pattern["composition"],
                "symmetry": pattern["symmetry"],
                "ambiguity_level": (
                    pattern["ambiguity_level"]
                ),
                "pattern_source": (
                    "base"
                    if pattern in dataset["base_patterns"]
                    else "heldout"
                ),
                "palette_id": PREVIEW_PALETTE_ID,
                "seed": preview_seed,
                "width": width,
                "height": height,
                "image_path": (
                    image_path
                    .relative_to(PROJECT_ROOT)
                    .as_posix()
                ),
                "file_sha256": sha256_file(image_path),
                "pixel_sha256": sha256_pixels(image),
            }
        )

    save_contact_sheet(
        image_paths=image_paths,
        labels=pattern_ids,
        output_path=CONTACT_SHEET_PATH,
        columns=4,
    )

    unique_file_hashes = {
        record["file_sha256"]
        for record in preview_records
    }

    unique_pixel_hashes = {
        record["pixel_sha256"]
        for record in preview_records
    }

    summary = {
        "schema_version": "1.0",
        "stage": "pattern_preview",
        "pattern_count": len(preview_records),
        "base_pattern_count": len(
            dataset["base_patterns"]
        ),
        "heldout_pattern_count": len(
            dataset["heldout_patterns"]
        ),
        "palette_id": PREVIEW_PALETTE_ID,
        "image_width": width,
        "image_height": height,
        "all_file_hashes_unique": (
            len(unique_file_hashes)
            == len(preview_records)
        ),
        "all_pixel_hashes_unique": (
            len(unique_pixel_hashes)
            == len(preview_records)
        ),
        "contact_sheet_path": (
            CONTACT_SHEET_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "contact_sheet_sha256": sha256_file(
            CONTACT_SHEET_PATH
        ),
        "previews": preview_records,
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print("=" * 76)
    print("PREVISUALIZACIÓN DE PATRONES V2 GENERADA")
    print("=" * 76)
    print(f"Patrones:              {len(preview_records)}")
    print(
        "Patrones base:         "
        f"{summary['base_pattern_count']}"
    )
    print(
        "Patrones OOD:          "
        f"{summary['heldout_pattern_count']}"
    )
    print(f"Paleta:                {PREVIEW_PALETTE_ID}")
    print(f"Dimensiones:           {width} x {height}")
    print(
        "Hashes de archivo:     "
        f"{summary['all_file_hashes_unique']}"
    )
    print(
        "Hashes de píxeles:     "
        f"{summary['all_pixel_hashes_unique']}"
    )
    print("\nLámina:")
    print(
        "- "
        + CONTACT_SHEET_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print("\nResumen:")
    print(
        "- "
        + SUMMARY_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )


if __name__ == "__main__":
    main()