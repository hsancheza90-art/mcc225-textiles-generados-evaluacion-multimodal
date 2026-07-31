"""Genera las 56 imágenes finales del dataset v2.

El script consume el plan de registros previamente validado y utiliza
exclusivamente el renderizador congelado en src/mcc225_textiles.
"""

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

IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "images"
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

PLAN_FIELDS = (
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

MANIFEST_FIELDS = (
    *PLAN_FIELDS,
    "file_size_bytes",
    "file_sha256",
    "pixel_sha256",
    "unique_color_count",
)


def load_json(path: Path) -> dict[str, Any]:
    """Carga un archivo JSON UTF-8."""

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def sha256_file(path: Path) -> str:
    """Calcula el SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Calcula SHA-256 para una cadena UTF-8."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_pixels(image: Image.Image) -> str:
    """Calcula SHA-256 sobre los píxeles RGB."""

    return hashlib.sha256(
        image.convert("RGB").tobytes()
    ).hexdigest()


def count_unique_colors(image: Image.Image) -> int:
    """Cuenta colores RGB diferentes."""

    rgb_image = image.convert("RGB")

    colors = rgb_image.getcolors(
        maxcolors=rgb_image.width * rgb_image.height
    )

    if colors is None:
        raise AssertionError(
            "No fue posible contabilizar los colores."
        )

    return len(colors)


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Escribe el manifiesto técnico."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(MANIFEST_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Escribe JSON estándar sin BOM."""

    path.parent.mkdir(parents=True, exist_ok=True)

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


def build_palette_map(
    config: dict[str, Any],
) -> dict[str, list[list[int]]]:
    """Combina paletas ID y OOD."""

    dataset = config["dataset_v2"]

    base_palettes = dataset["base_palettes"]
    heldout_palettes = dataset["heldout_palettes"]

    overlap = set(base_palettes).intersection(
        heldout_palettes
    )

    if overlap:
        raise AssertionError(
            "Las paletas base y OOD se superponen: "
            + ", ".join(sorted(overlap))
        )

    return {
        **base_palettes,
        **heldout_palettes,
    }


def validate_plan_records(
    records: list[dict[str, Any]],
) -> None:
    """Comprueba la estructura mínima del plan."""

    if len(records) != 56:
        raise AssertionError(
            f"Se esperaban 56 registros y se encontraron "
            f"{len(records)}."
        )

    expected_ids = [
        f"V2_{index:03d}"
        for index in range(1, 57)
    ]

    actual_ids = [
        record["image_id"]
        for record in records
    ]

    if actual_ids != expected_ids:
        raise AssertionError(
            "La secuencia de image_id no coincide con "
            "V2_001–V2_056."
        )

    for record in records:
        missing_fields = [
            field
            for field in PLAN_FIELDS
            if field not in record
        ]

        if missing_fields:
            raise AssertionError(
                f"{record.get('image_id', 'sin_id')}: "
                f"faltan campos {missing_fields}."
            )


def main() -> None:
    """Genera imágenes, manifiesto y resumen."""

    config = load_json(CONFIG_PATH)
    plan = load_json(PLAN_PATH)

    records = plan["records"]
    validate_plan_records(records)

    palette_map = build_palette_map(config)

    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_paths: set[Path] = set()
    manifest_records: list[dict[str, Any]] = []

    for record in records:
        image_id = record["image_id"]
        pattern_id = record["pattern_id"]
        palette_id = record["palette_id"]

        if palette_id not in palette_map:
            raise AssertionError(
                f"{image_id}: paleta desconocida "
                f"{palette_id}."
            )

        relative_path = Path(record["image_path"])

        expected_relative_path = Path(
            f"data/v2/images/{image_id}.png"
        )

        if relative_path != expected_relative_path:
            raise AssertionError(
                f"{image_id}: ruta inesperada "
                f"{relative_path.as_posix()}."
            )

        image_path = PROJECT_ROOT / relative_path
        expected_paths.add(image_path.resolve())

        image = render_pattern(
            pattern_id=pattern_id,
            raw_palette=palette_map[palette_id],
            seed=int(record["seed"]),
            width=int(record["image_width"]),
            height=int(record["image_height"]),
        )

        image.save(
            image_path,
            format="PNG",
            optimize=False,
            compress_level=9,
        )

        with Image.open(image_path) as saved_image:
            if saved_image.format != "PNG":
                raise AssertionError(
                    f"{image_id}: el formato no es PNG."
                )

            if saved_image.mode != "RGB":
                raise AssertionError(
                    f"{image_id}: modo inesperado "
                    f"{saved_image.mode}."
                )

            expected_size = (
                int(record["image_width"]),
                int(record["image_height"]),
            )

            if saved_image.size != expected_size:
                raise AssertionError(
                    f"{image_id}: dimensiones inesperadas "
                    f"{saved_image.size}."
                )

            pixel_hash = sha256_pixels(saved_image)
            unique_color_count = count_unique_colors(
                saved_image
            )

        if unique_color_count < 3:
            raise AssertionError(
                f"{image_id}: solo contiene "
                f"{unique_color_count} colores."
            )

        manifest_record = {
            field: record[field]
            for field in PLAN_FIELDS
        }

        manifest_record.update(
            {
                "file_size_bytes": (
                    image_path.stat().st_size
                ),
                "file_sha256": sha256_file(
                    image_path
                ),
                "pixel_sha256": pixel_hash,
                "unique_color_count": (
                    unique_color_count
                ),
            }
        )

        manifest_records.append(manifest_record)

    actual_paths = {
        path.resolve()
        for path in IMAGES_DIR.glob("*.png")
    }

    missing_paths = expected_paths - actual_paths
    unexpected_paths = actual_paths - expected_paths

    if missing_paths:
        raise AssertionError(
            "Faltan imágenes: "
            + ", ".join(
                sorted(path.name for path in missing_paths)
            )
        )

    if unexpected_paths:
        raise AssertionError(
            "Existen imágenes PNG no previstas: "
            + ", ".join(
                sorted(
                    path.name
                    for path in unexpected_paths
                )
            )
        )

    file_hashes = [
        record["file_sha256"]
        for record in manifest_records
    ]

    pixel_hashes = [
        record["pixel_sha256"]
        for record in manifest_records
    ]

    if len(set(file_hashes)) != 56:
        raise AssertionError(
            "Se detectaron archivos PNG duplicados."
        )

    if len(set(pixel_hashes)) != 56:
        raise AssertionError(
            "Se detectaron imágenes con píxeles duplicados."
        )

    write_csv(
        MANIFEST_PATH,
        manifest_records,
    )

    counts_by_split = Counter(
        record["split"]
        for record in manifest_records
    )

    counts_by_pattern = Counter(
        record["pattern_id"]
        for record in manifest_records
    )

    counts_by_palette = Counter(
        record["palette_id"]
        for record in manifest_records
    )

    inventory_material = "\n".join(
        (
            f"{record['image_id']}:"
            f"{record['file_sha256']}:"
            f"{record['pixel_sha256']}"
        )
        for record in manifest_records
    )

    summary = {
        "schema_version": "1.0",
        "dataset_version": "v2",
        "stage": "final_image_generation",
        "config_sha256": sha256_file(
            CONFIG_PATH
        ),
        "plan_sha256": sha256_file(
            PLAN_PATH
        ),
        "image_count": len(manifest_records),
        "counts_by_split": dict(counts_by_split),
        "counts_by_pattern": dict(
            sorted(counts_by_pattern.items())
        ),
        "counts_by_palette": dict(
            sorted(counts_by_palette.items())
        ),
        "all_file_hashes_unique": (
            len(set(file_hashes)) == 56
        ),
        "all_pixel_hashes_unique": (
            len(set(pixel_hashes)) == 56
        ),
        "all_images_have_at_least_three_colors": all(
            int(record["unique_color_count"]) >= 3
            for record in manifest_records
        ),
        "image_width": int(
            config["dataset_v2"]["image_width"]
        ),
        "image_height": int(
            config["dataset_v2"]["image_height"]
        ),
        "images_directory": (
            IMAGES_DIR
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "manifest_path": (
            MANIFEST_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "manifest_sha256": sha256_file(
            MANIFEST_PATH
        ),
        "inventory_sha256": sha256_text(
            inventory_material
        ),
        "generation_valid": True,
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print("=" * 76)
    print("IMÁGENES FINALES DEL DATASET V2 GENERADAS")
    print("=" * 76)
    print(f"Imágenes:               {summary['image_count']}")
    print(
        "Splits:                 "
        f"{summary['counts_by_split']}"
    )
    print(
        "Patrones:               "
        f"{len(summary['counts_by_pattern'])}"
    )
    print(
        "Paletas:                "
        f"{len(summary['counts_by_palette'])}"
    )
    print(
        "Hashes de archivo:      "
        f"{summary['all_file_hashes_unique']}"
    )
    print(
        "Hashes de píxeles:      "
        f"{summary['all_pixel_hashes_unique']}"
    )
    print(
        "Mínimo de tres colores: "
        f"{summary['all_images_have_at_least_three_colors']}"
    )
    print(
        "Dimensiones:            "
        f"{summary['image_width']} x "
        f"{summary['image_height']}"
    )
    print(
        "Inventario SHA-256:     "
        f"{summary['inventory_sha256']}"
    )
    print(f"Generación válida:       {summary['generation_valid']}")

    print("\nArtefactos:")
    print(
        "- "
        + IMAGES_DIR
        .relative_to(PROJECT_ROOT)
        .as_posix()
        + "/"
    )
    print(
        "- "
        + MANIFEST_PATH
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
