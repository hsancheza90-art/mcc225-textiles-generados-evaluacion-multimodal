"""Genera los captions positivos y el manifiesto multimodal v2."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "experimento_v2.json"
)

VOCABULARY_PATH = (
    PROJECT_ROOT
    / "config"
    / "vocabulario_captions_v2.json"
)

PLAN_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "plan_registros_v2.json"
)

IMAGE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "manifest_imagenes_v2.csv"
)

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
)

COLORLESS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_sin_color_v2.csv"
)

MULTIMODAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "manifest_multimodal_v2.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "resumen_captions_positivos_v2.json"
)

STRUCTURE_FIELDS = (
    "motif",
    "orientation",
    "composition",
    "symmetry",
)

CAPTION_FIELDS = (
    "caption_id",
    "image_id",
    "semantic_id",
    "structure_id",
    "colorless_caption_id",
    "template_id",
    "is_canonical",
    "split",
    "pattern_id",
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "caption_text",
    "caption_sha256",
)

COLORLESS_FIELDS = (
    "colorless_caption_id",
    "structure_id",
    "template_id",
    "is_canonical",
    "pattern_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "caption_text",
    "caption_sha256",
    "relevant_image_ids",
    "relevant_semantic_ids",
    "relevant_count",
)

MULTIMODAL_FIELDS = (
    "image_id",
    "semantic_id",
    "structure_id",
    "pattern_id",
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "ambiguity_level",
    "split",
    "image_path",
    "file_sha256",
    "pixel_sha256",
    "canonical_caption_id",
    "canonical_caption_text",
    "positive_caption_ids",
    "positive_caption_count",
    "colorless_caption_ids",
    "colorless_caption_count",
)


def load_json(path: Path) -> dict[str, Any]:
    """Carga un archivo JSON UTF-8."""

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    """Carga un CSV UTF-8."""

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def write_csv(
    path: Path,
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    """Escribe CSV UTF-8 con finales de línea LF."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Escribe JSON UTF-8 sin BOM y con finales LF."""

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
        newline="\n",
    )


def sha256_file(path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(65536),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Calcula SHA-256 de una cadena UTF-8."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_text(text: str) -> str:
    """Normaliza texto para controles de unicidad."""

    text = unicodedata.normalize(
        "NFKC",
        text,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_structure_signature(
    record: dict[str, Any],
) -> str:
    """Construye la firma de los atributos no cromáticos."""

    return "|".join(
        f"{field}={record[field]}"
        for field in STRUCTURE_FIELDS
    )


def build_structure_id(signature: str) -> str:
    """Construye un identificador estructural estable."""

    digest = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:16].upper()

    return f"STR-{digest}"


def build_caption_id(
    image_id: str,
    template_id: str,
) -> str:
    """Construye el ID estable de un caption completo."""

    return f"CAP-{image_id}-{template_id}"


def build_colorless_caption_id(
    structure_id: str,
    template_id: str,
) -> str:
    """Construye el ID estable de un caption sin color."""

    digest = structure_id.removeprefix("STR-")

    return f"SCAP-{digest}-{template_id}"


def main() -> None:
    """Genera y valida los tres artefactos tabulares."""

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )
    vocabulary = load_json(
        VOCABULARY_PATH
    )
    plan = load_json(
        PLAN_PATH
    )

    image_manifest = load_csv(
        IMAGE_MANIFEST_PATH
    )

    records = plan["records"]

    if len(records) != 56:
        raise AssertionError(
            f"Se esperaban 56 registros y se "
            f"encontraron {len(records)}."
        )

    if len(image_manifest) != 56:
        raise AssertionError(
            "El manifiesto de imágenes no contiene "
            "56 registros."
        )

    images_by_id = {
        record["image_id"]: record
        for record in image_manifest
    }

    if len(images_by_id) != 56:
        raise AssertionError(
            "Los image_id del manifiesto no son únicos."
        )

    palette_labels = vocabulary[
        "palette_labels"
    ]
    attribute_labels = vocabulary[
        "attribute_labels"
    ]

    full_templates = {
        item["template_id"]: item["text"]
        for item in vocabulary["full_templates"]
    }

    colorless_templates = {
        item["template_id"]: item["text"]
        for item in vocabulary[
            "colorless_templates"
        ]
    }

    template_ids = sorted(full_templates)

    if template_ids != [
        "TPL_01",
        "TPL_02",
        "TPL_03",
        "TPL_04",
        "TPL_05",
    ]:
        raise AssertionError(
            "Los template_id no coinciden con "
            "TPL_01–TPL_05."
        )

    if set(full_templates) != set(
        colorless_templates
    ):
        raise AssertionError(
            "Las plantillas completas y sin color "
            "no tienen los mismos identificadores."
        )

    canonical_template_id = vocabulary[
        "canonical_template_id"
    ]

    caption_records: list[dict[str, Any]] = []

    colorless_candidates: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    relevant_images: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    relevant_semantics: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)

    captions_by_image: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    structures_by_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        image_id = record["image_id"]

        if image_id not in images_by_id:
            raise AssertionError(
                f"No existe información visual para "
                f"{image_id}."
            )

        structure_signature = (
            build_structure_signature(record)
        )

        structure_id = build_structure_id(
            structure_signature
        )

        structure_values = {
            field: record[field]
            for field in STRUCTURE_FIELDS
        }

        previous_structure = structures_by_id.get(
            structure_id
        )

        if (
            previous_structure is not None
            and previous_structure != structure_values
        ):
            raise AssertionError(
                f"Colisión de structure_id: "
                f"{structure_id}."
            )

        structures_by_id[
            structure_id
        ] = structure_values

        values = {
            "palette": palette_labels[
                record["palette_id"]
            ],
            "motif": attribute_labels["motif"][
                record["motif"]
            ],
            "orientation": (
                attribute_labels["orientation"][
                    record["orientation"]
                ]
            ),
            "composition": (
                attribute_labels["composition"][
                    record["composition"]
                ]
            ),
            "symmetry": (
                attribute_labels["symmetry"][
                    record["symmetry"]
                ]
            ),
        }

        colorless_values = {
            key: value
            for key, value in values.items()
            if key != "palette"
        }

        for template_id in template_ids:
            caption_text = normalize_text(
                full_templates[
                    template_id
                ].format(**values)
            )

            colorless_text = normalize_text(
                colorless_templates[
                    template_id
                ].format(**colorless_values)
            )

            caption_id = build_caption_id(
                image_id,
                template_id,
            )

            colorless_caption_id = (
                build_colorless_caption_id(
                    structure_id,
                    template_id,
                )
            )

            caption_record = {
                "caption_id": caption_id,
                "image_id": image_id,
                "semantic_id": record[
                    "semantic_id"
                ],
                "structure_id": structure_id,
                "colorless_caption_id": (
                    colorless_caption_id
                ),
                "template_id": template_id,
                "is_canonical": (
                    "true"
                    if template_id
                    == canonical_template_id
                    else "false"
                ),
                "split": record["split"],
                "pattern_id": record["pattern_id"],
                "palette_id": record["palette_id"],
                "motif": record["motif"],
                "orientation": record["orientation"],
                "composition": record["composition"],
                "symmetry": record["symmetry"],
                "caption_text": caption_text,
                "caption_sha256": sha256_text(
                    caption_text
                ),
            }

            caption_records.append(
                caption_record
            )

            captions_by_image[
                image_id
            ].append(caption_record)

            key = (
                structure_id,
                template_id,
            )

            candidate = {
                "colorless_caption_id": (
                    colorless_caption_id
                ),
                "structure_id": structure_id,
                "template_id": template_id,
                "is_canonical": (
                    "true"
                    if template_id
                    == canonical_template_id
                    else "false"
                ),
                "pattern_id": record[
                    "pattern_id"
                ],
                "motif": record["motif"],
                "orientation": record[
                    "orientation"
                ],
                "composition": record[
                    "composition"
                ],
                "symmetry": record["symmetry"],
                "caption_text": colorless_text,
                "caption_sha256": sha256_text(
                    colorless_text
                ),
            }

            existing = colorless_candidates.get(
                key
            )

            if (
                existing is not None
                and existing != candidate
            ):
                raise AssertionError(
                    "Una estructura produjo dos captions "
                    f"sin color diferentes: {key}."
                )

            colorless_candidates[
                key
            ] = candidate

            relevant_images[key].add(
                image_id
            )

            relevant_semantics[key].add(
                record["semantic_id"]
            )

    if len(caption_records) != 280:
        raise AssertionError(
            "No se generaron 280 captions completos."
        )

    caption_ids = [
        record["caption_id"]
        for record in caption_records
    ]

    caption_texts = [
        record["caption_text"]
        for record in caption_records
    ]

    if len(set(caption_ids)) != 280:
        raise AssertionError(
            "Los caption_id no son únicos."
        )

    if len(set(caption_texts)) != 280:
        raise AssertionError(
            "Los captions completos no son únicos."
        )

    pattern_order = {
        pattern["pattern_id"]: index
        for index, pattern in enumerate(
            [
                *experiment["dataset_v2"][
                    "base_patterns"
                ],
                *experiment["dataset_v2"][
                    "heldout_patterns"
                ],
            ]
        )
    }

    colorless_records: list[
        dict[str, Any]
    ] = []

    sorted_colorless_keys = sorted(
        colorless_candidates,
        key=lambda key: (
            pattern_order[
                colorless_candidates[key][
                    "pattern_id"
                ]
            ],
            key[1],
        ),
    )

    for key in sorted_colorless_keys:
        candidate = dict(
            colorless_candidates[key]
        )

        image_ids = sorted(
            relevant_images[key]
        )

        semantic_ids = sorted(
            relevant_semantics[key]
        )

        if len(image_ids) != 7:
            raise AssertionError(
                f"{key}: se esperaban siete imágenes "
                f"relevantes y se encontraron "
                f"{len(image_ids)}."
            )

        if len(semantic_ids) != 7:
            raise AssertionError(
                f"{key}: se esperaban siete firmas "
                "semánticas relevantes."
            )

        candidate.update(
            {
                "relevant_image_ids": "|".join(
                    image_ids
                ),
                "relevant_semantic_ids": "|".join(
                    semantic_ids
                ),
                "relevant_count": 7,
            }
        )

        colorless_records.append(
            candidate
        )

    if len(colorless_records) != 40:
        raise AssertionError(
            "No se generaron 40 captions sin color."
        )

    colorless_ids = [
        record["colorless_caption_id"]
        for record in colorless_records
    ]

    colorless_texts = [
        record["caption_text"]
        for record in colorless_records
    ]

    if len(set(colorless_ids)) != 40:
        raise AssertionError(
            "Los colorless_caption_id no son únicos."
        )

    if len(set(colorless_texts)) != 40:
        raise AssertionError(
            "Los captions sin color no son únicos."
        )

    multimodal_records: list[
        dict[str, Any]
    ] = []

    for record in records:
        image_id = record["image_id"]
        image_record = images_by_id[image_id]

        image_captions = sorted(
            captions_by_image[image_id],
            key=lambda item: item["template_id"],
        )

        if len(image_captions) != 5:
            raise AssertionError(
                f"{image_id}: no tiene cinco captions."
            )

        canonical_matches = [
            item
            for item in image_captions
            if item["is_canonical"] == "true"
        ]

        if len(canonical_matches) != 1:
            raise AssertionError(
                f"{image_id}: caption canónico "
                "inválido."
            )

        canonical = canonical_matches[0]
        structure_id = canonical["structure_id"]

        positive_ids = [
            item["caption_id"]
            for item in image_captions
        ]

        structural_ids = [
            build_colorless_caption_id(
                structure_id,
                template_id,
            )
            for template_id in template_ids
        ]

        multimodal_records.append(
            {
                "image_id": image_id,
                "semantic_id": record[
                    "semantic_id"
                ],
                "structure_id": structure_id,
                "pattern_id": record[
                    "pattern_id"
                ],
                "palette_id": record[
                    "palette_id"
                ],
                "motif": record["motif"],
                "orientation": record[
                    "orientation"
                ],
                "composition": record[
                    "composition"
                ],
                "symmetry": record["symmetry"],
                "ambiguity_level": record[
                    "ambiguity_level"
                ],
                "split": record["split"],
                "image_path": record[
                    "image_path"
                ],
                "file_sha256": image_record[
                    "file_sha256"
                ],
                "pixel_sha256": image_record[
                    "pixel_sha256"
                ],
                "canonical_caption_id": (
                    canonical["caption_id"]
                ),
                "canonical_caption_text": (
                    canonical["caption_text"]
                ),
                "positive_caption_ids": "|".join(
                    positive_ids
                ),
                "positive_caption_count": 5,
                "colorless_caption_ids": "|".join(
                    structural_ids
                ),
                "colorless_caption_count": 5,
            }
        )

    if len(multimodal_records) != 56:
        raise AssertionError(
            "El manifiesto multimodal no contiene "
            "56 registros."
        )

    write_csv(
        CAPTIONS_PATH,
        caption_records,
        CAPTION_FIELDS,
    )

    write_csv(
        COLORLESS_PATH,
        colorless_records,
        COLORLESS_FIELDS,
    )

    write_csv(
        MULTIMODAL_PATH,
        multimodal_records,
        MULTIMODAL_FIELDS,
    )

    captions_by_template = Counter(
        record["template_id"]
        for record in caption_records
    )

    captions_by_split = Counter(
        record["split"]
        for record in caption_records
    )

    captions_by_semantic = Counter(
        record["semantic_id"]
        for record in caption_records
    )

    structures_by_pattern = Counter(
        record["pattern_id"]
        for record in colorless_records
    )

    inventory_material = "\n".join(
        (
            f"{record['caption_id']}:"
            f"{record['caption_sha256']}"
        )
        for record in caption_records
    )

    summary = {
        "schema_version": "1.0",
        "dataset_version": "v2",
        "stage": "positive_caption_generation",
        "config_sha256": sha256_file(
            EXPERIMENT_CONFIG_PATH
        ),
        "vocabulary_sha256": sha256_file(
            VOCABULARY_PATH
        ),
        "plan_sha256": sha256_file(
            PLAN_PATH
        ),
        "image_manifest_sha256": sha256_file(
            IMAGE_MANIFEST_PATH
        ),
        "image_count": len(
            multimodal_records
        ),
        "positive_caption_count": len(
            caption_records
        ),
        "unique_positive_caption_count": len(
            set(caption_texts)
        ),
        "colorless_caption_count": len(
            colorless_records
        ),
        "unique_colorless_caption_count": len(
            set(colorless_texts)
        ),
        "structure_count": len(
            structures_by_id
        ),
        "counts_by_template": dict(
            sorted(captions_by_template.items())
        ),
        "counts_by_split": dict(
            captions_by_split
        ),
        "captions_per_semantic_id": sorted(
            set(captions_by_semantic.values())
        ),
        "colorless_per_pattern": dict(
            sorted(structures_by_pattern.items())
        ),
        "canonical_template_id": (
            canonical_template_id
        ),
        "captions_path": (
            CAPTIONS_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "colorless_path": (
            COLORLESS_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "multimodal_manifest_path": (
            MULTIMODAL_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "captions_sha256": sha256_file(
            CAPTIONS_PATH
        ),
        "colorless_sha256": sha256_file(
            COLORLESS_PATH
        ),
        "multimodal_manifest_sha256": (
            sha256_file(MULTIMODAL_PATH)
        ),
        "caption_inventory_sha256": (
            sha256_text(inventory_material)
        ),
        "generation_valid": True,
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print("=" * 76)
    print("CAPTIONS POSITIVOS V2 GENERADOS")
    print("=" * 76)
    print(
        "Imágenes multimodales:     "
        f"{summary['image_count']}"
    )
    print(
        "Captions completos:        "
        f"{summary['positive_caption_count']}"
    )
    print(
        "Captions completos únicos: "
        f"{summary['unique_positive_caption_count']}"
    )
    print(
        "Captions sin color:        "
        f"{summary['colorless_caption_count']}"
    )
    print(
        "Captions sin color únicos: "
        f"{summary['unique_colorless_caption_count']}"
    )
    print(
        "Estructuras:               "
        f"{summary['structure_count']}"
    )
    print(
        "Captions por semantic_id:  "
        f"{summary['captions_per_semantic_id']}"
    )
    print(
        "Usos por plantilla:        "
        f"{summary['counts_by_template']}"
    )
    print(
        "Generación válida:         "
        f"{summary['generation_valid']}"
    )

    print("\nArtefactos:")
    print(
        "- "
        + CAPTIONS_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print(
        "- "
        + COLORLESS_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print(
        "- "
        + MULTIMODAL_PATH
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