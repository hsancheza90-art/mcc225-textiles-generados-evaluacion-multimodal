"""Valida los captions positivos y el manifiesto multimodal v2."""

from __future__ import annotations

import csv
import hashlib
import json
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(65536),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def structure_signature(
    record: dict[str, Any],
) -> str:
    return "|".join(
        f"{field}={record[field]}"
        for field in STRUCTURE_FIELDS
    )


def structure_id(
    signature: str,
) -> str:
    digest = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:16].upper()

    return f"STR-{digest}"


def caption_id(
    image_id: str,
    template_id: str,
) -> str:
    return f"CAP-{image_id}-{template_id}"


def colorless_id(
    current_structure_id: str,
    template_id: str,
) -> str:
    digest = current_structure_id.removeprefix(
        "STR-"
    )

    return f"SCAP-{digest}-{template_id}"


def assert_utf8_lf(path: Path) -> None:
    """Comprueba UTF-8 sin BOM y finales LF."""

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
    )

    raw.decode("utf-8")


def main() -> None:
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

    captions = load_csv(
        CAPTIONS_PATH
    )

    colorless = load_csv(
        COLORLESS_PATH
    )

    multimodal = load_csv(
        MULTIMODAL_PATH
    )

    summary = load_json(
        SUMMARY_PATH
    )

    for path in (
        CAPTIONS_PATH,
        COLORLESS_PATH,
        MULTIMODAL_PATH,
        SUMMARY_PATH,
    ):
        assert_utf8_lf(path)

    assert len(plan["records"]) == 56
    assert len(image_manifest) == 56
    assert len(captions) == 280
    assert len(colorless) == 40
    assert len(multimodal) == 56

    plan_by_id = {
        record["image_id"]: record
        for record in plan["records"]
    }

    images_by_id = {
        record["image_id"]: record
        for record in image_manifest
    }

    captions_by_id = {
        record["caption_id"]: record
        for record in captions
    }

    colorless_by_id = {
        record["colorless_caption_id"]: record
        for record in colorless
    }

    assert len(plan_by_id) == 56
    assert len(images_by_id) == 56
    assert len(captions_by_id) == 280
    assert len(colorless_by_id) == 40

    palette_labels = vocabulary[
        "palette_labels"
    ]

    labels = vocabulary[
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

    template_ids = sorted(
        full_templates
    )

    canonical_template_id = vocabulary[
        "canonical_template_id"
    ]

    caption_texts: list[str] = []
    colorless_texts: list[str] = []

    captions_by_image: dict[
        str,
        list[str],
    ] = defaultdict(list)

    captions_by_semantic: Counter[str] = Counter()
    captions_by_template: Counter[str] = Counter()

    expected_colorless_relations: dict[
        tuple[str, str],
        dict[str, set[str]],
    ] = defaultdict(
        lambda: {
            "image_ids": set(),
            "semantic_ids": set(),
        }
    )

    for plan_record in plan["records"]:
        image_id = plan_record["image_id"]

        current_structure_id = structure_id(
            structure_signature(plan_record)
        )

        values = {
            "palette": palette_labels[
                plan_record["palette_id"]
            ],
            "motif": labels["motif"][
                plan_record["motif"]
            ],
            "orientation": labels[
                "orientation"
            ][plan_record["orientation"]],
            "composition": labels[
                "composition"
            ][plan_record["composition"]],
            "symmetry": labels["symmetry"][
                plan_record["symmetry"]
            ],
        }

        no_color = {
            key: value
            for key, value in values.items()
            if key != "palette"
        }

        for template_id in template_ids:
            expected_caption_id = caption_id(
                image_id,
                template_id,
            )

            assert (
                expected_caption_id
                in captions_by_id
            )

            record = captions_by_id[
                expected_caption_id
            ]

            expected_text = full_templates[
                template_id
            ].format(**values)

            expected_text = " ".join(
                expected_text.split()
            )

            expected_colorless_text = (
                colorless_templates[
                    template_id
                ].format(**no_color)
            )

            expected_colorless_text = " ".join(
                expected_colorless_text.split()
            )

            expected_colorless_id = colorless_id(
                current_structure_id,
                template_id,
            )

            assert record["image_id"] == image_id
            assert (
                record["semantic_id"]
                == plan_record["semantic_id"]
            )
            assert (
                record["structure_id"]
                == current_structure_id
            )
            assert (
                record["colorless_caption_id"]
                == expected_colorless_id
            )
            assert (
                record["template_id"]
                == template_id
            )
            assert (
                record["caption_text"]
                == expected_text
            )
            assert (
                record["caption_sha256"]
                == sha256_text(expected_text)
            )

            expected_canonical = (
                "true"
                if template_id
                == canonical_template_id
                else "false"
            )

            assert (
                record["is_canonical"]
                == expected_canonical
            )

            for field in (
                "split",
                "pattern_id",
                "palette_id",
                "motif",
                "orientation",
                "composition",
                "symmetry",
            ):
                assert (
                    record[field]
                    == plan_record[field]
                )

            caption_texts.append(
                record["caption_text"]
            )

            captions_by_image[
                image_id
            ].append(record["caption_id"])

            captions_by_semantic[
                record["semantic_id"]
            ] += 1

            captions_by_template[
                template_id
            ] += 1

            relation = expected_colorless_relations[
                (
                    current_structure_id,
                    template_id,
                )
            ]

            relation["image_ids"].add(
                image_id
            )

            relation["semantic_ids"].add(
                plan_record["semantic_id"]
            )

            colorless_record = colorless_by_id[
                expected_colorless_id
            ]

            assert (
                colorless_record["caption_text"]
                == expected_colorless_text
            )

    assert len(caption_texts) == 280
    assert len(set(caption_texts)) == 280
    assert set(captions_by_semantic.values()) == {5}
    assert set(captions_by_template.values()) == {56}

    assert len(expected_colorless_relations) == 40

    for key, relation in (
        expected_colorless_relations.items()
    ):
        current_structure_id, template_id = key

        current_colorless_id = colorless_id(
            current_structure_id,
            template_id,
        )

        record = colorless_by_id[
            current_colorless_id
        ]

        image_ids = sorted(
            relation["image_ids"]
        )

        semantic_ids = sorted(
            relation["semantic_ids"]
        )

        assert len(image_ids) == 7
        assert len(semantic_ids) == 7

        assert (
            record["relevant_image_ids"]
            == "|".join(image_ids)
        )
        assert (
            record["relevant_semantic_ids"]
            == "|".join(semantic_ids)
        )
        assert int(
            record["relevant_count"]
        ) == 7

        assert (
            record["caption_sha256"]
            == sha256_text(
                record["caption_text"]
            )
        )

        expected_canonical = (
            "true"
            if template_id
            == canonical_template_id
            else "false"
        )

        assert (
            record["is_canonical"]
            == expected_canonical
        )

        colorless_texts.append(
            record["caption_text"]
        )

    assert len(colorless_texts) == 40
    assert len(set(colorless_texts)) == 40

    expected_ids = [
        f"V2_{index:03d}"
        for index in range(1, 57)
    ]

    assert [
        record["image_id"]
        for record in multimodal
    ] == expected_ids

    for record in multimodal:
        image_id = record["image_id"]
        plan_record = plan_by_id[image_id]
        image_record = images_by_id[image_id]

        current_structure_id = structure_id(
            structure_signature(plan_record)
        )

        positive_ids = record[
            "positive_caption_ids"
        ].split("|")

        structural_ids = record[
            "colorless_caption_ids"
        ].split("|")

        expected_positive_ids = [
            caption_id(
                image_id,
                template_id,
            )
            for template_id in template_ids
        ]

        expected_structural_ids = [
            colorless_id(
                current_structure_id,
                template_id,
            )
            for template_id in template_ids
        ]

        assert positive_ids == expected_positive_ids
        assert structural_ids == expected_structural_ids
        assert int(
            record["positive_caption_count"]
        ) == 5
        assert int(
            record["colorless_caption_count"]
        ) == 5

        canonical_id = caption_id(
            image_id,
            canonical_template_id,
        )

        assert (
            record["canonical_caption_id"]
            == canonical_id
        )

        assert (
            record["canonical_caption_text"]
            == captions_by_id[
                canonical_id
            ]["caption_text"]
        )

        assert (
            record["file_sha256"]
            == image_record["file_sha256"]
        )

        assert (
            record["pixel_sha256"]
            == image_record["pixel_sha256"]
        )

        for field in (
            "semantic_id",
            "pattern_id",
            "palette_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
            "ambiguity_level",
            "split",
            "image_path",
        ):
            assert (
                record[field]
                == plan_record[field]
            )

        assert (
            record["structure_id"]
            == current_structure_id
        )

    inventory_material = "\n".join(
        (
            f"{record['caption_id']}:"
            f"{record['caption_sha256']}"
        )
        for record in captions
    )

    assert summary["image_count"] == 56
    assert (
        summary["positive_caption_count"]
        == 280
    )
    assert (
        summary["unique_positive_caption_count"]
        == 280
    )
    assert (
        summary["colorless_caption_count"]
        == 40
    )
    assert (
        summary["unique_colorless_caption_count"]
        == 40
    )
    assert summary["structure_count"] == 8

    assert (
        summary["captions_sha256"]
        == sha256_file(CAPTIONS_PATH)
    )
    assert (
        summary["colorless_sha256"]
        == sha256_file(COLORLESS_PATH)
    )
    assert (
        summary["multimodal_manifest_sha256"]
        == sha256_file(MULTIMODAL_PATH)
    )
    assert (
        summary["caption_inventory_sha256"]
        == sha256_text(inventory_material)
    )

    assert (
        summary["vocabulary_sha256"]
        == sha256_file(VOCABULARY_PATH)
    )
    assert (
        summary["config_sha256"]
        == sha256_file(
            EXPERIMENT_CONFIG_PATH
        )
    )
    assert summary["generation_valid"] is True

    print("=" * 76)
    print("VALIDACIÓN DE CAPTIONS V2 SUPERADA")
    print("=" * 76)
    print("Imágenes multimodales:      56")
    print("Captions completos:         280")
    print("Captions completos únicos:  280")
    print("Captions sin color:         40")
    print("Captions sin color únicos:  40")
    print("Estructuras:                8")
    print("Captions por imagen:        5")
    print("Captions por plantilla:     56")
    print("Relevantes por estructura:  7")
    print("Reconstrucción exacta:      confirmada")
    print("UTF-8 sin BOM y LF:         confirmado")
    print("Manifiesto multimodal:      válido")


if __name__ == "__main__":
    main()