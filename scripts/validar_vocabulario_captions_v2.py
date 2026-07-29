"""Valida el vocabulario y las plantillas de captions del dataset v2."""

from __future__ import annotations

import hashlib
import json
import re
import string
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

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "diseno_captions_positivos_v2.md"
)

STRUCTURE_FIELDS = (
    "motif",
    "orientation",
    "composition",
    "symmetry",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def placeholder_names(template: str) -> set[str]:
    formatter = string.Formatter()

    return {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name is not None
    }


def structure_signature(
    record: dict[str, Any],
) -> str:
    return "|".join(
        f"{field}={record[field]}"
        for field in STRUCTURE_FIELDS
    )


def structure_id(signature: str) -> str:
    digest = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()[:16].upper()

    return f"STR-{digest}"


def word_count(text: str) -> int:
    return len(
        re.findall(
            r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b",
            text,
            flags=re.UNICODE,
        )
    )



def validate_typography(
    caption: str,
    context: str,
) -> None:
    """Comprueba reglas editoriales b?sicas del caption."""

    if caption != caption.strip():
        raise AssertionError(
            f"{context}: contiene espacios al inicio "
            "o al final."
        )

    if "  " in caption:
        raise AssertionError(
            f"{context}: contiene espacios dobles. "
            f"Caption: {caption}"
        )

    if re.search(r";(?=\S)", caption):
        raise AssertionError(
            f"{context}: falta un espacio despu?s de "
            f"punto y coma. Caption: {caption}"
        )

    if re.search(r",(?=\S)", caption):
        raise AssertionError(
            f"{context}: falta un espacio despu?s de "
            f"coma. Caption: {caption}"
        )

    if re.search(r"\s+[;,.]", caption):
        raise AssertionError(
            f"{context}: contiene un espacio antes de "
            f"un signo de puntuaci?n. Caption: {caption}"
        )

    normalized = normalize_text(caption)

    if "motivo motivos" in normalized:
        raise AssertionError(
            f"{context}: contiene la secuencia "
            f"'motivo motivos'. Caption: {caption}"
        )

def main() -> None:
    experiment = load_json(EXPERIMENT_CONFIG_PATH)
    vocabulary = load_json(VOCABULARY_PATH)
    plan = load_json(PLAN_PATH)

    records = plan["records"]

    assert len(records) == 56

    dataset = experiment["dataset_v2"]

    expected_palette_ids = {
        *dataset["base_palettes"].keys(),
        *dataset["heldout_palettes"].keys(),
    }

    palette_labels = vocabulary["palette_labels"]

    assert set(palette_labels) == expected_palette_ids
    assert len(set(palette_labels.values())) == 7

    expected_attribute_values: dict[str, set[str]] = {
        "motif": {
            record["motif"]
            for record in records
        },
        "orientation": {
            record["orientation"]
            for record in records
        },
        "composition": {
            record["composition"]
            for record in records
        },
        "symmetry": {
            record["symmetry"]
            for record in records
        },
    }

    attribute_labels = vocabulary["attribute_labels"]

    for field, expected_values in (
        expected_attribute_values.items()
    ):
        configured_values = set(
            attribute_labels[field]
        )

        assert configured_values == expected_values, (
            f"{field}: valores configurados distintos "
            "de los presentes en el plan."
        )

        labels = list(
            attribute_labels[field].values()
        )

        assert all(label.strip() for label in labels)

    full_templates = vocabulary["full_templates"]
    colorless_templates = vocabulary[
        "colorless_templates"
    ]

    assert len(full_templates) == 5
    assert len(colorless_templates) == 5

    expected_template_ids = {
        f"TPL_{index:02d}"
        for index in range(1, 6)
    }

    full_template_ids = {
        template["template_id"]
        for template in full_templates
    }

    colorless_template_ids = {
        template["template_id"]
        for template in colorless_templates
    }

    assert full_template_ids == expected_template_ids
    assert colorless_template_ids == expected_template_ids

    assert vocabulary["canonical_template_id"] == "TPL_01"

    validation = vocabulary["validation"]

    required_full = set(
        validation["required_full_placeholders"]
    )

    required_colorless = set(
        validation["required_colorless_placeholders"]
    )

    assert required_full == {
        "palette",
        "motif",
        "orientation",
        "composition",
        "symmetry",
    }

    assert required_colorless == {
        "motif",
        "orientation",
        "composition",
        "symmetry",
    }

    for template in full_templates:
        actual = placeholder_names(template["text"])

        assert actual == required_full, (
            f"{template['template_id']}: placeholders "
            f"completos inesperados: {actual}."
        )

    for template in colorless_templates:
        actual = placeholder_names(template["text"])

        assert actual == required_colorless, (
            f"{template['template_id']}: placeholders "
            f"sin color inesperados: {actual}."
        )

    full_by_id = {
        template["template_id"]: template["text"]
        for template in full_templates
    }

    colorless_by_id = {
        template["template_id"]: template["text"]
        for template in colorless_templates
    }

    minimum_words = int(validation["minimum_words"])
    maximum_words = int(validation["maximum_words"])

    forbidden_terms = [
        normalize_text(term)
        for term in validation["forbidden_terms"]
    ]

    rendered_full: list[str] = []
    rendered_colorless: list[str] = []

    captions_by_semantic_id: Counter[str] = Counter()
    captions_by_template_id: Counter[str] = Counter()

    structures: dict[str, set[str]] = defaultdict(set)

    full_word_counts: list[int] = []
    colorless_word_counts: list[int] = []

    for record in records:
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
            "symmetry": attribute_labels["symmetry"][
                record["symmetry"]
            ],
        }

        signature = structure_signature(record)
        current_structure_id = structure_id(signature)

        structures[current_structure_id].add(
            record["semantic_id"]
        )

        for template_id in sorted(expected_template_ids):
            full_caption = full_by_id[
                template_id
            ].format(**values)

            colorless_values = {
                key: value
                for key, value in values.items()
                if key != "palette"
            }

            colorless_caption = colorless_by_id[
                template_id
            ].format(**colorless_values)

            normalized_full = normalize_text(
                full_caption
            )
            normalized_colorless = normalize_text(
                colorless_caption
            )

            validate_typography(
                full_caption,
                (
                    f"{record['image_id']} "
                    f"{template_id} completo"
                ),
            )

            validate_typography(
                colorless_caption,
                (
                    f"{record['image_id']} "
                    f"{template_id} sin color"
                ),
            )

            for forbidden_term in forbidden_terms:
                assert forbidden_term not in normalized_full
                assert (
                    forbidden_term
                    not in normalized_colorless
                )

            current_full_word_count = word_count(
                full_caption
            )

            current_colorless_word_count = word_count(
                colorless_caption
            )

            if not (
                minimum_words
                <= current_full_word_count
                <= maximum_words
            ):
                raise AssertionError(
                    f"{record['image_id']} {template_id} "
                    f"completo: {current_full_word_count} "
                    f"palabras; rango permitido "
                    f"{minimum_words}?{maximum_words}. "
                    f"Caption: {full_caption}"
                )

            if not (
                minimum_words
                <= current_colorless_word_count
                <= maximum_words
            ):
                raise AssertionError(
                    f"{record['image_id']} {template_id} "
                    f"sin color: "
                    f"{current_colorless_word_count} "
                    f"palabras; rango permitido "
                    f"{minimum_words}?{maximum_words}. "
                    f"Caption: {colorless_caption}"
                )

            rendered_full.append(full_caption)
            rendered_colorless.append(
                colorless_caption
            )

            full_word_counts.append(
                current_full_word_count
            )

            colorless_word_counts.append(
                current_colorless_word_count
            )

            captions_by_semantic_id[
                record["semantic_id"]
            ] += 1

            captions_by_template_id[
                template_id
            ] += 1

    assert len(rendered_full) == 280
    assert len(set(rendered_full)) == 280

    assert set(captions_by_semantic_id.values()) == {5}
    assert len(captions_by_semantic_id) == 56

    assert captions_by_template_id == Counter(
        {
            template_id: 56
            for template_id in expected_template_ids
        }
    )

    assert len(structures) == 8
    assert set(
        len(semantic_ids)
        for semantic_ids in structures.values()
    ) == {7}

    unique_colorless = set(rendered_colorless)

    assert len(unique_colorless) == 40

    colorless_counts = Counter(rendered_colorless)

    assert set(colorless_counts.values()) == {7}

    document = normalize_text(
        DOCUMENT_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    required_document_concepts = {
        "280 captions": "280" in document,
        "40 captions sin color": (
            "40" in document
            and "captions sin color" in document
        ),
        "cinco atributos": (
            "cinco atributos" in document
        ),
        "atajos léxicos": (
            "atajos lexicos" in document
        ),
        "identidad cultural excluida": (
            "identidad cultural" in document
        ),
        "template canónico": (
            "tpl_01" in document
        ),
    }

    missing_document_concepts = [
        concept
        for concept, present in (
            required_document_concepts.items()
        )
        if not present
    ]

    assert not missing_document_concepts, (
        "Faltan conceptos en el documento: "
        + ", ".join(missing_document_concepts)
    )

    print("=" * 76)
    print("VOCABULARIO DE CAPTIONS V2 VÁLIDO")
    print("=" * 76)
    print("Paletas cubiertas:          7")
    print("Patrones estructurales:     8")
    print("Plantillas completas:       5")
    print("Plantillas sin color:       5")
    print("Captions completos posibles: 280")
    print("Captions completos únicos:  280")
    print("Captions sin color únicos:  40")
    print("Repetición sin color:       7 por patrón-paleta")
    print("Captions por semantic_id:   5")
    print("Usos por plantilla:         56")
    print(
        "Longitud completa:        "
        f"{min(full_word_counts)}–"
        f"{max(full_word_counts)} palabras"
    )
    print(
        "Longitud sin color:        "
        f"{min(colorless_word_counts)}–"
        f"{max(colorless_word_counts)} palabras"
    )
    print("Términos culturales:        excluidos")
    print("Documento metodológico:     consistente")


if __name__ == "__main__":
    main()