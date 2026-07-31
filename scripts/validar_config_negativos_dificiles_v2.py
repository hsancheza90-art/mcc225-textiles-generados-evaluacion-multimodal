"""Valida el diseño de negativos difíciles del dataset v2."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_PATH = (
    PROJECT_ROOT / "config" / "experimento_v2.json"
)

NEGATIVE_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "negativos_dificiles_v2.json"
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

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
)

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "diseno_negativos_dificiles_v2.md"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_csv(path: Path) -> list[dict[str, str]]:
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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def render_caption(
    attributes: dict[str, str],
    template: str,
    vocabulary: dict[str, Any],
) -> str:
    labels = vocabulary["attribute_labels"]

    values = {
        "palette": vocabulary["palette_labels"][
            attributes["palette_id"]
        ],
        "motif": labels["motif"][
            attributes["motif"]
        ],
        "orientation": labels["orientation"][
            attributes["orientation"]
        ],
        "composition": labels["composition"][
            attributes["composition"]
        ],
        "symmetry": labels["symmetry"][
            attributes["symmetry"]
        ],
    }

    return normalize_text(
        template.format(**values)
    )


def positive_position(
    query_index: int,
    schedule: dict[str, Any],
) -> int:
    numerator = (
        int(schedule["multiplier"])
        * query_index
        + int(schedule["block_increment"])
        * (
            query_index
            // int(schedule["block_divisor"])
        )
        + int(schedule["offset"])
    )

    return (
        numerator % int(schedule["modulus"])
        + int(schedule["output_base"])
    )


def main() -> None:
    experiment = load_json(EXPERIMENT_PATH)
    config = load_json(NEGATIVE_CONFIG_PATH)
    vocabulary = load_json(VOCABULARY_PATH)
    plan = load_json(PLAN_PATH)
    captions = load_csv(CAPTIONS_PATH)

    records = sorted(
        plan["records"],
        key=lambda record: record["image_id"],
    )

    assert len(records) == 56
    assert config["query_count"] == 56
    assert config["candidates_per_query"] == 5
    assert config["positives_per_query"] == 1
    assert config["hard_negatives_per_query"] == 4
    assert config["chance_level"] == 0.2
    assert config["negative_scope"] == "query_local"
    assert config["template_id"] == "TPL_01"

    dataset = experiment["dataset_v2"]
    metrics = experiment["metrics"]

    assert (
        dataset["expected_images"]
        == config["query_count"]
    )
    assert (
        dataset["hard_negatives_per_query"]
        == config["hard_negatives_per_query"]
    )
    assert (
        metrics["hard_negative_chance_level"]
        == config["chance_level"]
    )

    attributes = config["attribute_order"]

    assert attributes == [
        "palette_id",
        "motif",
        "orientation",
        "composition",
        "symmetry",
    ]

    assert (
        dataset["hard_negative_attributes"]
        == attributes
    )

    expected_domains = {
        attribute: {
            record[attribute]
            for record in records
        }
        for attribute in attributes
    }

    configured_domains = config["domains"]

    for attribute in attributes:
        domain = configured_domains[attribute]

        assert len(domain) == len(set(domain)), (
            f"{attribute}: dominio con duplicados."
        )

        assert set(domain) == expected_domains[attribute], (
            f"{attribute}: el dominio no coincide "
            "con el plan v2."
        )

        assert len(domain) >= 2

    palette_domain = [
        *dataset["base_palettes"].keys(),
        *dataset["heldout_palettes"].keys(),
    ]

    assert configured_domains["palette_id"] == (
        palette_domain
    )

    templates = {
        item["template_id"]: item["text"]
        for item in vocabulary["full_templates"]
    }

    template = templates[config["template_id"]]

    canonical_captions = {
        row["image_id"]: row
        for row in captions
        if row["template_id"] == "TPL_01"
        and row["is_canonical"] == "true"
    }

    assert len(canonical_captions) == 56

    global_positive_texts = {
        row["caption_text"]
        for row in canonical_captions.values()
    }

    omission = config["omission_schedule"]
    position_schedule = config[
        "positive_position_schedule"
    ]

    omission_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    position_counts: Counter[int] = Counter()
    omission_position_pairs: Counter[
        tuple[str, int]
    ] = Counter()

    negative_position_pairs: Counter[
        tuple[str, int]
    ] = Counter()

    overlap_counts: Counter[str] = Counter()
    transition_counts: Counter[
        tuple[str, str, str]
    ] = Counter()

    total_candidates = 0
    total_negatives = 0
    query_text_sets: list[set[str]] = []

    for query_index, record in enumerate(records):
        image_id = record["image_id"]

        original = {
            attribute: record[attribute]
            for attribute in attributes
        }

        omitted_attribute = attributes[
            (
                query_index
                + int(omission["offset"])
            )
            % len(attributes)
        ]

        omission_counts[omitted_attribute] += 1

        current_positive_position = positive_position(
            query_index,
            position_schedule,
        )

        assert 1 <= current_positive_position <= 5

        position_counts[
            current_positive_position
        ] += 1

        omission_position_pairs[
            (
                omitted_attribute,
                current_positive_position,
            )
        ] += 1

        positive_text = render_caption(
            original,
            template,
            vocabulary,
        )

        assert (
            positive_text
            == canonical_captions[
                image_id
            ]["caption_text"]
        )

        remaining_attributes = [
            attribute
            for attribute in attributes
            if attribute != omitted_attribute
        ]

        rotation = query_index % 4

        rotated_attributes = (
            remaining_attributes[rotation:]
            + remaining_attributes[:rotation]
        )

        available_positions = [
            position
            for position in range(1, 6)
            if position != current_positive_position
        ]

        candidate_texts = {positive_text}

        assert len(rotated_attributes) == 4
        assert len(available_positions) == 4

        for changed_attribute, candidate_position in zip(
            rotated_attributes,
            available_positions,
            strict=True,
        ):
            domain = configured_domains[
                changed_attribute
            ]

            old_value = original[
                changed_attribute
            ]

            old_index = domain.index(old_value)

            new_value = domain[
                (
                    old_index
                    + int(
                        config[
                            "replacement_rule"
                        ]["offset"]
                    )
                )
                % len(domain)
            ]

            assert new_value != old_value

            counterfactual = dict(original)
            counterfactual[
                changed_attribute
            ] = new_value

            actual_changes = [
                attribute
                for attribute in attributes
                if (
                    counterfactual[attribute]
                    != original[attribute]
                )
            ]

            assert actual_changes == [
                changed_attribute
            ]

            negative_text = render_caption(
                counterfactual,
                template,
                vocabulary,
            )

            assert negative_text != positive_text
            assert negative_text not in candidate_texts

            candidate_texts.add(negative_text)

            change_counts[changed_attribute] += 1
            negative_position_pairs[
                (
                    changed_attribute,
                    candidate_position,
                )
            ] += 1

            transition_counts[
                (
                    changed_attribute,
                    old_value,
                    new_value,
                )
            ] += 1

            if negative_text in global_positive_texts:
                overlap_counts[
                    changed_attribute
                ] += 1

            total_negatives += 1

        assert len(candidate_texts) == 5

        query_text_sets.append(candidate_texts)
        total_candidates += len(candidate_texts)

    expected_omissions = {
        key: int(value)
        for key, value in omission[
            "expected_omission_counts"
        ].items()
    }

    expected_changes = {
        key: int(value)
        for key, value in omission[
            "expected_change_counts"
        ].items()
    }

    expected_positions = {
        int(key): int(value)
        for key, value in position_schedule[
            "expected_position_counts"
        ].items()
    }

    assert dict(omission_counts) == (
        expected_omissions
    )

    assert dict(change_counts) == expected_changes
    assert dict(position_counts) == expected_positions

    assert len(omission_position_pairs) == 25

    assert len(omission_position_pairs) == int(
        position_schedule[
            "expected_omission_position_pairs"
        ]
    )

    assert min(
        omission_position_pairs.values()
    ) >= 2

    assert max(
        omission_position_pairs.values()
    ) <= 3

    assert total_negatives == 224
    assert total_candidates == 280
    assert len(query_text_sets) == 56

    expectations = config[
        "validation_expectations"
    ]

    expected_overlap_counts = Counter(
        {
            attribute: int(count)
            for attribute, count in expectations[
                "expected_global_overlap_by_attribute"
            ].items()
        }
    )

    assert sum(overlap_counts.values()) == int(
        expectations[
            "expected_global_overlap_count"
        ]
    )

    assert overlap_counts == expected_overlap_counts

    for attribute in expectations[
        "expected_attributes_without_global_overlap"
    ]:
        assert overlap_counts[attribute] == 0

    expected_transition_count = sum(
        len(domain)
        for domain in configured_domains.values()
    )

    assert expected_transition_count == 29

    assert len(transition_counts) == int(
        expectations[
            "expected_distinct_cyclic_transitions"
        ]
    )

    if expectations[
        "expected_all_cyclic_transitions_covered"
    ]:
        assert (
            len(transition_counts)
            == expected_transition_count
        )
    assert all(
        len(texts) == 5
        for texts in query_text_sets
    )

    constraints = config["constraints"]

    assert constraints[
        "same_template_within_query"
    ] is True

    assert constraints[
        "exactly_one_changed_attribute_per_negative"
    ] is True

    assert constraints[
        "allow_negative_positive_for_other_query"
    ] is True

    assert constraints[
        "global_retrieval_use_prohibited"
    ] is True

    document = normalize_text(
        DOCUMENT_PATH.read_text(
            encoding="utf-8"
        )
    ).casefold()

    required_concepts = {
        "280 candidatos": (
            "280" in document
        ),
        "azar 0.20": (
            "0.20" in document
        ),
        "un atributo": (
            "un único atributo" in document
        ),
        "contrafactual": (
            "contrafactual" in document
        ),
        "relevancia local": (
            "relevancia local" in document
        ),
        "no identificación cultural": (
            "identificación cultural" in document
        ),
    }

    missing = [
        concept
        for concept, present
        in required_concepts.items()
        if not present
    ]

    assert not missing, (
        "Faltan conceptos documentales: "
        + ", ".join(missing)
    )

    print("=" * 76)
    print("CONFIGURACIÓN DE NEGATIVOS DIFÍCILES V2 VÁLIDA")
    print("=" * 76)
    print("Consultas simuladas:        56")
    print("Candidatos simulados:       280")
    print("Negativos simulados:        224")
    print("Candidatos por consulta:    5")
    print("Cambios por negativo:       1")
    print(
        "Omisiones por atributo:    "
        f"{dict(omission_counts)}"
    )
    print(
        "Cambios por atributo:      "
        f"{dict(change_counts)}"
    )
    print(
        "Posiciones del positivo:   "
        f"{dict(position_counts)}"
    )
    print(
        "Pares omisión–posición:    "
        f"{len(omission_position_pairs)} de 25"
    )
    print(
        "Solapamientos globales:    "
        f"{sum(overlap_counts.values())}"
    )
    print(
        "Solapamiento por atributo: "
        f"{dict(overlap_counts)}"
    )
    print(
        "Transiciones distintas:    "
        f"{len(transition_counts)}"
    )
    print("Relevancia:                 local a consulta")
    print("Plantilla única:            TPL_01")
    print("Documento metodológico:     consistente")


if __name__ == "__main__":
    main()