"""Genera las consultas y candidatos de negativos difíciles v2."""

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

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "negativos_dificiles_v2.json"
)

VOCABULARY_PATH = (
    PROJECT_ROOT
    / "config"
    / "vocabulario_captions_v2.json"
)

MULTIMODAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "manifest_multimodal_v2.csv"
)

POSITIVE_CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
)

QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "consultas_negativos_dificiles_v2.csv"
)

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "candidatos_negativos_dificiles_v2.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "resumen_negativos_dificiles_v2.json"
)

ATTRIBUTES = (
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
)

QUERY_FIELDS = (
    "query_id",
    "query_index",
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
    "template_id",
    "omitted_attribute",
    "positive_position",
    "positive_candidate_id",
    "positive_caption_id",
    "positive_caption_text",
    "candidate_ids",
    "candidate_count",
    "negative_count",
    "global_overlap_negative_count",
    "nonoverlap_negative_count",
)

CANDIDATE_FIELDS = (
    "query_id",
    "query_index",
    "candidate_id",
    "candidate_position",
    "image_id",
    "semantic_id",
    "split",
    "pattern_id",
    "ambiguity_level",
    "template_id",
    "candidate_role",
    "is_positive",
    "relevance_label",
    "changed_attribute",
    "omitted_attribute",
    "old_value",
    "new_value",
    "hamming_distance",
    "negative_type",
    "is_global_positive_text",
    "is_negative_global_overlap",
    "matching_positive_image_ids",
    "matching_positive_semantic_ids",
    "matching_positive_count",
    "original_palette_id",
    "original_motif",
    "original_orientation",
    "original_composition",
    "original_symmetry",
    "candidate_palette_id",
    "candidate_motif",
    "candidate_orientation",
    "candidate_composition",
    "candidate_symmetry",
    "candidate_text",
    "candidate_sha256",
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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
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
        writer.writerows(rows)


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
        newline="\n",
    )


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
    value = (
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
        value % int(schedule["modulus"])
        + int(schedule["output_base"])
    )


def cyclic_successor(
    current_value: str,
    domain: list[str],
    offset: int,
) -> str:
    current_index = domain.index(current_value)

    return domain[
        (current_index + offset) % len(domain)
    ]


def boolean_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    config = load_json(CONFIG_PATH)
    vocabulary = load_json(VOCABULARY_PATH)

    multimodal = sorted(
        load_csv(MULTIMODAL_PATH),
        key=lambda row: row["image_id"],
    )

    positive_captions = load_csv(
        POSITIVE_CAPTIONS_PATH
    )

    if len(multimodal) != 56:
        raise AssertionError(
            "El manifiesto multimodal no tiene "
            "56 registros."
        )

    template_id = config["template_id"]

    templates = {
        item["template_id"]: item["text"]
        for item in vocabulary["full_templates"]
    }

    template = templates[template_id]

    canonical_by_image = {
        row["image_id"]: row
        for row in positive_captions
        if (
            row["template_id"] == template_id
            and row["is_canonical"] == "true"
        )
    }

    if len(canonical_by_image) != 56:
        raise AssertionError(
            "No existen 56 captions canónicos."
        )

    positive_images_by_text: dict[
        str,
        list[str],
    ] = defaultdict(list)

    positive_semantics_by_text: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for row in canonical_by_image.values():
        positive_images_by_text[
            row["caption_text"]
        ].append(row["image_id"])

        positive_semantics_by_text[
            row["caption_text"]
        ].append(row["semantic_id"])

    attributes = list(config["attribute_order"])

    if attributes != list(ATTRIBUTES):
        raise AssertionError(
            "El orden de atributos es inesperado."
        )

    domains = config["domains"]
    omission_schedule = config[
        "omission_schedule"
    ]
    position_schedule = config[
        "positive_position_schedule"
    ]

    replacement_offset = int(
        config["replacement_rule"]["offset"]
    )

    query_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    omission_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    position_counts: Counter[int] = Counter()
    overlap_counts: Counter[str] = Counter()
    negative_type_counts: Counter[str] = Counter()
    split_candidate_counts: Counter[str] = Counter()

    transition_counts: Counter[
        tuple[str, str, str]
    ] = Counter()

    omission_position_pairs: Counter[
        tuple[str, int]
    ] = Counter()

    for query_index, record in enumerate(multimodal):
        image_id = record["image_id"]
        query_id = f"HNQ-{image_id}"

        original = {
            attribute: record[attribute]
            for attribute in attributes
        }

        omitted_attribute = attributes[
            (
                query_index
                + int(omission_schedule["offset"])
            )
            % len(attributes)
        ]

        current_positive_position = positive_position(
            query_index,
            position_schedule,
        )

        omission_counts[omitted_attribute] += 1
        position_counts[current_positive_position] += 1
        omission_position_pairs[
            (
                omitted_attribute,
                current_positive_position,
            )
        ] += 1

        remaining_attributes = [
            attribute
            for attribute in attributes
            if attribute != omitted_attribute
        ]

        rotation = (
            query_index
            % len(remaining_attributes)
        )

        rotated_attributes = (
            remaining_attributes[rotation:]
            + remaining_attributes[:rotation]
        )

        available_positions = [
            position
            for position in range(1, 6)
            if position != current_positive_position
        ]

        changed_attribute_by_position = dict(
            zip(
                available_positions,
                rotated_attributes,
                strict=True,
            )
        )

        positive_caption = canonical_by_image[
            image_id
        ]

        rendered_positive = render_caption(
            original,
            template,
            vocabulary,
        )

        if (
            rendered_positive
            != positive_caption["caption_text"]
        ):
            raise AssertionError(
                f"{image_id}: el caption canónico "
                "no coincide con la reconstrucción."
            )

        query_candidate_rows: list[
            dict[str, Any]
        ] = []

        query_candidate_texts: set[str] = set()
        query_overlap_count = 0

        for candidate_position in range(1, 6):
            is_positive = (
                candidate_position
                == current_positive_position
            )

            candidate_attributes = dict(original)

            changed_attribute = ""
            old_value = ""
            new_value = ""

            if is_positive:
                candidate_role = "positive"
                relevance_label = 1
                hamming_distance = 0
                negative_type = "not_applicable"
            else:
                candidate_role = "hard_negative"
                relevance_label = 0
                hamming_distance = 1

                changed_attribute = (
                    changed_attribute_by_position[
                        candidate_position
                    ]
                )

                old_value = original[
                    changed_attribute
                ]

                new_value = cyclic_successor(
                    old_value,
                    domains[changed_attribute],
                    replacement_offset,
                )

                if new_value == old_value:
                    raise AssertionError(
                        "El valor sustituto no cambió."
                    )

                candidate_attributes[
                    changed_attribute
                ] = new_value

            actual_changes = [
                attribute
                for attribute in attributes
                if (
                    candidate_attributes[attribute]
                    != original[attribute]
                )
            ]

            if is_positive:
                if actual_changes:
                    raise AssertionError(
                        f"{image_id}: el positivo cambió "
                        "atributos."
                    )
            elif actual_changes != [changed_attribute]:
                raise AssertionError(
                    f"{image_id}: el negativo no cambia "
                    "exactamente un atributo."
                )

            candidate_text = render_caption(
                candidate_attributes,
                template,
                vocabulary,
            )

            if candidate_text in query_candidate_texts:
                raise AssertionError(
                    f"{image_id}: candidatos textuales "
                    "duplicados."
                )

            query_candidate_texts.add(candidate_text)

            matching_images = sorted(
                positive_images_by_text.get(
                    candidate_text,
                    [],
                )
            )

            matching_semantics = sorted(
                positive_semantics_by_text.get(
                    candidate_text,
                    [],
                )
            )

            is_global_positive_text = bool(
                matching_images
            )

            is_negative_global_overlap = (
                not is_positive
                and is_global_positive_text
            )

            if is_positive:
                if matching_images != [image_id]:
                    raise AssertionError(
                        f"{image_id}: correspondencia "
                        "global positiva inesperada."
                    )

                negative_type = "not_applicable"
            elif is_negative_global_overlap:
                negative_type = "global_overlap"
                query_overlap_count += 1
                overlap_counts[
                    changed_attribute
                ] += 1
            else:
                negative_type = "counterfactual_only"

            if not is_positive:
                change_counts[
                    changed_attribute
                ] += 1

                transition_counts[
                    (
                        changed_attribute,
                        old_value,
                        new_value,
                    )
                ] += 1

                negative_type_counts[
                    negative_type
                ] += 1

            candidate_id = (
                f"HNC-{image_id}-"
                f"P{candidate_position:02d}"
            )

            candidate_row = {
                "query_id": query_id,
                "query_index": query_index,
                "candidate_id": candidate_id,
                "candidate_position": (
                    candidate_position
                ),
                "image_id": image_id,
                "semantic_id": record[
                    "semantic_id"
                ],
                "split": record["split"],
                "pattern_id": record[
                    "pattern_id"
                ],
                "ambiguity_level": record[
                    "ambiguity_level"
                ],
                "template_id": template_id,
                "candidate_role": candidate_role,
                "is_positive": boolean_text(
                    is_positive
                ),
                "relevance_label": relevance_label,
                "changed_attribute": (
                    changed_attribute
                ),
                "omitted_attribute": (
                    omitted_attribute
                ),
                "old_value": old_value,
                "new_value": new_value,
                "hamming_distance": hamming_distance,
                "negative_type": negative_type,
                "is_global_positive_text": (
                    boolean_text(
                        is_global_positive_text
                    )
                ),
                "is_negative_global_overlap": (
                    boolean_text(
                        is_negative_global_overlap
                    )
                ),
                "matching_positive_image_ids": (
                    "|".join(matching_images)
                ),
                "matching_positive_semantic_ids": (
                    "|".join(matching_semantics)
                ),
                "matching_positive_count": len(
                    matching_images
                ),
                "original_palette_id": original[
                    "palette_id"
                ],
                "original_motif": original["motif"],
                "original_orientation": original[
                    "orientation"
                ],
                "original_composition": original[
                    "composition"
                ],
                "original_symmetry": original[
                    "symmetry"
                ],
                "candidate_palette_id": (
                    candidate_attributes[
                        "palette_id"
                    ]
                ),
                "candidate_motif": (
                    candidate_attributes["motif"]
                ),
                "candidate_orientation": (
                    candidate_attributes[
                        "orientation"
                    ]
                ),
                "candidate_composition": (
                    candidate_attributes[
                        "composition"
                    ]
                ),
                "candidate_symmetry": (
                    candidate_attributes[
                        "symmetry"
                    ]
                ),
                "candidate_text": candidate_text,
                "candidate_sha256": sha256_text(
                    candidate_text
                ),
            }

            query_candidate_rows.append(
                candidate_row
            )
            candidate_rows.append(candidate_row)

            split_candidate_counts[
                record["split"]
            ] += 1

        if len(query_candidate_rows) != 5:
            raise AssertionError(
                f"{image_id}: número de candidatos "
                "inválido."
            )

        positive_candidates = [
            row
            for row in query_candidate_rows
            if row["is_positive"] == "true"
        ]

        if len(positive_candidates) != 1:
            raise AssertionError(
                f"{image_id}: debe existir un positivo."
            )

        positive_candidate = (
            positive_candidates[0]
        )

        candidate_ids = [
            row["candidate_id"]
            for row in query_candidate_rows
        ]

        query_rows.append(
            {
                "query_id": query_id,
                "query_index": query_index,
                "image_id": image_id,
                "semantic_id": record[
                    "semantic_id"
                ],
                "structure_id": record[
                    "structure_id"
                ],
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
                "template_id": template_id,
                "omitted_attribute": (
                    omitted_attribute
                ),
                "positive_position": (
                    current_positive_position
                ),
                "positive_candidate_id": (
                    positive_candidate[
                        "candidate_id"
                    ]
                ),
                "positive_caption_id": (
                    positive_caption[
                        "caption_id"
                    ]
                ),
                "positive_caption_text": (
                    positive_caption[
                        "caption_text"
                    ]
                ),
                "candidate_ids": "|".join(
                    candidate_ids
                ),
                "candidate_count": 5,
                "negative_count": 4,
                "global_overlap_negative_count": (
                    query_overlap_count
                ),
                "nonoverlap_negative_count": (
                    4 - query_overlap_count
                ),
            }
        )

    if len(query_rows) != 56:
        raise AssertionError(
            "No se generaron 56 consultas."
        )

    if len(candidate_rows) != 280:
        raise AssertionError(
            "No se generaron 280 candidatos."
        )

    positive_count = sum(
        row["is_positive"] == "true"
        for row in candidate_rows
    )

    negative_count = len(candidate_rows) - positive_count

    overlap_negative_count = sum(
        row["is_negative_global_overlap"]
        == "true"
        for row in candidate_rows
    )

    nonoverlap_negative_count = (
        negative_count - overlap_negative_count
    )

    if positive_count != 56:
        raise AssertionError(
            "El número de positivos no es 56."
        )

    if negative_count != 224:
        raise AssertionError(
            "El número de negativos no es 224."
        )

    expectations = config[
        "validation_expectations"
    ]

    if overlap_negative_count != int(
        expectations[
            "expected_global_overlap_count"
        ]
    ):
        raise AssertionError(
            "El número de solapamientos globales "
            "es inesperado."
        )

    if nonoverlap_negative_count != 174:
        raise AssertionError(
            "Se esperaban 174 negativos "
            "contrafactuales sin solapamiento."
        )

    if len(transition_counts) != int(
        expectations[
            "expected_distinct_cyclic_transitions"
        ]
    ):
        raise AssertionError(
            "No se cubrieron las 29 transiciones."
        )

    write_csv(
        QUERIES_PATH,
        query_rows,
        QUERY_FIELDS,
    )

    write_csv(
        CANDIDATES_PATH,
        candidate_rows,
        CANDIDATE_FIELDS,
    )

    expected_attributes = list(ATTRIBUTES)

    summary = {
        "schema_version": "1.0",
        "dataset_version": "v2",
        "stage": "hard_negative_generation",
        "config_sha256": sha256_file(
            CONFIG_PATH
        ),
        "vocabulary_sha256": sha256_file(
            VOCABULARY_PATH
        ),
        "multimodal_manifest_sha256": (
            sha256_file(MULTIMODAL_PATH)
        ),
        "positive_captions_sha256": (
            sha256_file(POSITIVE_CAPTIONS_PATH)
        ),
        "query_count": len(query_rows),
        "candidate_count": len(candidate_rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "global_overlap_negative_count": (
            overlap_negative_count
        ),
        "nonoverlap_negative_count": (
            nonoverlap_negative_count
        ),
        "candidates_per_query": 5,
        "negatives_per_query": 4,
        "chance_level": config["chance_level"],
        "omission_counts": {
            attribute: omission_counts[attribute]
            for attribute in expected_attributes
        },
        "change_counts": {
            attribute: change_counts[attribute]
            for attribute in expected_attributes
        },
        "positive_position_counts": {
            str(position): position_counts[position]
            for position in range(1, 6)
        },
        "global_overlap_by_attribute": {
            attribute: overlap_counts[attribute]
            for attribute in expected_attributes
        },
        "negative_type_counts": {
            "global_overlap": (
                negative_type_counts[
                    "global_overlap"
                ]
            ),
            "counterfactual_only": (
                negative_type_counts[
                    "counterfactual_only"
                ]
            ),
        },
        "candidate_counts_by_split": {
            split: split_candidate_counts[split]
            for split in (
                "id",
                "ood_palette",
                "ood_pattern",
                "ood_both",
            )
        },
        "omission_position_pair_count": len(
            omission_position_pairs
        ),
        "distinct_transition_count": len(
            transition_counts
        ),
        "query_path": (
            QUERIES_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "candidate_path": (
            CANDIDATES_PATH
            .relative_to(PROJECT_ROOT)
            .as_posix()
        ),
        "queries_sha256": sha256_file(
            QUERIES_PATH
        ),
        "candidates_sha256": sha256_file(
            CANDIDATES_PATH
        ),
        "generation_valid": True,
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print("=" * 76)
    print("NEGATIVOS DIFÍCILES V2 GENERADOS")
    print("=" * 76)
    print(
        "Consultas:                    "
        f"{summary['query_count']}"
    )
    print(
        "Candidatos:                   "
        f"{summary['candidate_count']}"
    )
    print(
        "Positivos:                    "
        f"{summary['positive_count']}"
    )
    print(
        "Negativos:                    "
        f"{summary['negative_count']}"
    )
    print(
        "Negativos con solapamiento:   "
        f"{summary['global_overlap_negative_count']}"
    )
    print(
        "Contrafactuales sin solapamiento: "
        f"{summary['nonoverlap_negative_count']}"
    )
    print(
        "Omisiones por atributo:       "
        f"{summary['omission_counts']}"
    )
    print(
        "Cambios por atributo:         "
        f"{summary['change_counts']}"
    )
    print(
        "Posiciones del positivo:      "
        f"{summary['positive_position_counts']}"
    )
    print(
        "Solapamientos por atributo:   "
        f"{summary['global_overlap_by_attribute']}"
    )
    print(
        "Transiciones distintas:       "
        f"{summary['distinct_transition_count']}"
    )
    print(
        "Generación válida:            "
        f"{summary['generation_valid']}"
    )

    print("\nArtefactos:")
    print(
        "- "
        + QUERIES_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print(
        "- "
        + CANDIDATES_PATH
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