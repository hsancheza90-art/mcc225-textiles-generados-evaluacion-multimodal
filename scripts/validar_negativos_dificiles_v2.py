"""Valida las consultas y candidatos de negativos difíciles v2."""

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
    value: str,
    domain: list[str],
    offset: int,
) -> str:
    index = domain.index(value)

    return domain[
        (index + offset) % len(domain)
    ]


def assert_utf8_lf(path: Path) -> None:
    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
    )

    raw.decode("utf-8")


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

    queries = load_csv(QUERIES_PATH)
    candidates = load_csv(CANDIDATES_PATH)
    summary = load_json(SUMMARY_PATH)

    for path in (
        QUERIES_PATH,
        CANDIDATES_PATH,
        SUMMARY_PATH,
    ):
        assert_utf8_lf(path)

    assert len(multimodal) == 56
    assert len(queries) == 56
    assert len(candidates) == 280

    query_ids = [
        row["query_id"]
        for row in queries
    ]

    candidate_ids = [
        row["candidate_id"]
        for row in candidates
    ]

    assert len(set(query_ids)) == 56
    assert len(set(candidate_ids)) == 280

    queries_by_id = {
        row["query_id"]: row
        for row in queries
    }

    candidates_by_query: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in candidates:
        candidates_by_query[
            row["query_id"]
        ].append(row)

    canonical_by_image = {
        row["image_id"]: row
        for row in positive_captions
        if (
            row["template_id"]
            == config["template_id"]
            and row["is_canonical"] == "true"
        )
    }

    assert len(canonical_by_image) == 56

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

    templates = {
        item["template_id"]: item["text"]
        for item in vocabulary["full_templates"]
    }

    template = templates[config["template_id"]]
    attributes = list(config["attribute_order"])
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

    omission_counts: Counter[str] = Counter()
    change_counts: Counter[str] = Counter()
    position_counts: Counter[int] = Counter()
    overlap_counts: Counter[str] = Counter()
    negative_type_counts: Counter[str] = Counter()

    transition_counts: Counter[
        tuple[str, str, str]
    ] = Counter()

    omission_position_pairs: Counter[
        tuple[str, int]
    ] = Counter()

    positive_total = 0
    negative_total = 0

    for query_index, source in enumerate(multimodal):
        image_id = source["image_id"]
        query_id = f"HNQ-{image_id}"

        assert query_id in queries_by_id

        query = queries_by_id[query_id]

        assert int(query["query_index"]) == query_index
        assert query["image_id"] == image_id
        assert (
            query["semantic_id"]
            == source["semantic_id"]
        )
        assert (
            query["structure_id"]
            == source["structure_id"]
        )
        assert query["split"] == source["split"]
        assert (
            query["image_path"]
            == source["image_path"]
        )
        assert (
            query["template_id"]
            == config["template_id"]
        )

        for field in (
            "pattern_id",
            "palette_id",
            "motif",
            "orientation",
            "composition",
            "symmetry",
            "ambiguity_level",
        ):
            assert query[field] == source[field]

        original = {
            attribute: source[attribute]
            for attribute in attributes
        }

        omitted_attribute = attributes[
            (
                query_index
                + int(omission_schedule["offset"])
            )
            % len(attributes)
        ]

        expected_positive_position = (
            positive_position(
                query_index,
                position_schedule,
            )
        )

        assert (
            query["omitted_attribute"]
            == omitted_attribute
        )

        assert int(
            query["positive_position"]
        ) == expected_positive_position

        omission_counts[omitted_attribute] += 1
        position_counts[
            expected_positive_position
        ] += 1

        omission_position_pairs[
            (
                omitted_attribute,
                expected_positive_position,
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

        negative_positions = [
            position
            for position in range(1, 6)
            if position != expected_positive_position
        ]

        changed_by_position = dict(
            zip(
                negative_positions,
                rotated_attributes,
                strict=True,
            )
        )

        current_candidates = sorted(
            candidates_by_query[query_id],
            key=lambda row: int(
                row["candidate_position"]
            ),
        )

        assert len(current_candidates) == 5
        assert [
            int(row["candidate_position"])
            for row in current_candidates
        ] == [1, 2, 3, 4, 5]

        expected_candidate_ids = [
            f"HNC-{image_id}-P{position:02d}"
            for position in range(1, 6)
        ]

        assert [
            row["candidate_id"]
            for row in current_candidates
        ] == expected_candidate_ids

        assert (
            query["candidate_ids"]
            == "|".join(expected_candidate_ids)
        )

        assert int(query["candidate_count"]) == 5
        assert int(query["negative_count"]) == 4

        canonical = canonical_by_image[image_id]

        assert (
            query["positive_caption_id"]
            == canonical["caption_id"]
        )

        assert (
            query["positive_caption_text"]
            == canonical["caption_text"]
        )

        query_overlap_count = 0
        candidate_texts: set[str] = set()
        positive_rows = []

        for candidate in current_candidates:
            position = int(
                candidate["candidate_position"]
            )

            is_positive = (
                position
                == expected_positive_position
            )

            expected_candidate_id = (
                f"HNC-{image_id}-P{position:02d}"
            )

            assert (
                candidate["candidate_id"]
                == expected_candidate_id
            )

            assert candidate["image_id"] == image_id
            assert (
                candidate["semantic_id"]
                == source["semantic_id"]
            )
            assert (
                candidate["omitted_attribute"]
                == omitted_attribute
            )
            assert (
                candidate["template_id"]
                == config["template_id"]
            )
            assert candidate["split"] == source["split"]

            for attribute in attributes:
                assert (
                    candidate[
                        f"original_{attribute}"
                    ]
                    == original[attribute]
                )

            candidate_attributes = dict(original)

            if is_positive:
                expected_role = "positive"
                expected_label = 1
                expected_changed_attribute = ""
                expected_old_value = ""
                expected_new_value = ""
                expected_hamming = 0
            else:
                expected_role = "hard_negative"
                expected_label = 0

                expected_changed_attribute = (
                    changed_by_position[position]
                )

                expected_old_value = original[
                    expected_changed_attribute
                ]

                expected_new_value = cyclic_successor(
                    expected_old_value,
                    domains[
                        expected_changed_attribute
                    ],
                    replacement_offset,
                )

                candidate_attributes[
                    expected_changed_attribute
                ] = expected_new_value

                expected_hamming = 1

            assert (
                candidate["candidate_role"]
                == expected_role
            )

            assert (
                candidate["is_positive"]
                == (
                    "true"
                    if is_positive
                    else "false"
                )
            )

            assert int(
                candidate["relevance_label"]
            ) == expected_label

            assert (
                candidate["changed_attribute"]
                == expected_changed_attribute
            )

            assert (
                candidate["old_value"]
                == expected_old_value
            )

            assert (
                candidate["new_value"]
                == expected_new_value
            )

            assert int(
                candidate["hamming_distance"]
            ) == expected_hamming

            for attribute in attributes:
                assert (
                    candidate[
                        f"candidate_{attribute}"
                    ]
                    == candidate_attributes[
                        attribute
                    ]
                )

            expected_text = render_caption(
                candidate_attributes,
                template,
                vocabulary,
            )

            assert (
                candidate["candidate_text"]
                == expected_text
            )

            assert (
                candidate["candidate_sha256"]
                == sha256_text(expected_text)
            )

            assert expected_text not in candidate_texts
            candidate_texts.add(expected_text)

            matching_images = sorted(
                positive_images_by_text.get(
                    expected_text,
                    [],
                )
            )

            matching_semantics = sorted(
                positive_semantics_by_text.get(
                    expected_text,
                    [],
                )
            )

            expected_global_positive = bool(
                matching_images
            )

            expected_negative_overlap = (
                not is_positive
                and expected_global_positive
            )

            assert (
                candidate[
                    "is_global_positive_text"
                ]
                == (
                    "true"
                    if expected_global_positive
                    else "false"
                )
            )

            assert (
                candidate[
                    "is_negative_global_overlap"
                ]
                == (
                    "true"
                    if expected_negative_overlap
                    else "false"
                )
            )

            assert (
                candidate[
                    "matching_positive_image_ids"
                ]
                == "|".join(matching_images)
            )

            assert (
                candidate[
                    "matching_positive_semantic_ids"
                ]
                == "|".join(matching_semantics)
            )

            assert int(
                candidate["matching_positive_count"]
            ) == len(matching_images)

            if is_positive:
                expected_negative_type = (
                    "not_applicable"
                )

                assert matching_images == [image_id]

                positive_rows.append(candidate)
                positive_total += 1
            else:
                negative_total += 1

                change_counts[
                    expected_changed_attribute
                ] += 1

                transition_counts[
                    (
                        expected_changed_attribute,
                        expected_old_value,
                        expected_new_value,
                    )
                ] += 1

                if expected_negative_overlap:
                    expected_negative_type = (
                        "global_overlap"
                    )

                    overlap_counts[
                        expected_changed_attribute
                    ] += 1

                    query_overlap_count += 1
                else:
                    expected_negative_type = (
                        "counterfactual_only"
                    )

                negative_type_counts[
                    expected_negative_type
                ] += 1

            assert (
                candidate["negative_type"]
                == expected_negative_type
            )

        assert len(positive_rows) == 1

        positive_candidate = positive_rows[0]

        assert (
            query["positive_candidate_id"]
            == positive_candidate["candidate_id"]
        )

        assert int(
            query[
                "global_overlap_negative_count"
            ]
        ) == query_overlap_count

        assert int(
            query["nonoverlap_negative_count"]
        ) == 4 - query_overlap_count

    assert positive_total == 56
    assert negative_total == 224

    assert omission_counts == Counter({
        "palette_id": 12,
        "motif": 11,
        "orientation": 11,
        "composition": 11,
        "symmetry": 11,
    })

    assert change_counts == Counter({
        "palette_id": 44,
        "motif": 45,
        "orientation": 45,
        "composition": 45,
        "symmetry": 45,
    })

    assert position_counts == Counter({
        1: 11,
        2: 11,
        3: 12,
        4: 11,
        5: 11,
    })

    assert len(omission_position_pairs) == 25
    assert len(transition_counts) == 29

    assert overlap_counts == Counter({
        "palette_id": 44,
        "orientation": 6,
    })

    assert negative_type_counts == Counter({
        "global_overlap": 50,
        "counterfactual_only": 174,
    })

    assert summary["query_count"] == 56
    assert summary["candidate_count"] == 280
    assert summary["positive_count"] == 56
    assert summary["negative_count"] == 224

    assert (
        summary["global_overlap_negative_count"]
        == 50
    )

    assert (
        summary["nonoverlap_negative_count"]
        == 174
    )

    assert (
        summary["queries_sha256"]
        == sha256_file(QUERIES_PATH)
    )

    assert (
        summary["candidates_sha256"]
        == sha256_file(CANDIDATES_PATH)
    )

    assert (
        summary["config_sha256"]
        == sha256_file(CONFIG_PATH)
    )

    assert (
        summary["vocabulary_sha256"]
        == sha256_file(VOCABULARY_PATH)
    )

    assert summary["generation_valid"] is True

    print("=" * 76)
    print("VALIDACIÓN DE NEGATIVOS DIFÍCILES V2 SUPERADA")
    print("=" * 76)
    print("Consultas:                       56")
    print("Candidatos:                      280")
    print("Positivos:                       56")
    print("Negativos:                       224")
    print("Negativos con solapamiento:      50")
    print("Contrafactuales sin solapamiento: 174")
    print("Cambios por negativo:            1")
    print("Candidatos únicos por consulta:  5")
    print("Posiciones positivas balanceadas: confirmado")
    print("Pares omisión–posición:          25")
    print("Transiciones cíclicas:           29")
    print("Reconstrucción textual exacta:   confirmada")
    print("UTF-8 sin BOM y LF:              confirmado")
    print("Hashes del resumen:              válidos")


if __name__ == "__main__":
    main()