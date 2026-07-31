"""Valida la consistencia interna del protocolo experimental v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "experimento_v2.json"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la configuración: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def validate_unique_ids(
    records: list[dict[str, Any]],
    field: str,
    label: str,
) -> list[str]:
    values = [str(record[field]) for record in records]

    if len(values) != len(set(values)):
        raise AssertionError(
            f"Hay {label} duplicados en el campo {field}."
        )

    return values


def main() -> None:
    config = load_config(CONFIG_PATH)

    project = config["project"]
    reproducibility = config["reproducibility"]
    dataset = config["dataset_v2"]
    metrics = config["metrics"]

    assert project["task"] == (
        "cross_modal_image_to_text_retrieval"
    )
    assert project["cultural_identification"] is False
    assert project["fine_tuning"] is False

    assert reproducibility["seed"] == 225
    assert reproducibility["relative_paths_only"] is True
    assert reproducibility["bootstrap_iterations"] == 2000

    base_patterns = dataset["base_patterns"]
    heldout_patterns = dataset["heldout_patterns"]
    base_palettes = dataset["base_palettes"]
    heldout_palettes = dataset["heldout_palettes"]

    base_pattern_ids = validate_unique_ids(
        base_patterns,
        "pattern_id",
        "patrones base",
    )
    heldout_pattern_ids = validate_unique_ids(
        heldout_patterns,
        "pattern_id",
        "patrones OOD",
    )

    assert set(base_pattern_ids).isdisjoint(
        heldout_pattern_ids
    ), "Los patrones base y OOD deben ser disjuntos."

    assert set(base_palettes).isdisjoint(
        heldout_palettes
    ), "Las paletas base y OOD deben ser disjuntas."

    assert len(base_patterns) == 6
    assert len(heldout_patterns) == 2
    assert len(base_palettes) == 5
    assert len(heldout_palettes) == 2

    expected_split_counts = {
        "id": len(base_patterns) * len(base_palettes),
        "ood_palette": (
            len(base_patterns) * len(heldout_palettes)
        ),
        "ood_pattern": (
            len(heldout_patterns) * len(base_palettes)
        ),
        "ood_both": (
            len(heldout_patterns)
            * len(heldout_palettes)
        ),
    }

    configured_splits = dataset["splits"]

    for split_name, expected_count in (
        expected_split_counts.items()
    ):
        configured_count = configured_splits[
            split_name
        ]["expected_images"]

        assert configured_count == expected_count, (
            f"{split_name}: se esperaban {expected_count} "
            f"imágenes, pero se configuraron "
            f"{configured_count}."
        )

    total_expected = sum(expected_split_counts.values())

    assert total_expected == dataset["expected_images"]
    assert total_expected == 56

    semantic_fields = dataset["semantic_id_fields"]
    negative_attributes = dataset[
        "hard_negative_attributes"
    ]

    assert len(semantic_fields) == 5
    assert len(semantic_fields) == len(
        set(semantic_fields)
    )
    assert set(negative_attributes) == set(
        semantic_fields
    )

    assert dataset["positive_captions_per_image"] == 5
    assert dataset["hard_negatives_per_query"] == 4

    assert metrics["primary"] == [
        "recall_at_1",
        "mrr",
        "ndcg_at_10",
    ]
    assert metrics["hard_negative_chance_level"] == 0.2

    experiment_ids = [
        experiment["experiment_id"]
        for experiment in config["experiments"]
    ]

    assert experiment_ids == [
        "E0",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
    ]
    assert len(experiment_ids) == len(
        set(experiment_ids)
    )

    assert len(config["baselines"]) == 2

    print("=" * 76)
    print("ESPECIFICACIÓN EXPERIMENTAL V2 VÁLIDA")
    print("=" * 76)
    print(f"Patrones base:          {len(base_patterns)}")
    print(f"Patrones OOD:           {len(heldout_patterns)}")
    print(f"Paletas base:           {len(base_palettes)}")
    print(f"Paletas OOD:            {len(heldout_palettes)}")
    print(f"Imágenes ID:            {expected_split_counts['id']}")
    print(
        "Imágenes OOD-paleta:    "
        f"{expected_split_counts['ood_palette']}"
    )
    print(
        "Imágenes OOD-patrón:    "
        f"{expected_split_counts['ood_pattern']}"
    )
    print(
        "Imágenes OOD-combinado: "
        f"{expected_split_counts['ood_both']}"
    )
    print(f"Total esperado:         {total_expected}")
    print(
        "Captions positivos:     "
        f"{total_expected * 5}"
    )
    print(
        "Negativos difíciles:    "
        f"{total_expected * 4}"
    )
    print(
        "Métricas principales:   "
        + ", ".join(metrics["primary"])
    )
    print(
        "Nivel de azar E2:       "
        f"{metrics['hard_negative_chance_level']:.2f}"
    )


if __name__ == "__main__":
    main()
