"""Valida el contrato reproducible de los baselines de E3."""

from __future__ import annotations

import colorsys
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINES_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "baselines_v2.json"
)

EXPERIMENT_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "experimento_v2.json"
)

CAPTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "v2"
    / "captions_positivos_v2.csv"
)

IMAGE_INDEX_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "embeddings"
    / "index_images_v2.csv"
)

OPENCLIP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "evaluacion"
    / "global_openclip_v2"
    / "resumen_global_openclip_v2.json"
)


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene CRLF."
    )

    return json.loads(
        raw.decode("utf-8")
    )


def load_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene CRLF."
    )

    raw.decode("utf-8")

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return list(reader.fieldnames or []), rows


def assert_close(
    actual: float,
    expected: float,
    context: str,
    tolerance: float = 1e-12,
) -> None:
    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ), (
        f"{context}: actual={actual}, "
        f"esperado={expected}."
    )


def histogram_bin(
    value: float,
    bin_count: int,
) -> int:
    assert 0.0 <= value
    assert value <= 1.0
    assert bin_count > 0

    index = int(
        math.floor(
            value * bin_count
        )
    )

    return min(
        index,
        bin_count - 1,
    )


def descriptor_from_rgb_counts(
    rgb_colors: list[tuple[int, int, int]],
    weights: np.ndarray,
    h_bins: int,
    s_bins: int,
    v_bins: int,
) -> np.ndarray:
    assert rgb_colors
    assert weights.ndim == 1
    assert len(rgb_colors) == weights.size
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert float(weights.sum()) > 0.0

    dimensions = (
        h_bins
        * s_bins
        * v_bins
    )

    descriptor = np.zeros(
        dimensions,
        dtype=np.float64,
    )

    normalized_weights = (
        weights
        / weights.sum()
    )

    for rgb, weight in zip(
        rgb_colors,
        normalized_weights,
        strict=True,
    ):
        red, green, blue = rgb

        assert 0 <= red
        assert red <= 255
        assert 0 <= green
        assert green <= 255
        assert 0 <= blue
        assert blue <= 255

        hue, saturation, value = (
            colorsys.rgb_to_hsv(
                red / 255.0,
                green / 255.0,
                blue / 255.0,
            )
        )

        h_index = histogram_bin(
            hue,
            h_bins,
        )

        s_index = histogram_bin(
            saturation,
            s_bins,
        )

        v_index = histogram_bin(
            value,
            v_bins,
        )

        flat_index = (
            (
                h_index
                * s_bins
                + s_index
            )
            * v_bins
            + v_index
        )

        descriptor[
            flat_index
        ] += float(weight)

    assert descriptor.shape == (
        dimensions,
    )

    assert np.isfinite(
        descriptor
    ).all()

    assert np.all(
        descriptor >= 0.0
    )

    assert_close(
        float(descriptor.sum()),
        1.0,
        "descriptor_l1",
    )

    return descriptor


def histogram_intersection(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    assert left.shape == right.shape
    assert left.ndim == 1

    score = float(
        np.minimum(
            left,
            right,
        ).sum()
    )

    assert score >= -1e-12
    assert score <= 1.0 + 1e-12

    return score


def build_palette_mapping(
    experiment: dict[str, Any],
) -> dict[str, list[tuple[int, int, int]]]:
    dataset = experiment[
        "dataset_v2"
    ]

    mapping: dict[
        str,
        list[tuple[int, int, int]],
    ] = {}

    for section_name in (
        "base_palettes",
        "heldout_palettes",
    ):
        for palette_id, colors in (
            dataset[
                section_name
            ].items()
        ):
            assert palette_id not in mapping

            mapping[palette_id] = [
                tuple(
                    int(channel)
                    for channel in color
                )
                for color in colors
            ]

    assert len(mapping) == 7

    return mapping


def validate_random_contract(
    config: dict[str, Any],
) -> None:
    random_config = config[
        "conditions"
    ][
        "random"
    ]

    assert random_config["seed"] == 225

    assert (
        random_config["generator"]
        == "numpy.random.Generator"
    )

    assert (
        random_config[
            "bit_generator"
        ]
        == "PCG64"
    )

    assert (
        random_config[
            "distribution"
        ]
        == "uniform_0_1"
    )

    assert (
        random_config[
            "matrix_shape"
        ]
        == [280, 56]
    )

    seed = int(
        random_config["seed"]
    )

    generator_a = np.random.Generator(
        np.random.PCG64(seed)
    )

    generator_b = np.random.Generator(
        np.random.PCG64(seed)
    )

    matrix_a = generator_a.random(
        (280, 56)
    )

    matrix_b = generator_b.random(
        (280, 56)
    )

    assert matrix_a.shape == (
        280,
        56,
    )

    assert matrix_a.dtype == np.float64
    assert np.isfinite(matrix_a).all()
    assert np.all(matrix_a >= 0.0)
    assert np.all(matrix_a < 1.0)

    assert np.array_equal(
        matrix_a,
        matrix_b,
    )

    different_generator = (
        np.random.Generator(
            np.random.PCG64(
                seed + 1
            )
        )
    )

    matrix_different = (
        different_generator.random(
            (280, 56)
        )
    )

    assert not np.array_equal(
        matrix_a,
        matrix_different,
    )

    print(
        "Matriz aleatoria reproducible:",
        matrix_a.shape,
    )

    print(
        "Primer score aleatorio:",
        format(
            float(matrix_a[0, 0]),
            ".12f",
        ),
    )


def validate_random_expectation(
    config: dict[str, Any],
) -> None:
    record = config[
        "theoretical_random_expectation"
    ]

    gallery_size = int(
        record["gallery_size"]
    )

    assert gallery_size == 56

    expected_recall_at_1 = (
        1.0 / gallery_size
    )

    expected_recall_at_5 = (
        5.0 / gallery_size
    )

    expected_mrr = (
        sum(
            1.0 / rank
            for rank in range(
                1,
                gallery_size + 1,
            )
        )
        / gallery_size
    )

    expected_ndcg_at_10 = (
        sum(
            1.0
            / math.log2(rank + 1.0)
            for rank in range(1, 11)
        )
        / gallery_size
    )

    assert_close(
        float(
            record["recall_at_1"]
        ),
        expected_recall_at_1,
        "random_expectation.recall_at_1",
    )

    assert_close(
        float(
            record["recall_at_5"]
        ),
        expected_recall_at_5,
        "random_expectation.recall_at_5",
    )

    assert_close(
        float(record["mrr"]),
        expected_mrr,
        "random_expectation.mrr",
    )

    assert_close(
        float(
            record["ndcg_at_10"]
        ),
        expected_ndcg_at_10,
        "random_expectation.ndcg_at_10",
    )


def main() -> None:
    config = load_json(
        BASELINES_CONFIG_PATH
    )

    experiment = load_json(
        EXPERIMENT_CONFIG_PATH
    )

    openclip_summary = load_json(
        OPENCLIP_SUMMARY_PATH
    )

    caption_fields, captions = load_csv(
        CAPTIONS_PATH
    )

    image_fields, image_rows = load_csv(
        IMAGE_INDEX_PATH
    )

    assert config["schema_version"] == "1.0"
    assert config["dataset_version"] == "v2"
    assert config["experiment_id"] == "E3"

    assert (
        config["experiment_name"]
        == "baseline_comparison"
    )

    experiments = {
        record["experiment_id"]: record
        for record in experiment[
            "experiments"
        ]
    }

    assert experiments["E3"] == {
        "experiment_id": "E3",
        "name": "baseline_comparison",
        "conditions": [
            "openclip",
            "random",
            "color_histogram",
        ],
    }

    baselines = {
        record["baseline_id"]: record
        for record in experiment[
            "baselines"
        ]
    }

    assert baselines["random"][
        "seed"
    ] == 225

    assert (
        baselines[
            "color_histogram"
        ][
            "color_space"
        ]
        == "HSV"
    )

    assert (
        baselines[
            "color_histogram"
        ][
            "uses_only_attribute"
        ]
        == "palette_id"
    )

    task = config["task"]

    assert task["query_count"] == 280
    assert task["gallery_count"] == 56

    assert (
        task[
            "relevant_images_per_query"
        ]
        == 1
    )

    assert (
        task["tie_breaker"]
        == "image_id_ascending"
    )

    assert len(captions) == 280
    assert len(image_rows) == 56

    assert {
        "caption_id",
        "image_id",
        "palette_id",
    }.issubset(
        set(caption_fields)
    )

    assert {
        "image_row_index",
        "image_id",
        "palette_id",
        "image_path",
    }.issubset(
        set(image_fields)
    )

    assert (
        openclip_summary[
            "experiment_id"
        ]
        == "E1"
    )

    assert (
        openclip_summary[
            "evaluation_valid"
        ]
        is True
    )

    assert (
        openclip_summary[
            "counts"
        ][
            "queries"
        ]
        == 280
    )

    assert (
        openclip_summary[
            "counts"
        ][
            "gallery_images"
        ]
        == 56
    )

    validate_random_contract(
        config
    )

    validate_random_expectation(
        config
    )

    color_config = config[
        "conditions"
    ][
        "color_histogram"
    ]

    assert (
        color_config[
            "color_space"
        ]
        == "HSV"
    )

    assert (
        color_config[
            "rgb_to_hsv_implementation"
        ]
        == "colorsys.rgb_to_hsv"
    )

    assert (
        color_config[
            "image_descriptor"
        ]
        == "l1_normalized_pixel_histogram"
    )

    assert (
        color_config[
            "text_descriptor"
        ]
        == (
            "uniform_mass_over_"
            "configured_palette_colors"
        )
    )

    assert (
        color_config["similarity"]
        == "histogram_intersection"
    )

    h_bins = int(
        color_config["bins"]["h"]
    )

    s_bins = int(
        color_config["bins"]["s"]
    )

    v_bins = int(
        color_config["bins"]["v"]
    )

    assert h_bins == 18
    assert s_bins == 4
    assert v_bins == 4

    dimensions = (
        h_bins
        * s_bins
        * v_bins
    )

    assert dimensions == 288

    assert (
        color_config[
            "descriptor_dimensions"
        ]
        == dimensions
    )

    palette_mapping = (
        build_palette_mapping(
            experiment
        )
    )

    palette_descriptors = {}

    for palette_id, colors in (
        palette_mapping.items()
    ):
        weights = np.ones(
            len(colors),
            dtype=np.float64,
        )

        palette_descriptors[
            palette_id
        ] = descriptor_from_rgb_counts(
            rgb_colors=colors,
            weights=weights,
            h_bins=h_bins,
            s_bins=s_bins,
            v_bins=v_bins,
        )

    descriptor_bytes = {
        descriptor.tobytes()
        for descriptor
        in palette_descriptors.values()
    }

    assert len(descriptor_bytes) == 7

    palette_ids = sorted(
        palette_mapping
    )

    prototype_similarity = np.zeros(
        (7, 7),
        dtype=np.float64,
    )

    for left_index, left_id in enumerate(
        palette_ids
    ):
        for right_index, right_id in enumerate(
            palette_ids
        ):
            prototype_similarity[
                left_index,
                right_index,
            ] = histogram_intersection(
                palette_descriptors[
                    left_id
                ],
                palette_descriptors[
                    right_id
                ],
            )

    assert np.allclose(
        np.diag(
            prototype_similarity
        ),
        1.0,
        atol=1e-12,
    )

    for row_index in range(7):
        other_scores = np.delete(
            prototype_similarity[
                row_index
            ],
            row_index,
        )

        assert float(
            other_scores.max()
        ) < 1.0

    ordered_images = sorted(
        image_rows,
        key=lambda row: int(
            row["image_row_index"]
        ),
    )

    assert [
        int(row["image_row_index"])
        for row in ordered_images
    ] == list(range(56))

    palette_top1_correct = 0
    self_margins = []

    for image_record in ordered_images:
        image_path = (
            PROJECT_ROOT
            / image_record["image_path"]
        )

        image = np.asarray(
            Image.open(
                image_path
            ).convert("RGB"),
            dtype=np.uint8,
        )

        pixels = image.reshape(-1, 3)

        colors_array, counts = np.unique(
            pixels,
            axis=0,
            return_counts=True,
        )

        colors = [
            tuple(
                int(channel)
                for channel in color
            )
            for color in colors_array
        ]

        image_descriptor = (
            descriptor_from_rgb_counts(
                rgb_colors=colors,
                weights=counts.astype(
                    np.float64
                ),
                h_bins=h_bins,
                s_bins=s_bins,
                v_bins=v_bins,
            )
        )

        scores = {
            palette_id: (
                histogram_intersection(
                    image_descriptor,
                    palette_descriptor,
                )
            )
            for palette_id, palette_descriptor
            in palette_descriptors.items()
        }

        ranked_palettes = sorted(
            scores,
            key=lambda palette_id: (
                -scores[palette_id],
                palette_id,
            ),
        )

        expected_palette = (
            image_record["palette_id"]
        )

        if (
            ranked_palettes[0]
            == expected_palette
        ):
            palette_top1_correct += 1

        other_scores = [
            score
            for palette_id, score
            in scores.items()
            if palette_id
            != expected_palette
        ]

        self_margins.append(
            scores[expected_palette]
            - max(other_scores)
        )

    palette_accuracy = (
        palette_top1_correct
        / 56.0
    )

    minimum_self_margin = min(
        self_margins
    )

    caption_palette_counts = Counter(
        row["palette_id"]
        for row in captions
    )

    assert set(
        caption_palette_counts
    ) == set(
        palette_mapping
    )

    assert set(
        caption_palette_counts.values()
    ) == {40}

    print("=" * 80)
    print("VALIDACIÓN DEL CONTRATO DE BASELINES V2 SUPERADA")
    print("=" * 80)
    print("Experimento: E3 baseline_comparison")
    print("Consultas: 280")
    print("Galería: 56")
    print("Semilla aleatoria: 225")
    print("Generador aleatorio: NumPy PCG64")
    print(
        "Descriptor HSV:",
        dimensions,
        "dimensiones",
    )
    print(
        "Bins:",
        {
            "h": h_bins,
            "s": s_bins,
            "v": v_bins,
        },
    )
    print(
        "Prototipos de paleta distintos:",
        len(descriptor_bytes),
    )
    print(
        "Captions por paleta:",
        dict(
            sorted(
                caption_palette_counts.items()
            )
        ),
    )
    print(
        "Acierto top-1 de paleta visual:",
        format(
            palette_accuracy,
            ".12f",
        ),
    )
    print(
        "Margen mínimo de paleta correcta:",
        format(
            minimum_self_margin,
            ".12f",
        ),
    )
    print()
    print("Expectativa aleatoria teórica:")
    print(
        "- Recall@1:",
        format(
            config[
                "theoretical_random_expectation"
            ][
                "recall_at_1"
            ],
            ".12f",
        ),
    )
    print(
        "- Recall@5:",
        format(
            config[
                "theoretical_random_expectation"
            ][
                "recall_at_5"
            ],
            ".12f",
        ),
    )
    print(
        "- MRR:",
        format(
            config[
                "theoretical_random_expectation"
            ][
                "mrr"
            ],
            ".12f",
        ),
    )
    print(
        "- nDCG@10:",
        format(
            config[
                "theoretical_random_expectation"
            ][
                "ndcg_at_10"
            ],
            ".12f",
        ),
    )
    print()
    print(
        "No se generaron resultados experimentales."
    )
    print("Contrato válido: True")


if __name__ == "__main__":
    main()
