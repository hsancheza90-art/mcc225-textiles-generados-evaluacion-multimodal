"""Valida el contrato computacional de embeddings OpenCLIP v2."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "embeddings_v2.json"
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "config"
    / "experimento_v2.json"
)

ENVIRONMENT_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "entorno_cpu_v2.json"
)

DAMAGED_WORD_PATTERN = re.compile(
    r"[^\W\d_]+\?[^\W\d_]+",
    flags=re.UNICODE,
)

EXPECTED_SOURCE_ORDER = [
    "positivos",
    "sin_color",
    "negativos_dificiles",
]

EXPECTED_SOURCE_COUNTS = {
    "positivos": 280,
    "sin_color": 40,
    "negativos_dificiles": 280,
}

EXPECTED_OUTPUT_SUFFIXES = {
    "image_index_csv": ".csv",
    "text_index_csv": ".csv",
    "text_usage_csv": ".csv",
    "image_original_npy": ".npy",
    "image_grayscale_npy": ".npy",
    "text_unique_npy": ".npy",
    "summary_json": ".json",
}


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON UTF-8."""

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Carga un CSV y valida UTF-8 sin BOM y finales LF."""

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
    )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    return list(reader.fieldnames or []), rows


def sha256_text(text: str) -> str:
    """Calcula SHA-256 sobre la representación UTF-8."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def validate_relative_path(
    value: str,
    context: str,
) -> Path:
    """Valida una ruta relativa expresada con separadores POSIX."""

    assert value, f"{context}: ruta vacía."

    path = Path(value)

    assert not path.is_absolute(), (
        f"{context}: la ruta debe ser relativa."
    )

    assert ".." not in path.parts, (
        f"{context}: la ruta no puede contener '..'."
    )

    assert "\\" not in value, (
        f"{context}: debe usar separadores '/'."
    )

    return path


def validate_text(
    text: str,
    context: str,
) -> None:
    """Valida integridad mínima del texto."""

    assert text.strip(), (
        f"{context}: texto vacío."
    )

    assert "\ufffd" not in text, (
        f"{context}: contiene U+FFFD."
    )

    damaged_words = (
        DAMAGED_WORD_PATTERN.findall(text)
    )

    assert not damaged_words, (
        f"{context}: contiene palabras dañadas: "
        f"{sorted(set(damaged_words))}."
    )


def validate_configuration_file() -> dict[str, Any]:
    """Valida la codificación y carga el contrato."""

    raw = CONFIG_PATH.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), "embeddings_v2.json contiene BOM."

    assert b"\r\n" not in raw, (
        "embeddings_v2.json contiene CRLF."
    )

    return load_json(CONFIG_PATH)


def validate_model(
    config: dict[str, Any],
    experiment: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    """Comprueba coherencia entre contrato, experimento y entorno."""

    model = config["model"]
    experiment_model = experiment["model"]
    environment_model = environment["model"]

    assert model["library"] == "open_clip"
    assert model["architecture"] == "ViT-B-32"

    assert (
        model["pretrained"]
        == "laion2b_s34b_b79k"
    )

    assert model["mode"] == "zero_shot"
    assert model["frozen"] is True
    assert model["embedding_dimension"] == 512
    assert model["normalization"] == "l2"

    for field in (
        "library",
        "architecture",
        "pretrained",
    ):
        values = {
            "embeddings": model[field],
            "experimento": experiment_model[field],
            "entorno": environment_model[field],
        }

        assert len(set(values.values())) == 1, (
            f"El campo model.{field} no coincide "
            f"entre los tres contratos: {values}."
        )

    assert environment["environment_valid"] is True

    assert (
        environment["runtime"]["canonical_device"]
        == "cpu"
    )


def validate_images(
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Valida el manifiesto y el orden contractual de imágenes."""

    image_input = config["inputs"][
        "image_manifest"
    ]

    manifest_relative = validate_relative_path(
        image_input["path"],
        "inputs.image_manifest.path",
    )

    manifest_path = (
        PROJECT_ROOT
        / manifest_relative
    )

    columns, rows = load_csv(manifest_path)

    id_field = image_input["id_field"]
    path_field = image_input["path_field"]

    assert id_field in columns
    assert path_field in columns

    assert image_input["expected_count"] == 56
    assert len(rows) == 56

    image_ids = [
        row[id_field]
        for row in rows
    ]

    assert len(set(image_ids)) == 56

    for row_index, row in enumerate(rows):
        image_id = row[id_field]

        assert image_id, (
            f"Fila visual {row_index}: image_id vacío."
        )

        relative_path = validate_relative_path(
            row[path_field],
            f"imagen {image_id}",
        )

        image_path = (
            PROJECT_ROOT
            / relative_path
        )

        assert image_path.exists(), image_path

        assert image_path.suffix.lower() == ".png"

    return rows


def validate_text_sources(
    config: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
    dict[str, int],
]:
    """Valida fuentes, usos y deduplicación textual."""

    sources = config["inputs"]["text_sources"]

    source_order = [
        source["source_id"]
        for source in sources
    ]

    assert source_order == EXPECTED_SOURCE_ORDER

    assert source_order == config[
        "deduplication"
    ]["source_order"]

    all_texts: list[str] = []
    usage_keys: list[tuple[str, str]] = []
    source_counts: dict[str, int] = {}

    for source in sources:
        source_id = source["source_id"]

        relative_path = validate_relative_path(
            source["path"],
            f"fuente {source_id}",
        )

        path = PROJECT_ROOT / relative_path

        columns, rows = load_csv(path)

        record_id_field = source[
            "record_id_field"
        ]

        text_field = source["text_field"]

        assert record_id_field in columns
        assert text_field in columns

        expected_count = (
            EXPECTED_SOURCE_COUNTS[source_id]
        )

        assert (
            source["expected_count"]
            == expected_count
        )
        assert len(rows) == expected_count

        record_ids = [
            row[record_id_field]
            for row in rows
        ]

        assert len(record_ids) == len(
            set(record_ids)
        )

        source_counts[source_id] = len(rows)

        for row_index, row in enumerate(rows):
            record_id = row[record_id_field]
            text = row[text_field]

            assert record_id

            validate_text(
                text,
                (
                    f"{source_id}:"
                    f"{record_id}:"
                    f"fila={row_index}"
                ),
            )

            usage_keys.append(
                (
                    source_id,
                    record_id,
                )
            )

            all_texts.append(text)

    assert len(usage_keys) == 600
    assert len(set(usage_keys)) == 600
    assert len(all_texts) == 600

    unique_texts = list(
        dict.fromkeys(all_texts)
    )

    assert len(unique_texts) == 494

    hashes = [
        sha256_text(text)
        for text in unique_texts
    ]

    assert len(set(hashes)) == 494

    return (
        all_texts,
        unique_texts,
        source_counts,
    )


def validate_outputs(
    config: dict[str, Any],
) -> None:
    """Valida nombres, extensiones y ubicación de salidas."""

    outputs = config["outputs"]

    expected_keys = {
        "directory",
        *EXPECTED_OUTPUT_SUFFIXES,
    }

    assert set(outputs) == expected_keys

    directory = validate_relative_path(
        outputs["directory"],
        "outputs.directory",
    )

    output_paths: list[Path] = []

    for key, expected_suffix in (
        EXPECTED_OUTPUT_SUFFIXES.items()
    ):
        relative_path = validate_relative_path(
            outputs[key],
            f"outputs.{key}",
        )

        assert relative_path.suffix == expected_suffix

        assert directory in relative_path.parents, (
            f"{key}: debe encontrarse dentro "
            f"de {directory.as_posix()}."
        )

        output_paths.append(relative_path)

    assert len(output_paths) == len(
        set(output_paths)
    )


def validate_variants_and_shapes(
    config: dict[str, Any],
) -> None:
    """Valida transformaciones y dimensiones previstas."""

    assert config["image_variants"] == [
        {
            "variant_id": "original",
            "transform": "openclip_preprocess",
        },
        {
            "variant_id": "grayscale",
            "transform": (
                "pil_convert_L_then_RGB_"
                "then_openclip_preprocess"
            ),
        },
    ]

    assert config["expected_shapes"] == {
        "image_original": [56, 512],
        "image_grayscale": [56, 512],
        "text_unique": [494, 512],
        "image_index_rows": 56,
        "text_index_rows": 494,
        "text_usage_rows": 600,
    }


def validate_reproducibility(
    config: dict[str, Any],
) -> None:
    """Valida parámetros canónicos de ejecución."""

    assert config["reproducibility"] == {
        "seed": 225,
        "canonical_device": "cpu",
        "dtype": "float32",
        "torch_num_threads": 8,
        "image_batch_size": 8,
        "text_batch_size": 64,
    }

    validation = config["validation"]

    assert validation == {
        "finite_values_required": True,
        "l2_normalized": True,
        "norm_absolute_tolerance": 1e-5,
        "row_order_is_contractual": True,
        "relative_paths_only": True,
        "utf8_without_bom": True,
        "line_endings": "lf",
    }


def main() -> None:
    """Ejecuta la validación completa sin generar embeddings."""

    config = validate_configuration_file()
    experiment = load_json(EXPERIMENT_PATH)
    environment = load_json(ENVIRONMENT_PATH)

    assert config["schema_version"] == "1.0"

    assert (
        config["artifact_version"]
        == "embeddings-v2"
    )

    assert config["dataset_version"] == "v2"

    validate_model(
        config,
        experiment,
        environment,
    )

    validate_reproducibility(config)

    image_rows = validate_images(config)

    (
        all_texts,
        unique_texts,
        source_counts,
    ) = validate_text_sources(config)

    validate_outputs(config)
    validate_variants_and_shapes(config)

    deduplication = config["deduplication"]

    assert deduplication["method"] == (
        "exact_unicode_string"
    )

    assert deduplication["text_hash"] == (
        "sha256_utf8"
    )

    assert deduplication[
        "within_source_order"
    ] == "csv_row_order"

    assert deduplication[
        "expected_usage_count"
    ] == 600

    assert deduplication[
        "expected_unique_text_count"
    ] == 494

    print("=" * 80)
    print("CONTRATO DE EMBEDDINGS V2 VÁLIDO")
    print("=" * 80)
    print("Imágenes:", len(image_rows))
    print(
        "Variantes visuales:",
        len(config["image_variants"]),
    )
    print("Usos textuales:", len(all_texts))
    print("Textos únicos:", len(unique_texts))
    print(
        "Duplicaciones esperadas:",
        len(all_texts) - len(unique_texts),
    )
    print(
        "Dimensión:",
        config["model"][
            "embedding_dimension"
        ],
    )
    print(
        "Dispositivo:",
        config["reproducibility"][
            "canonical_device"
        ],
    )
    print(
        "Batch de imágenes:",
        config["reproducibility"][
            "image_batch_size"
        ],
    )
    print(
        "Batch de textos:",
        config["reproducibility"][
            "text_batch_size"
        ],
    )
    print(
        "Fuentes textuales:",
        source_counts,
    )
    print(
        "Matrices previstas:",
        {
            "original": [56, 512],
            "grayscale": [56, 512],
            "textos": [494, 512],
        },
    )
    print(
        "El validador no generó embeddings."
    )


if __name__ == "__main__":
    main()
