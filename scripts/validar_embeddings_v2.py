"""Valida independientemente los embeddings OpenCLIP del experimento v2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


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

IMAGE_INDEX_FIELDS = (
    "image_row_index",
    "image_id",
    "semantic_id",
    "semantic_signature",
    "split",
    "pattern_id",
    "palette_id",
    "motif",
    "orientation",
    "composition",
    "symmetry",
    "ambiguity_level",
    "image_path",
    "file_sha256",
    "pixel_sha256",
)

TEXT_INDEX_FIELDS = (
    "text_row_index",
    "text_id",
    "text_sha256",
    "usage_count",
    "first_source_id",
    "first_record_id",
    "text",
)

TEXT_USAGE_FIELDS = (
    "usage_row_index",
    "source_id",
    "source_row_index",
    "record_id",
    "text_row_index",
    "text_id",
    "text_sha256",
)


def load_json(path: Path) -> dict[str, Any]:
    """Carga un JSON UTF-8 sin BOM y con finales LF."""

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM UTF-8."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
    )

    return json.loads(
        raw.decode("utf-8")
    )


def load_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """Carga un CSV UTF-8 sin BOM y con finales LF."""

    raw = path.read_bytes()

    assert not raw.startswith(
        b"\xef\xbb\xbf"
    ), f"{path}: contiene BOM UTF-8."

    assert b"\r\n" not in raw, (
        f"{path}: contiene finales CRLF."
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


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 sobre los bytes de un archivo."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Calcula SHA-256 sobre un texto codificado en UTF-8."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def validate_text(
    text: str,
    context: str,
) -> None:
    """Comprueba la integridad mínima de un texto."""

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


def assert_float_close(
    actual: float,
    expected: float,
    context: str,
    tolerance: float = 1e-9,
) -> None:
    """Compara dos valores reales con tolerancia absoluta."""

    assert math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ), (
        f"{context}: actual={actual}, "
        f"esperado={expected}."
    )


def resolve_output_paths(
    config: dict[str, Any],
) -> dict[str, Path]:
    """Resuelve las rutas contractuales de salida."""

    output_paths: dict[str, Path] = {}

    for key, value in config[
        "outputs"
    ].items():
        output_paths[key] = (
            PROJECT_ROOT
            / value
        )

    return output_paths


def validate_output_inventory(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """Comprueba que existan exactamente los siete artefactos."""

    output_directory = paths["directory"]

    assert output_directory.is_dir(), (
        f"No existe {output_directory}."
    )

    expected_names = {
        Path(value).name
        for key, value in config[
            "outputs"
        ].items()
        if key != "directory"
    }

    actual_files = {
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
    }

    actual_directories = [
        path.name
        for path in output_directory.iterdir()
        if path.is_dir()
    ]

    assert not actual_directories, (
        "El directorio contiene subdirectorios "
        f"inesperados: {actual_directories}."
    )

    assert actual_files == expected_names, (
        "Inventario de salidas incorrecto: "
        f"esperado={sorted(expected_names)}, "
        f"actual={sorted(actual_files)}."
    )


def reconstruct_image_index(
    config: dict[str, Any],
) -> list[dict[str, str]]:
    """Reconstruye el índice visual desde el manifiesto."""

    specification = config[
        "inputs"
    ]["image_manifest"]

    manifest_path = (
        PROJECT_ROOT
        / specification["path"]
    )

    fields, rows = load_csv(
        manifest_path
    )

    assert specification["id_field"] in fields
    assert specification["path_field"] in fields
    assert specification["expected_count"] == 56
    assert len(rows) == 56

    image_ids = [
        row[specification["id_field"]]
        for row in rows
    ]

    assert len(set(image_ids)) == 56

    expected_rows: list[dict[str, str]] = []

    for row_index, row in enumerate(rows):
        image_path = (
            PROJECT_ROOT
            / row[specification["path_field"]]
        )

        assert image_path.exists()
        assert image_path.suffix.lower() == ".png"

        assert (
            sha256_file(image_path)
            == row["file_sha256"]
        )

        expected_record = {
            "image_row_index": str(
                row_index
            ),
        }

        for field in IMAGE_INDEX_FIELDS:
            if field == "image_row_index":
                continue

            expected_record[field] = row[field]

        expected_rows.append(
            expected_record
        )

    return expected_rows


def reconstruct_text_indices(
    config: dict[str, Any],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
]:
    """Reconstruye textos únicos y usos desde las fuentes."""

    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    text_index_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}

    usage_row_index = 0

    expected_source_order = config[
        "deduplication"
    ]["source_order"]

    actual_source_order = [
        source["source_id"]
        for source in config[
            "inputs"
        ]["text_sources"]
    ]

    assert (
        actual_source_order
        == expected_source_order
    )

    for source in config[
        "inputs"
    ]["text_sources"]:
        source_id = source["source_id"]

        source_path = (
            PROJECT_ROOT
            / source["path"]
        )

        fields, rows = load_csv(
            source_path
        )

        record_field = source[
            "record_id_field"
        ]

        text_field = source["text_field"]

        assert record_field in fields
        assert text_field in fields

        expected_count = source[
            "expected_count"
        ]

        assert len(rows) == expected_count

        source_counts[source_id] = len(
            rows
        )

        record_ids = [
            row[record_field]
            for row in rows
        ]

        assert len(record_ids) == len(
            set(record_ids)
        )

        for source_row_index, row in enumerate(
            rows
        ):
            record_id = row[record_field]
            text = row[text_field]

            validate_text(
                text,
                f"{source_id}:{record_id}",
            )

            text_hash = sha256_text(text)

            source_hash_fields = [
                field
                for field in row
                if field.endswith("_sha256")
            ]

            assert len(source_hash_fields) <= 1

            if source_hash_fields:
                assert (
                    row[source_hash_fields[0]]
                    == text_hash
                )

            if text not in text_to_index:
                text_row_index = len(
                    unique_texts
                )

                text_to_index[text] = (
                    text_row_index
                )

                unique_texts.append(text)

                text_index_rows.append(
                    {
                        "text_row_index": (
                            text_row_index
                        ),
                        "text_id": (
                            f"TXT-"
                            f"{text_row_index + 1:04d}"
                        ),
                        "text_sha256": (
                            text_hash
                        ),
                        "usage_count": 0,
                        "first_source_id": (
                            source_id
                        ),
                        "first_record_id": (
                            record_id
                        ),
                        "text": text,
                    }
                )

            text_row_index = (
                text_to_index[text]
            )

            text_record = (
                text_index_rows[
                    text_row_index
                ]
            )

            assert (
                text_record["text_sha256"]
                == text_hash
            )

            text_record["usage_count"] += 1

            usage_rows.append(
                {
                    "usage_row_index": str(
                        usage_row_index
                    ),
                    "source_id": source_id,
                    "source_row_index": str(
                        source_row_index
                    ),
                    "record_id": record_id,
                    "text_row_index": str(
                        text_row_index
                    ),
                    "text_id": str(
                        text_record["text_id"]
                    ),
                    "text_sha256": (
                        text_hash
                    ),
                }
            )

            usage_row_index += 1

    assert len(unique_texts) == 494
    assert len(text_index_rows) == 494
    assert len(usage_rows) == 600

    normalized_text_index = [
        {
            field: str(row[field])
            for field in TEXT_INDEX_FIELDS
        }
        for row in text_index_rows
    ]

    assert sum(
        int(row["usage_count"])
        for row in normalized_text_index
    ) == 600

    assert len(
        {
            row["text_sha256"]
            for row in normalized_text_index
        }
    ) == 494

    return (
        normalized_text_index,
        usage_rows,
        source_counts,
    )


def validate_image_index(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> list[dict[str, str]]:
    """Valida el índice de imágenes contra su reconstrucción."""

    fields, actual_rows = load_csv(
        paths["image_index_csv"]
    )

    assert tuple(fields) == (
        IMAGE_INDEX_FIELDS
    )

    expected_rows = (
        reconstruct_image_index(config)
    )

    assert actual_rows == expected_rows

    assert [
        int(row["image_row_index"])
        for row in actual_rows
    ] == list(range(56))

    return actual_rows


def validate_text_indices(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
]:
    """Valida el índice textual y la tabla de usos."""

    text_fields, actual_text_rows = (
        load_csv(
            paths["text_index_csv"]
        )
    )

    usage_fields, actual_usage_rows = (
        load_csv(
            paths["text_usage_csv"]
        )
    )

    assert tuple(text_fields) == (
        TEXT_INDEX_FIELDS
    )

    assert tuple(usage_fields) == (
        TEXT_USAGE_FIELDS
    )

    (
        expected_text_rows,
        expected_usage_rows,
        source_counts,
    ) = reconstruct_text_indices(config)

    assert (
        actual_text_rows
        == expected_text_rows
    )

    assert (
        actual_usage_rows
        == expected_usage_rows
    )

    assert [
        int(row["text_row_index"])
        for row in actual_text_rows
    ] == list(range(494))

    assert [
        row["text_id"]
        for row in actual_text_rows
    ] == [
        f"TXT-{index + 1:04d}"
        for index in range(494)
    ]

    assert [
        int(row["usage_row_index"])
        for row in actual_usage_rows
    ] == list(range(600))

    for usage in actual_usage_rows:
        text_row_index = int(
            usage["text_row_index"]
        )

        text_record = actual_text_rows[
            text_row_index
        ]

        assert (
            usage["text_id"]
            == text_record["text_id"]
        )

        assert (
            usage["text_sha256"]
            == text_record["text_sha256"]
        )

    return (
        actual_text_rows,
        actual_usage_rows,
        source_counts,
    )


def validate_matrix(
    path: Path,
    expected_shape: tuple[int, int],
    tolerance: float,
) -> tuple[
    np.ndarray,
    dict[str, Any],
]:
    """Valida una matriz NPY y calcula sus estadísticas."""

    assert path.exists()
    assert path.suffix == ".npy"

    matrix = np.load(
        path,
        allow_pickle=False,
    )

    assert matrix.shape == expected_shape
    assert matrix.dtype == np.dtype(
        "float32"
    )
    assert np.isfinite(matrix).all()

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    maximum_norm_error = float(
        np.max(
            np.abs(norms - 1.0)
        )
    )

    assert maximum_norm_error <= tolerance

    statistics = {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "finite": True,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "norm_mean": float(norms.mean()),
        "maximum_norm_error": (
            maximum_norm_error
        ),
    }

    return matrix, statistics


def validate_matrix_summary(
    matrix_key: str,
    path: Path,
    statistics: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Compara una matriz con los metadatos del resumen."""

    record = summary["matrices"][
        matrix_key
    ]

    assert (
        record["path"]
        == path.relative_to(
            PROJECT_ROOT
        ).as_posix()
    )

    assert (
        record["shape"]
        == statistics["shape"]
    )

    assert (
        record["dtype"]
        == statistics["dtype"]
    )

    assert record["finite"] is True

    assert (
        record["sha256"]
        == sha256_file(path)
    )

    for field in (
        "norm_min",
        "norm_max",
        "norm_mean",
        "maximum_norm_error",
    ):
        assert_float_close(
            float(record[field]),
            float(statistics[field]),
            f"{matrix_key}.{field}",
        )


def validate_input_hashes(
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Valida los hashes de todas las entradas registradas."""

    inputs = summary[
        "input_artifacts"
    ]

    fixed_inputs = {
        "embeddings_config": CONFIG_PATH,
        "experiment_config": (
            EXPERIMENT_PATH
        ),
        "environment_audit": (
            ENVIRONMENT_PATH
        ),
    }

    for key, path in fixed_inputs.items():
        record = inputs[key]

        assert (
            record["path"]
            == path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )

    image_specification = config[
        "inputs"
    ]["image_manifest"]

    image_manifest = (
        PROJECT_ROOT
        / image_specification["path"]
    )

    image_record = inputs[
        "image_manifest"
    ]

    assert (
        image_record["path"]
        == image_specification["path"]
    )

    assert image_record["records"] == 56

    assert (
        image_record["sha256"]
        == sha256_file(image_manifest)
    )

    source_records = inputs[
        "text_sources"
    ]

    for source in config[
        "inputs"
    ]["text_sources"]:
        source_id = source["source_id"]
        path = PROJECT_ROOT / source["path"]
        record = source_records[source_id]

        assert record["path"] == source["path"]

        assert (
            record["records"]
            == source["expected_count"]
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )


def validate_summary(
    config: dict[str, Any],
    paths: dict[str, Path],
    image_rows: list[dict[str, str]],
    text_rows: list[dict[str, str]],
    usage_rows: list[dict[str, str]],
    source_counts: dict[str, int],
    original: np.ndarray,
    grayscale: np.ndarray,
    original_statistics: dict[str, Any],
    grayscale_statistics: dict[str, Any],
    text_statistics: dict[str, Any],
) -> dict[str, Any]:
    """Valida el resumen completo de la generación."""

    summary = load_json(
        paths["summary_json"]
    )

    assert summary["schema_version"] == "1.0"
    assert summary["dataset_version"] == "v2"

    assert (
        summary["artifact_version"]
        == config["artifact_version"]
    )

    assert summary["stage"] == (
        "openclip_embeddings_generation"
    )

    assert summary["generation_valid"] is True

    assert (
        summary["model"]["architecture"]
        == config["model"]["architecture"]
    )

    assert (
        summary["model"]["pretrained"]
        == config["model"]["pretrained"]
    )

    assert (
        summary["model"]["library"]
        == config["model"]["library"]
    )

    assert (
        summary["model"][
            "embedding_dimension"
        ]
        == 512
    )

    assert (
        summary["model"][
            "trainable_parameters"
        ]
        == 0
    )

    assert (
        summary["runtime"]["device"]
        == "cpu"
    )

    assert (
        summary["runtime"][
            "torch_num_threads"
        ]
        == config["reproducibility"][
            "torch_num_threads"
        ]
    )

    assert (
        summary["reproducibility"]
        == config["reproducibility"]
    )

    expected_counts = {
        "images": len(image_rows),
        "image_variants": 2,
        "text_usages": len(usage_rows),
        "unique_texts": len(text_rows),
        "deduplicated_usages": (
            len(usage_rows)
            - len(text_rows)
        ),
        "text_sources": source_counts,
    }

    assert (
        summary["counts"]
        == expected_counts
    )

    validate_input_hashes(
        config,
        summary,
    )

    validate_matrix_summary(
        "image_original",
        paths["image_original_npy"],
        original_statistics,
        summary,
    )

    validate_matrix_summary(
        "image_grayscale",
        paths["image_grayscale_npy"],
        grayscale_statistics,
        summary,
    )

    validate_matrix_summary(
        "text_unique",
        paths["text_unique_npy"],
        text_statistics,
        summary,
    )

    index_expectations = {
        "images": (
            paths["image_index_csv"],
            len(image_rows),
        ),
        "unique_texts": (
            paths["text_index_csv"],
            len(text_rows),
        ),
        "text_usages": (
            paths["text_usage_csv"],
            len(usage_rows),
        ),
    }

    for key, (
        path,
        expected_rows,
    ) in index_expectations.items():
        record = summary["indices"][key]

        assert (
            record["path"]
            == path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        )

        assert (
            record["rows"]
            == expected_rows
        )

        assert (
            record["sha256"]
            == sha256_file(path)
        )

    assert not np.array_equal(
        original,
        grayscale,
    )

    rowwise_cosines = np.sum(
        original * grayscale,
        axis=1,
    )

    ablation = summary[
        "visual_ablation_check"
    ]

    assert (
        ablation[
            "original_and_grayscale_differ"
        ]
        is True
    )

    assert_float_close(
        float(
            ablation[
                "rowwise_cosine_min"
            ]
        ),
        float(rowwise_cosines.min()),
        "rowwise_cosine_min",
    )

    assert_float_close(
        float(
            ablation[
                "rowwise_cosine_max"
            ]
        ),
        float(rowwise_cosines.max()),
        "rowwise_cosine_max",
    )

    assert_float_close(
        float(
            ablation[
                "rowwise_cosine_mean"
            ]
        ),
        float(rowwise_cosines.mean()),
        "rowwise_cosine_mean",
    )

    return summary


def main() -> None:
    """Ejecuta la validación independiente completa."""

    config = load_json(CONFIG_PATH)
    paths = resolve_output_paths(
        config
    )

    validate_output_inventory(
        config,
        paths,
    )

    image_rows = validate_image_index(
        config,
        paths,
    )

    (
        text_rows,
        usage_rows,
        source_counts,
    ) = validate_text_indices(
        config,
        paths,
    )

    tolerance = config[
        "validation"
    ]["norm_absolute_tolerance"]

    original, original_statistics = (
        validate_matrix(
            paths["image_original_npy"],
            (56, 512),
            tolerance,
        )
    )

    grayscale, grayscale_statistics = (
        validate_matrix(
            paths["image_grayscale_npy"],
            (56, 512),
            tolerance,
        )
    )

    text_matrix, text_statistics = (
        validate_matrix(
            paths["text_unique_npy"],
            (494, 512),
            tolerance,
        )
    )

    assert (
        text_matrix.shape[0]
        == len(text_rows)
    )

    summary = validate_summary(
        config=config,
        paths=paths,
        image_rows=image_rows,
        text_rows=text_rows,
        usage_rows=usage_rows,
        source_counts=source_counts,
        original=original,
        grayscale=grayscale,
        original_statistics=(
            original_statistics
        ),
        grayscale_statistics=(
            grayscale_statistics
        ),
        text_statistics=text_statistics,
    )

    print("=" * 80)
    print("VALIDACIÓN INDEPENDIENTE DE EMBEDDINGS V2 SUPERADA")
    print("=" * 80)
    print("Imágenes indexadas:", len(image_rows))
    print("Textos únicos:", len(text_rows))
    print("Usos textuales:", len(usage_rows))
    print(
        "Duplicaciones:",
        len(usage_rows) - len(text_rows),
    )
    print(
        "Forma visual original:",
        original.shape,
    )
    print(
        "Forma visual grayscale:",
        grayscale.shape,
    )
    print(
        "Forma textual:",
        text_matrix.shape,
    )
    print(
        "Error máximo de norma original:",
        original_statistics[
            "maximum_norm_error"
        ],
    )
    print(
        "Error máximo de norma grayscale:",
        grayscale_statistics[
            "maximum_norm_error"
        ],
    )
    print(
        "Error máximo de norma textual:",
        text_statistics[
            "maximum_norm_error"
        ],
    )
    print(
        "Coseno medio original–grayscale:",
        summary[
            "visual_ablation_check"
        ]["rowwise_cosine_mean"],
    )
    print("Hashes de entradas: válidos")
    print("Hashes de matrices: válidos")
    print("Hashes de índices: válidos")
    print("Orden de filas: válido")
    print("Generación válida: True")


if __name__ == "__main__":
    main()
