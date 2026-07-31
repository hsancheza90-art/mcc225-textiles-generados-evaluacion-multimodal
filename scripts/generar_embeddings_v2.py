"""Genera los embeddings OpenCLIP reproducibles del experimento v2."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image


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
    """Carga un archivo JSON UTF-8."""

    return json.loads(
        path.read_text(encoding="utf-8")
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


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    """Escribe un CSV determinista UTF-8/LF."""

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
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Escribe un JSON determinista UTF-8/LF."""

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
    """Calcula SHA-256 sobre un texto UTF-8."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def validate_input_text_hash(
    row: dict[str, str],
    text: str,
    context: str,
) -> None:
    """Comprueba el hash textual registrado cuando existe."""

    hash_fields = [
        field
        for field in row
        if field.endswith("_sha256")
    ]

    assert len(hash_fields) <= 1, (
        f"{context}: existen varios campos SHA-256."
    )

    if not hash_fields:
        return

    hash_field = hash_fields[0]
    expected_hash = row[hash_field]
    actual_hash = sha256_text(text)

    assert expected_hash == actual_hash, (
        f"{context}: el hash textual no coincide."
    )


def build_image_index(
    config: dict[str, Any],
) -> tuple[
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    """Construye el índice visual preservando el orden del manifiesto."""

    specification = config[
        "inputs"
    ]["image_manifest"]

    manifest_path = (
        PROJECT_ROOT
        / specification["path"]
    )

    columns, rows = load_csv(
        manifest_path
    )

    required_fields = {
        field
        for field in IMAGE_INDEX_FIELDS
        if field != "image_row_index"
    }

    assert required_fields.issubset(
        set(columns)
    )

    expected_count = specification[
        "expected_count"
    ]

    assert expected_count == 56
    assert len(rows) == expected_count

    image_ids = [
        row[specification["id_field"]]
        for row in rows
    ]

    assert len(image_ids) == len(
        set(image_ids)
    )

    index_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        image_path = (
            PROJECT_ROOT
            / row[specification["path_field"]]
        )

        assert image_path.exists(), image_path
        assert image_path.suffix.lower() == ".png"

        actual_hash = sha256_file(image_path)

        assert actual_hash == row["file_sha256"], (
            f"{row['image_id']}: SHA-256 incorrecto."
        )

        index_record: dict[str, Any] = {
            "image_row_index": row_index,
        }

        for field in IMAGE_INDEX_FIELDS:
            if field == "image_row_index":
                continue

            index_record[field] = row[field]

        index_rows.append(index_record)

    return rows, index_rows


def build_text_contract(
    config: dict[str, Any],
) -> tuple[
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Deduplica textos y construye índices de uso."""

    unique_texts: list[str] = []
    text_to_index: dict[str, int] = {}
    text_index_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    usage_row_index = 0

    for source in config["inputs"][
        "text_sources"
    ]:
        source_id = source["source_id"]

        path = (
            PROJECT_ROOT
            / source["path"]
        )

        columns, rows = load_csv(path)

        record_id_field = source[
            "record_id_field"
        ]

        text_field = source["text_field"]

        assert record_id_field in columns
        assert text_field in columns

        expected_count = source[
            "expected_count"
        ]

        assert len(rows) == expected_count

        source_counts[source_id] = len(rows)

        record_ids = [
            row[record_id_field]
            for row in rows
        ]

        assert len(record_ids) == len(
            set(record_ids)
        )

        for source_row_index, row in enumerate(rows):
            record_id = row[record_id_field]
            text = row[text_field]

            assert record_id
            assert text.strip()
            assert "\ufffd" not in text

            validate_input_text_hash(
                row,
                text,
                (
                    f"{source_id}:"
                    f"{record_id}"
                ),
            )

            text_hash = sha256_text(text)

            if text not in text_to_index:
                text_row_index = len(unique_texts)
                text_id = (
                    f"TXT-{text_row_index + 1:04d}"
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
                        "text_id": text_id,
                        "text_sha256": text_hash,
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

            text_record = text_index_rows[
                text_row_index
            ]

            assert (
                text_record["text_sha256"]
                == text_hash
            )

            text_record["usage_count"] += 1

            usage_rows.append(
                {
                    "usage_row_index": (
                        usage_row_index
                    ),
                    "source_id": source_id,
                    "source_row_index": (
                        source_row_index
                    ),
                    "record_id": record_id,
                    "text_row_index": (
                        text_row_index
                    ),
                    "text_id": (
                        text_record["text_id"]
                    ),
                    "text_sha256": text_hash,
                }
            )

            usage_row_index += 1

    expected_usage_count = config[
        "deduplication"
    ]["expected_usage_count"]

    expected_unique_count = config[
        "deduplication"
    ]["expected_unique_text_count"]

    assert expected_usage_count == 600
    assert len(usage_rows) == 600

    assert expected_unique_count == 494
    assert len(unique_texts) == 494

    assert sum(
        int(row["usage_count"])
        for row in text_index_rows
    ) == 600

    assert len(
        {
            row["text_sha256"]
            for row in text_index_rows
        }
    ) == 494

    return (
        unique_texts,
        text_index_rows,
        usage_rows,
        source_counts,
    )


def load_model(
    config: dict[str, Any],
) -> tuple[Any, Any]:
    """Carga y congela el modelo OpenCLIP."""

    model_config = config["model"]

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            model_config["architecture"],
            pretrained=model_config[
                "pretrained"
            ],
            device="cpu",
        )
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert trainable_parameters == 0

    return model, preprocess


def encode_images(
    model: Any,
    preprocess: Any,
    image_rows: list[dict[str, str]],
    variant_id: str,
    batch_size: int,
) -> np.ndarray:
    """Codifica una variante visual en el orden contractual."""

    assert variant_id in {
        "original",
        "grayscale",
    }

    chunks: list[np.ndarray] = []
    total = len(image_rows)

    print()
    print(
        f"Codificando imágenes: {variant_id}"
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        tensors: list[torch.Tensor] = []

        for row in image_rows[start:end]:
            image_path = (
                PROJECT_ROOT
                / row["image_path"]
            )

            with Image.open(
                image_path
            ) as image:
                image_rgb = image.convert(
                    "RGB"
                )

                if variant_id == "grayscale":
                    image_rgb = (
                        image_rgb
                        .convert("L")
                        .convert("RGB")
                    )

                tensors.append(
                    preprocess(image_rgb)
                )

        batch = torch.stack(tensors)

        with torch.inference_mode():
            features = model.encode_image(
                batch,
                normalize=True,
            )

        chunk = (
            features
            .detach()
            .cpu()
            .float()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        chunks.append(chunk)

        print(
            f"- {end:02d}/{total:02d}"
        )

    return np.concatenate(
        chunks,
        axis=0,
    )


def encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Codifica los textos únicos en el orden contractual."""

    chunks: list[np.ndarray] = []
    total = len(texts)

    print()
    print("Codificando textos únicos")

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        tokens = tokenizer(
            texts[start:end]
        )

        with torch.inference_mode():
            features = model.encode_text(
                tokens,
                normalize=True,
            )

        chunk = (
            features
            .detach()
            .cpu()
            .float()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        chunks.append(chunk)

        print(
            f"- {end:03d}/{total:03d}"
        )

    return np.concatenate(
        chunks,
        axis=0,
    )


def matrix_statistics(
    matrix: np.ndarray,
    expected_shape: tuple[int, int],
    tolerance: float,
) -> dict[str, Any]:
    """Valida una matriz y devuelve estadísticas reproducibles."""

    assert matrix.shape == expected_shape
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    maximum_norm_error = float(
        np.max(np.abs(norms - 1.0))
    )

    assert maximum_norm_error <= tolerance

    return {
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


def temporary_output_paths(
    config: dict[str, Any],
    temporary_directory: Path,
) -> dict[str, Path]:
    """Resuelve las salidas dentro del directorio temporal."""

    paths: dict[str, Path] = {}

    for key, value in config[
        "outputs"
    ].items():
        if key == "directory":
            continue

        paths[key] = (
            temporary_directory
            / Path(value).name
        )

    return paths


def replace_output_directory(
    temporary_directory: Path,
    output_directory: Path,
) -> None:
    """Publica el directorio temporal conservando recuperación."""

    backup_directory = (
        output_directory.with_name(
            output_directory.name
            + ".previous"
        )
    )

    if backup_directory.exists():
        shutil.rmtree(backup_directory)

    if output_directory.exists():
        output_directory.replace(
            backup_directory
        )

    try:
        temporary_directory.replace(
            output_directory
        )
    except Exception:
        if (
            backup_directory.exists()
            and not output_directory.exists()
        ):
            backup_directory.replace(
                output_directory
            )

        raise

    if backup_directory.exists():
        shutil.rmtree(backup_directory)


def main() -> None:
    """Genera matrices, índices y resumen."""

    config = load_json(CONFIG_PATH)
    experiment = load_json(
        EXPERIMENT_PATH
    )
    environment = load_json(
        ENVIRONMENT_PATH
    )

    reproducibility = config[
        "reproducibility"
    ]

    seed = reproducibility["seed"]
    thread_count = reproducibility[
        "torch_num_threads"
    ]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(thread_count)

    assert (
        reproducibility[
            "canonical_device"
        ]
        == "cpu"
    )

    assert (
        environment[
            "environment_valid"
        ]
        is True
    )

    assert (
        config["model"]["architecture"]
        == experiment["model"][
            "architecture"
        ]
    )

    assert (
        config["model"]["pretrained"]
        == experiment["model"][
            "pretrained"
        ]
    )

    print("=" * 80)
    print("GENERACIÓN DE EMBEDDINGS OPENCLIP V2")
    print("=" * 80)
    print(
        "Modelo:",
        config["model"]["architecture"],
    )
    print(
        "Pesos:",
        config["model"]["pretrained"],
    )
    print("Dispositivo: cpu")
    print("Hilos:", torch.get_num_threads())
    print(
        "Batch visual:",
        reproducibility[
            "image_batch_size"
        ],
    )
    print(
        "Batch textual:",
        reproducibility[
            "text_batch_size"
        ],
    )

    (
        image_rows,
        image_index_rows,
    ) = build_image_index(config)

    (
        unique_texts,
        text_index_rows,
        usage_rows,
        source_counts,
    ) = build_text_contract(config)

    print()
    print("Imágenes:", len(image_rows))
    print("Textos únicos:", len(unique_texts))
    print("Usos textuales:", len(usage_rows))

    model, preprocess = load_model(
        config
    )

    tokenizer = open_clip.get_tokenizer(
        config["model"]["architecture"]
    )

    image_original = encode_images(
        model=model,
        preprocess=preprocess,
        image_rows=image_rows,
        variant_id="original",
        batch_size=reproducibility[
            "image_batch_size"
        ],
    )

    image_grayscale = encode_images(
        model=model,
        preprocess=preprocess,
        image_rows=image_rows,
        variant_id="grayscale",
        batch_size=reproducibility[
            "image_batch_size"
        ],
    )

    text_unique = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=unique_texts,
        batch_size=reproducibility[
            "text_batch_size"
        ],
    )

    expected_shapes = config[
        "expected_shapes"
    ]

    tolerance = config[
        "validation"
    ]["norm_absolute_tolerance"]

    original_statistics = (
        matrix_statistics(
            image_original,
            tuple(
                expected_shapes[
                    "image_original"
                ]
            ),
            tolerance,
        )
    )

    grayscale_statistics = (
        matrix_statistics(
            image_grayscale,
            tuple(
                expected_shapes[
                    "image_grayscale"
                ]
            ),
            tolerance,
        )
    )

    text_statistics = matrix_statistics(
        text_unique,
        tuple(
            expected_shapes["text_unique"]
        ),
        tolerance,
    )

    assert not np.array_equal(
        image_original,
        image_grayscale,
    )

    visual_variant_cosines = np.sum(
        image_original * image_grayscale,
        axis=1,
    )

    output_directory = (
        PROJECT_ROOT
        / config["outputs"]["directory"]
    )

    temporary_directory = (
        output_directory.with_name(
            output_directory.name
            + ".tmp"
        )
    )

    if temporary_directory.exists():
        shutil.rmtree(
            temporary_directory
        )

    temporary_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    paths = temporary_output_paths(
        config,
        temporary_directory,
    )

    try:
        np.save(
            paths["image_original_npy"],
            image_original,
            allow_pickle=False,
        )

        np.save(
            paths["image_grayscale_npy"],
            image_grayscale,
            allow_pickle=False,
        )

        np.save(
            paths["text_unique_npy"],
            text_unique,
            allow_pickle=False,
        )

        write_csv(
            paths["image_index_csv"],
            IMAGE_INDEX_FIELDS,
            image_index_rows,
        )

        write_csv(
            paths["text_index_csv"],
            TEXT_INDEX_FIELDS,
            text_index_rows,
        )

        write_csv(
            paths["text_usage_csv"],
            TEXT_USAGE_FIELDS,
            usage_rows,
        )

        matrix_records = {
            "image_original": {
                "path": config["outputs"][
                    "image_original_npy"
                ],
                **original_statistics,
                "sha256": sha256_file(
                    paths[
                        "image_original_npy"
                    ]
                ),
            },
            "image_grayscale": {
                "path": config["outputs"][
                    "image_grayscale_npy"
                ],
                **grayscale_statistics,
                "sha256": sha256_file(
                    paths[
                        "image_grayscale_npy"
                    ]
                ),
            },
            "text_unique": {
                "path": config["outputs"][
                    "text_unique_npy"
                ],
                **text_statistics,
                "sha256": sha256_file(
                    paths["text_unique_npy"]
                ),
            },
        }

        index_records = {
            "images": {
                "path": config["outputs"][
                    "image_index_csv"
                ],
                "rows": len(
                    image_index_rows
                ),
                "sha256": sha256_file(
                    paths["image_index_csv"]
                ),
            },
            "unique_texts": {
                "path": config["outputs"][
                    "text_index_csv"
                ],
                "rows": len(
                    text_index_rows
                ),
                "sha256": sha256_file(
                    paths["text_index_csv"]
                ),
            },
            "text_usages": {
                "path": config["outputs"][
                    "text_usage_csv"
                ],
                "rows": len(usage_rows),
                "sha256": sha256_file(
                    paths["text_usage_csv"]
                ),
            },
        }

        input_artifacts = {
            "embeddings_config": {
                "path": (
                    CONFIG_PATH
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "sha256": sha256_file(
                    CONFIG_PATH
                ),
            },
            "experiment_config": {
                "path": (
                    EXPERIMENT_PATH
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "sha256": sha256_file(
                    EXPERIMENT_PATH
                ),
            },
            "environment_audit": {
                "path": (
                    ENVIRONMENT_PATH
                    .relative_to(
                        PROJECT_ROOT
                    )
                    .as_posix()
                ),
                "sha256": sha256_file(
                    ENVIRONMENT_PATH
                ),
            },
            "image_manifest": {
                "path": config["inputs"][
                    "image_manifest"
                ]["path"],
                "records": len(image_rows),
                "sha256": sha256_file(
                    PROJECT_ROOT
                    / config["inputs"][
                        "image_manifest"
                    ]["path"]
                ),
            },
            "text_sources": {
                source["source_id"]: {
                    "path": source["path"],
                    "records": (
                        source_counts[
                            source["source_id"]
                        ]
                    ),
                    "sha256": sha256_file(
                        PROJECT_ROOT
                        / source["path"]
                    ),
                }
                for source in config[
                    "inputs"
                ]["text_sources"]
            },
        }

        summary = {
            "schema_version": "1.0",
            "dataset_version": "v2",
            "artifact_version": (
                config["artifact_version"]
            ),
            "stage": (
                "openclip_embeddings_generation"
            ),
            "model": {
                **config["model"],
                "total_parameters": sum(
                    parameter.numel()
                    for parameter
                    in model.parameters()
                ),
                "trainable_parameters": 0,
            },
            "runtime": {
                "python": (
                    platform.python_version()
                ),
                "torch": torch.__version__,
                "open_clip": getattr(
                    open_clip,
                    "__version__",
                    "unknown",
                ),
                "numpy": np.__version__,
                "device": "cpu",
                "torch_num_threads": (
                    torch.get_num_threads()
                ),
            },
            "reproducibility": (
                reproducibility
            ),
            "input_artifacts": (
                input_artifacts
            ),
            "counts": {
                "images": len(image_rows),
                "image_variants": 2,
                "text_usages": len(
                    usage_rows
                ),
                "unique_texts": len(
                    unique_texts
                ),
                "deduplicated_usages": (
                    len(usage_rows)
                    - len(unique_texts)
                ),
                "text_sources": (
                    source_counts
                ),
            },
            "matrices": matrix_records,
            "indices": index_records,
            "visual_ablation_check": {
                "original_and_grayscale_differ": (
                    True
                ),
                "rowwise_cosine_min": float(
                    visual_variant_cosines.min()
                ),
                "rowwise_cosine_max": float(
                    visual_variant_cosines.max()
                ),
                "rowwise_cosine_mean": float(
                    visual_variant_cosines.mean()
                ),
            },
            "row_order": {
                "images": (
                    "manifest_imagenes_v2.csv "
                    "row order"
                ),
                "texts": (
                    "first exact occurrence using "
                    "source_order and csv_row_order"
                ),
                "text_usages": (
                    "source_order followed by "
                    "csv_row_order"
                ),
            },
            "generation_valid": True,
        }

        write_json(
            paths["summary_json"],
            summary,
        )

        expected_names = {
            Path(value).name
            for key, value
            in config["outputs"].items()
            if key != "directory"
        }

        actual_names = {
            path.name
            for path
            in temporary_directory.iterdir()
            if path.is_file()
        }

        assert actual_names == expected_names

        replace_output_directory(
            temporary_directory,
            output_directory,
        )

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory
            )

        raise

    print()
    print("=" * 80)
    print("EMBEDDINGS V2 GENERADOS CORRECTAMENTE")
    print("=" * 80)
    print(
        "Imagen original:",
        image_original.shape,
        image_original.dtype,
    )
    print(
        "Imagen grayscale:",
        image_grayscale.shape,
        image_grayscale.dtype,
    )
    print(
        "Textos únicos:",
        text_unique.shape,
        text_unique.dtype,
    )
    print(
        "Máximo error de norma visual:",
        original_statistics[
            "maximum_norm_error"
        ],
    )
    print(
        "Máximo error de norma grayscale:",
        grayscale_statistics[
            "maximum_norm_error"
        ],
    )
    print(
        "Máximo error de norma textual:",
        text_statistics[
            "maximum_norm_error"
        ],
    )
    print(
        "Coseno original–grayscale medio:",
        float(
            visual_variant_cosines.mean()
        ),
    )
    print(
        "Directorio:",
        config["outputs"]["directory"],
    )
    print("Generación válida: True")


if __name__ == "__main__":
    main()
