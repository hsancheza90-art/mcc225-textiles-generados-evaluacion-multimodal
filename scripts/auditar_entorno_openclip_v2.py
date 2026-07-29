"""Audita y registra el entorno CPU de OpenCLIP para el dataset v2."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import open_clip
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "config"
    / "experimento_v2.json"
)

REQUIREMENTS_PATH = (
    PROJECT_ROOT
    / "requirements.txt"
)

ENVIRONMENT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "entorno_cpu_v2.json"
)

PIP_FREEZE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "entorno_cpu_pip_freeze.txt"
)

TEXT_SOURCES = (
    (
        "positivos",
        PROJECT_ROOT
        / "data"
        / "v2"
        / "captions_positivos_v2.csv",
        "caption_text",
        280,
    ),
    (
        "sin_color",
        PROJECT_ROOT
        / "data"
        / "v2"
        / "captions_sin_color_v2.csv",
        "caption_text",
        40,
    ),
    (
        "negativos_dificiles",
        PROJECT_ROOT
        / "data"
        / "v2"
        / "candidatos_negativos_dificiles_v2.csv",
        "candidate_text",
        280,
    ),
)

EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "matplotlib": "3.9.2",
    "Pillow": "10.4.0",
    "torch": "2.13.0+cpu",
    "torchvision": "0.28.0+cpu",
    "open-clip-torch": "3.3.0",
    "scikit-learn": "1.5.1",
    "tqdm": "4.66.5",
    "ipykernel": "6.29.5",
}


EXPECTED_REQUIREMENT_LINES = (
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "matplotlib==3.9.2",
    "Pillow==10.4.0",
    "torch==2.13.0+cpu",
    "torchvision==0.28.0+cpu",
    "open-clip-torch==3.3.0",
    "scikit-learn==1.5.1",
    "tqdm==4.66.5",
    "ipykernel==6.29.5",
)


def validate_requirements(
    path: Path,
) -> list[str]:
    """Valida las dependencias directas congeladas."""

    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if (
            line.strip()
            and not line.lstrip().startswith("#")
        )
    ]

    if tuple(lines) != EXPECTED_REQUIREMENT_LINES:
        raise AssertionError(
            "requirements.txt no coincide con el "
            "protocolo congelado. "
            f"Esperado: {EXPECTED_REQUIREMENT_LINES}. "
            f"Actual: {tuple(lines)}."
        )

    return lines


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


def write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )


def package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in EXPECTED_VERSIONS
    }


def build_pip_freeze() -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "freeze",
            "--all",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    lines = sorted(
        (
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        ),
        key=str.casefold,
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    experiment = load_json(EXPERIMENT_PATH)

    requirement_lines = validate_requirements(
        REQUIREMENTS_PATH
    )

    versions = package_versions()

    assert versions == EXPECTED_VERSIONS, (
        "Las versiones instaladas no coinciden "
        "con el protocolo CPU fijado."
    )

    assert sys.version_info[:2] == (3, 11)
    assert sys.prefix != sys.base_prefix
    assert torch.__version__ == "2.13.0+cpu"
    assert torch.version.cuda is None
    assert torch.cuda.is_available() is False
    assert torch.cuda.device_count() == 0

    model_config = experiment["model"]

    assert model_config["library"] == "open_clip"
    assert model_config["architecture"] == "ViT-B-32"
    assert (
        model_config["pretrained"]
        == "laion2b_s34b_b79k"
    )

    texts: list[str] = []
    source_labels: list[str] = []
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}

    for (
        source_name,
        path,
        text_field,
        expected_count,
    ) in TEXT_SOURCES:
        rows = load_csv(path)

        assert len(rows) == expected_count

        source_texts = [
            row[text_field].strip()
            for row in rows
        ]

        assert all(source_texts)

        texts.extend(source_texts)
        source_labels.extend(
            [source_name] * len(source_texts)
        )

        source_paths[source_name] = (
            path.relative_to(PROJECT_ROOT).as_posix()
        )

        source_hashes[source_name] = (
            sha256_file(path)
        )

    assert len(texts) == 600
    assert len(set(texts)) == 494

    tokenizer = open_clip.get_tokenizer(
        model_config["architecture"]
    )

    tokens = tokenizer(texts)

    assert isinstance(tokens, torch.Tensor)
    assert tuple(tokens.shape) == (600, 77)

    nonzero_counts = (
        tokens != 0
    ).sum(dim=1)

    source_token_counts: dict[
        str,
        list[int],
    ] = {
        source_name: []
        for source_name, *_ in TEXT_SOURCES
    }

    for source_name, count in zip(
        source_labels,
        nonzero_counts.tolist(),
        strict=True,
    ):
        source_token_counts[source_name].append(
            int(count)
        )

    fully_occupied = int(
        (
            nonzero_counts
            == tokens.shape[1]
        )
        .sum()
        .item()
    )

    assert fully_occupied == 0

    pip_freeze = build_pip_freeze()

    write_text(
        PIP_FREEZE_OUTPUT_PATH,
        pip_freeze,
    )

    tokenizer_by_source = {
        source_name: {
            "count": len(counts),
            "minimum_nonzero_tokens": min(counts),
            "maximum_nonzero_tokens": max(counts),
        }
        for source_name, counts
        in source_token_counts.items()
    }

    payload = {
        "schema_version": "1.0",
        "dataset_version": "v2",
        "stage": "cpu_environment_audit",
        "python": {
            "version": platform.python_version(),
            "implementation": (
                platform.python_implementation()
            ),
            "executable": str(
                Path(sys.executable).resolve()
            ),
            "virtual_environment": (
                sys.prefix != sys.base_prefix
            ),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": (
                platform.processor()
                or "no informado"
            ),
            "logical_cpu_count": os.cpu_count(),
        },
        "runtime": {
            "canonical_device": "cpu",
            "torch_version": torch.__version__,
            "torchvision_version": versions[
                "torchvision"
            ],
            "cuda_compiled": torch.version.cuda,
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "cuda_device_count": (
                torch.cuda.device_count()
            ),
            "torch_num_threads": (
                torch.get_num_threads()
            ),
        },
        "model": {
            "library": model_config["library"],
            "architecture": (
                model_config["architecture"]
            ),
            "pretrained": (
                model_config["pretrained"]
            ),
            "embedding_dimension": 512,
            "context_length": 77,
        },
        "packages": versions,
        "tokenizer_audit": {
            "text_count": len(texts),
            "unique_text_count": len(set(texts)),
            "duplicate_occurrence_count": (
                len(texts) - len(set(texts))
            ),
            "token_tensor_shape": list(
                tokens.shape
            ),
            "fully_occupied_sequence_count": (
                fully_occupied
            ),
            "by_source": tokenizer_by_source,
            "source_counts": dict(
                Counter(source_labels)
            ),
            "truncation_evidence": False,
        },
        "inputs": {
            "experiment_config_path": (
                EXPERIMENT_PATH
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "experiment_config_sha256": (
                sha256_file(EXPERIMENT_PATH)
            ),
            "requirements_path": (
                REQUIREMENTS_PATH
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "requirements_sha256": (
                sha256_file(REQUIREMENTS_PATH)
            ),
            "requirements_validated": True,
            "requirements_line_count": len(
                requirement_lines
            ),
            "direct_dependency_count": (
                len(requirement_lines) - 1
            ),
            "text_source_paths": source_paths,
            "text_source_sha256": source_hashes,
        },
        "pip_freeze": {
            "path": (
                PIP_FREEZE_OUTPUT_PATH
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "sha256": sha256_file(
                PIP_FREEZE_OUTPUT_PATH
            ),
            "package_line_count": len(
                pip_freeze.splitlines()
            ),
        },
        "environment_valid": True,
    }

    write_json(
        ENVIRONMENT_OUTPUT_PATH,
        payload,
    )

    print("=" * 76)
    print("AUDITORÍA DEL ENTORNO CPU V2 SUPERADA")
    print("=" * 76)
    print(
        "Python:",
        payload["python"]["version"],
    )
    print(
        "Entorno virtual:",
        payload["python"][
            "virtual_environment"
        ],
    )
    print(
        "PyTorch:",
        payload["runtime"]["torch_version"],
    )
    print(
        "OpenCLIP:",
        payload["packages"][
            "open-clip-torch"
        ],
    )
    print(
        "Dispositivo canónico:",
        payload["runtime"][
            "canonical_device"
        ],
    )
    print(
        "Textos auditados:",
        payload["tokenizer_audit"][
            "text_count"
        ],
    )
    print(
        "Textos únicos:",
        payload["tokenizer_audit"][
            "unique_text_count"
        ],
    )
    print(
        "Secuencias completamente ocupadas:",
        payload["tokenizer_audit"][
            "fully_occupied_sequence_count"
        ],
    )
    print(
        "Líneas de pip freeze:",
        payload["pip_freeze"][
            "package_line_count"
        ],
    )
    print(
        "Requirements válidos:",
        payload["inputs"][
            "requirements_validated"
        ],
    )
    print(
        "Dependencias directas:",
        payload["inputs"][
            "direct_dependency_count"
        ],
    )
    print(
        "Entorno válido:",
        payload["environment_valid"],
    )

    print("\nArtefactos:")
    print(
        "- "
        + ENVIRONMENT_OUTPUT_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )
    print(
        "- "
        + PIP_FREEZE_OUTPUT_PATH
        .relative_to(PROJECT_ROOT)
        .as_posix()
    )


if __name__ == "__main__":
    main()