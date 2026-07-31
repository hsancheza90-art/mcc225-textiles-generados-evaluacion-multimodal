"""Genera el manifiesto integral y reproducible de la entrega v2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTRACT_PATH = (
    PROJECT_ROOT
    / "config"
    / "entrega_final_v2.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "manifiesto_entrega_v2.json"
)

UTF8_BOM = b"\xef\xbb\xbf"


def run_git(
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def load_json(
    path: Path,
) -> dict[str, Any]:
    raw = path.read_bytes()

    assert not raw.startswith(UTF8_BOM), (
        f"{path}: contiene BOM."
    )

    assert b"\r" not in raw, (
        f"{path}: no utiliza exclusivamente LF."
    )

    assert raw.endswith(b"\n"), (
        f"{path}: no termina con salto de línea."
    )

    value = json.loads(
        raw.decode("utf-8")
    )

    assert isinstance(value, dict)

    return value


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    return sha256_bytes(
        path.read_bytes()
    )


def git_blob_worktree(
    relative_path: str,
) -> str:
    process = run_git(
        [
            "hash-object",
            f"--path={relative_path}",
            "--",
            relative_path,
        ]
    )

    value = process.stdout.strip()

    assert len(value) == 40
    assert all(
        character in "0123456789abcdef"
        for character in value
    )

    return value


def git_blob_index(
    relative_path: str,
) -> str | None:
    process = run_git(
        [
            "rev-parse",
            f":{relative_path}",
        ],
        check=False,
    )

    if process.returncode != 0:
        return None

    value = process.stdout.strip()

    assert len(value) == 40
    assert all(
        character in "0123456789abcdef"
        for character in value
    )

    return value


def normalize_tracked_text(
    data: bytes,
) -> str:
    text = data.decode("utf-8-sig")

    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def inspect_content(
    data: bytes,
) -> dict[str, Any]:
    bom = data.startswith(UTF8_BOM)

    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {
            "encoding": "binary",
            "bom": False,
            "line_endings": "not_applicable",
        }

    without_crlf = data.replace(
        b"\r\n",
        b"",
    )

    crlf_count = data.count(b"\r\n")
    lf_count = without_crlf.count(b"\n")
    cr_count = without_crlf.count(b"\r")

    active_styles = sum(
        count > 0
        for count in (
            crlf_count,
            lf_count,
            cr_count,
        )
    )

    if active_styles > 1:
        line_endings = "mixed"

    elif crlf_count > 0:
        line_endings = "crlf"

    elif lf_count > 0:
        line_endings = "lf"

    elif cr_count > 0:
        line_endings = "cr"

    else:
        line_endings = "none"

    return {
        "encoding": (
            "utf-8-sig"
            if bom
            else "utf-8"
        ),
        "bom": bom,
        "line_endings": line_endings,
    }


def classify_family(
    relative_path: str,
    rules: list[dict[str, str]],
) -> str:
    matching_rules = [
        rule
        for rule in rules
        if relative_path.startswith(
            rule["prefix"]
        )
    ]

    if not matching_rules:
        return "root"

    selected = max(
        matching_rules,
        key=lambda rule: len(
            rule["prefix"]
        ),
    )

    return selected["family"]


def inventory_digest(
    inventory: list[dict[str, Any]],
) -> str:
    canonical = json.dumps(
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return sha256_bytes(canonical)


def directory_inventory(
    relative_paths: list[str],
) -> list[dict[str, Any]]:
    rows = []

    for relative_path in sorted(
        relative_paths
    ):
        path = PROJECT_ROOT / relative_path

        assert path.exists(), relative_path
        assert path.is_file(), relative_path

        rows.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return rows


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    serialized = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    assert "\r" not in serialized

    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    assert not temporary_path.exists(), (
        f"Existe un temporal residual: "
        f"{temporary_path}"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        temporary_path.write_bytes(
            serialized.encode("utf-8")
        )

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def current_git_paths() -> dict[str, list[str]]:
    unstaged = sorted(
        line.strip()
        for line in run_git(
            [
                "diff",
                "--name-only",
            ]
        ).stdout.splitlines()
        if line.strip()
    )

    staged = sorted(
        line.strip()
        for line in run_git(
            [
                "diff",
                "--cached",
                "--name-only",
            ]
        ).stdout.splitlines()
        if line.strip()
    )

    untracked = sorted(
        line.strip()
        for line in run_git(
            [
                "ls-files",
                "--others",
                "--exclude-standard",
            ]
        ).stdout.splitlines()
        if line.strip()
    )

    return {
        "unstaged": unstaged,
        "staged": staged,
        "untracked": untracked,
    }


def main() -> None:
    contract = load_json(
        CONTRACT_PATH
    )

    assert contract["schema_version"] == 1
    assert contract["contract_id"] == "DELIVERY-V2"
    assert contract["dataset_version"] == "v2"
    assert (
        contract["stage"]
        == "final_delivery_freeze"
    )

    delivery_paths = contract[
        "delivery_paths"
    ]

    manifest_relative_path = (
        delivery_paths["manifest"]
    )

    assert (
        OUTPUT_PATH
        == PROJECT_ROOT
        / manifest_relative_path
    )

    manifest_input_mode = contract[
        "inventory_contract"
    ]["input_mode"]

    if (
        manifest_input_mode
        == "git_worktree_tracked_revision"
    ):
        assert OUTPUT_PATH.exists(), (
            "La revisión requiere un manifiesto "
            "rastreado preexistente."
        )
        assert OUTPUT_PATH.is_file(), (
            "La ruta del manifiesto preexistente "
            "no es un archivo."
        )

        manifest_tracking = run_git(
            [
                "ls-files",
                "--error-unmatch",
                manifest_relative_path,
            ],
            check=False,
        )

        assert manifest_tracking.returncode == 0, (
            "El manifiesto de la revisión debe "
            "estar rastreado por Git."
        )
        assert (
            manifest_tracking.stdout.strip()
            == manifest_relative_path
        )

    else:
        assert not OUTPUT_PATH.exists(), (
            "El manifiesto ya existe. "
            "No se sobrescribirá."
        )

    expected_before_generation = sorted(
        path
        for path in delivery_paths[
            "allowed_final_worktree_paths"
        ]
        if path != manifest_relative_path
    )

    git_paths = current_git_paths()

    assert git_paths["staged"] == [], (
        "No debe haber cambios preparados."
    )

    actual_changed_paths = sorted(
        {
            *git_paths["unstaged"],
            *git_paths["untracked"],
        }
    )

    assert (
        actual_changed_paths
        == expected_before_generation
    ), (
        "El conjunto previo de cambios no es "
        "el esperado. "
        f"Actual={actual_changed_paths}, "
        f"esperado={expected_before_generation}."
    )

    git_contract = contract[
        "git_contract"
    ]

    branch = run_git(
        [
            "rev-parse",
            "--abbrev-ref",
            "HEAD",
        ]
    ).stdout.strip()

    upstream = run_git(
        [
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{u}",
        ]
    ).stdout.strip()

    origin_url = run_git(
        [
            "remote",
            "get-url",
            "origin",
        ]
    ).stdout.strip()

    source_head_commit = run_git(
        [
            "rev-parse",
            "HEAD",
        ]
    ).stdout.strip()

    source_head_short = run_git(
        [
            "rev-parse",
            "--short=12",
            "HEAD",
        ]
    ).stdout.strip()

    source_head_tree = run_git(
        [
            "rev-parse",
            "HEAD^{tree}",
        ]
    ).stdout.strip()

    source_head_commit_time = run_git(
        [
            "show",
            "-s",
            "--format=%cI",
            "HEAD",
        ]
    ).stdout.strip()

    assert branch == git_contract["branch"]
    assert upstream == git_contract["upstream"]
    assert origin_url == git_contract["origin_url"]

    release_tag = git_contract[
        "expected_release_tag"
    ]

    assert not run_git(
        [
            "tag",
            "--list",
            release_tag,
        ]
    ).stdout.strip(), (
        f"La etiqueta ya existe: {release_tag}"
    )

    tracked_paths = sorted(
        line.strip()
        for line in run_git(
            [
                "ls-files",
            ]
        ).stdout.splitlines()
        if line.strip()
    )

    expected_tracked_files = contract[
        "inventory_contract"
    ]["expected_final_tracked_files"]

    assert len(tracked_paths) == (
        expected_tracked_files
    )
    assert len(set(tracked_paths)) == (
        expected_tracked_files
    )

    delivery_source_paths = delivery_paths[
        "delivery_source_paths"
    ]

    all_inventory_paths = sorted(
        {
            *tracked_paths,
            *delivery_source_paths,
        }
        - {manifest_relative_path}
    )

    assert manifest_relative_path not in (
        all_inventory_paths
    )

    inventory_contract = contract[
        "inventory_contract"
    ]

    assert (
        len(all_inventory_paths)
        == inventory_contract[
            "expected_inventory_entries"
        ]
    )

    tracked_set = set(tracked_paths)
    unstaged_set = set(
        git_paths["unstaged"]
    )
    untracked_set = set(
        git_paths["untracked"]
    )

    family_rules = contract[
        "artifact_family_rules"
    ]

    inventory: list[
        dict[str, Any]
    ] = []

    for relative_path in all_inventory_paths:
        path = PROJECT_ROOT / relative_path

        assert path.exists(), relative_path
        assert path.is_file(), relative_path

        data = path.read_bytes()

        worktree_blob = git_blob_worktree(
            relative_path
        )

        index_blob = (
            git_blob_index(relative_path)
            if relative_path in tracked_set
            else None
        )

        if relative_path in untracked_set:
            git_state = "untracked_delivery"

        elif relative_path in unstaged_set:
            git_state = "tracked_modified"

        else:
            git_state = "tracked_clean"

        if git_state == "tracked_clean":
            assert index_blob == worktree_blob

        elif git_state == "tracked_modified":
            assert index_blob is not None
            assert index_blob != worktree_blob

        else:
            assert index_blob is None

        content = inspect_content(data)

        inventory.append(
            {
                "path": relative_path,
                "family": classify_family(
                    relative_path,
                    family_rules,
                ),
                "size_bytes": len(data),
                "sha256": sha256_bytes(data),
                "git_blob_worktree": worktree_blob,
                "git_blob_index": index_blob,
                "git_state": git_state,
                "encoding": content[
                    "encoding"
                ],
                "bom": content["bom"],
                "line_endings": content[
                    "line_endings"
                ],
            }
        )

    assert [
        row["path"]
        for row in inventory
    ] == sorted(
        row["path"]
        for row in inventory
    )

    assert len(
        {
            row["path"]
            for row in inventory
        }
    ) == len(inventory)

    required_fields = set(
        inventory_contract[
            "required_fields"
        ]
    )

    for row in inventory:
        assert set(row) == required_fields

    state_counts = Counter(
        row["git_state"]
        for row in inventory
    )

    inventory_path_set = set(
        all_inventory_paths
    )
    expected_tracked_modified = len(
        unstaged_set & inventory_path_set
    )
    expected_untracked_delivery = len(
        untracked_set & inventory_path_set
    )
    expected_tracked_clean = (
        len(all_inventory_paths)
        - expected_tracked_modified
        - expected_untracked_delivery
    )

    assert state_counts == Counter(
        {
            "tracked_clean": (
                expected_tracked_clean
            ),
            "tracked_modified": (
                expected_tracked_modified
            ),
            "untracked_delivery": (
                expected_untracked_delivery
            ),
        }
    )

    family_counts = Counter(
        row["family"]
        for row in inventory
    )

    family_sizes = Counter()

    for row in inventory:
        family_sizes[
            row["family"]
        ] += row["size_bytes"]

    artifact_families = [
        {
            "family": family,
            "file_count": (
                family_counts[family]
            ),
            "size_bytes": (
                family_sizes[family]
            ),
        }
        for family in sorted(
            family_counts
        )
    ]

    historical_results_paths = [
        relative_path
        for relative_path in tracked_paths
        if (
            relative_path.startswith(
                "results/"
            )
            and not relative_path.startswith(
                "results/v2/"
            )
        )
    ]

    historical_figures_paths = [
        relative_path
        for relative_path in tracked_paths
        if relative_path.startswith(
            "figures/"
        )
    ]

    historical_results_inventory = (
        directory_inventory(
            historical_results_paths
        )
    )

    historical_figures_inventory = (
        directory_inventory(
            historical_figures_paths
        )
    )

    conclusion_relative_path = (
        "results/conclusion_tecnica.md"
    )

    conclusion_worktree = (
        PROJECT_ROOT
        / conclusion_relative_path
    ).read_bytes()

    conclusion_index = run_git(
        [
            "show",
            f":{conclusion_relative_path}",
        ]
    ).stdout.encode("utf-8")

    conclusion_matches_index = (
        normalize_tracked_text(
            conclusion_worktree
        )
        == normalize_tracked_text(
            conclusion_index
        )
    )

    assert conclusion_matches_index

    environment_contract = contract[
        "environment_contract"
    ]

    environment_path = (
        PROJECT_ROOT
        / environment_contract[
            "source"
        ]
    )

    environment = load_json(
        environment_path
    )

    assert (
        environment["python"]["version"]
        == environment_contract["python"]
    )

    assert (
        environment["runtime"][
            "torch_version"
        ]
        == environment_contract["torch"]
    )

    assert (
        environment["runtime"][
            "torchvision_version"
        ]
        == environment_contract[
            "torchvision"
        ]
    )

    assert (
        environment["packages"][
            "open-clip-torch"
        ]
        == environment_contract[
            "open_clip"
        ]
    )

    assert (
        environment["runtime"][
            "canonical_device"
        ]
        == environment_contract[
            "canonical_device"
        ]
    )

    assert (
        environment["model"][
            "architecture"
        ]
        == environment_contract[
            "model_architecture"
        ]
    )

    assert (
        environment["model"][
            "pretrained"
        ]
        == environment_contract[
            "pretrained"
        ]
    )

    assert (
        environment["model"][
            "embedding_dimension"
        ]
        == environment_contract[
            "embedding_dimension"
        ]
    )

    validation_contract = contract[
        "validation_contract"
    ]

    validators = validation_contract[
        "validators"
    ]

    assert len(validators) == 25

    validator_static_records = []

    forbidden_import_roots = {
        "open_clip",
        "torch",
        "torchvision",
        "requests",
        "urllib3",
    }

    for validator in validators:
        path = PROJECT_ROOT / validator

        assert path.exists(), validator
        assert path.is_file(), validator

        source = path.read_text(
            encoding="utf-8-sig"
        )

        compile(
            source,
            validator,
            "exec",
            dont_inherit=True,
            optimize=0,
        )

        imported_roots = set()

        import ast

        tree = ast.parse(
            source,
            filename=validator,
        )

        main_count = 0

        for node in tree.body:
            if (
                isinstance(
                    node,
                    ast.FunctionDef,
                )
                and node.name == "main"
            ):
                main_count += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_roots.add(
                        alias.name.split(
                            ".",
                            maxsplit=1,
                        )[0]
                    )

            elif isinstance(
                node,
                ast.ImportFrom,
            ):
                if node.module:
                    imported_roots.add(
                        node.module.split(
                            ".",
                            maxsplit=1,
                        )[0]
                    )

        forbidden_imports = sorted(
            imported_roots
            & forbidden_import_roots
        )

        assert main_count == 1
        assert not forbidden_imports, (
            f"{validator}: "
            f"{forbidden_imports}"
        )

        validator_static_records.append(
            {
                "path": validator,
                "main_count": main_count,
                "syntax_valid": True,
                "forbidden_imports": (
                    forbidden_imports
                ),
            }
        )

    inventory_sha256 = inventory_digest(
        inventory
    )

    required_sections = contract[
        "required_manifest_sections"
    ]

    payload = {
        "schema_version": 1,
        "contract_id": "DELIVERY-V2-MANIFEST",
        "dataset_version": "v2",
        "stage": "final_delivery_manifest",
        "generated_at": (
            source_head_commit_time
        ),
        "project": contract["project"],
        "git": {
            "branch": branch,
            "upstream": upstream,
            "origin_url": origin_url,
            "source_head_commit": (
                source_head_commit
            ),
            "source_head_short": (
                source_head_short
            ),
            "source_head_tree": (
                source_head_tree
            ),
            "source_head_commit_time": (
                source_head_commit_time
            ),
            "generation_time_basis": (
                "source_head_commit_time"
            ),
            "expected_release_tag": (
                release_tag
            ),
            "staged_paths": (
                git_paths["staged"]
            ),
            "unstaged_paths": (
                git_paths["unstaged"]
            ),
            "untracked_paths_before_manifest": (
                git_paths["untracked"]
            ),
        },
        "environment": {
            "source": (
                environment_contract[
                    "source"
                ]
            ),
            "source_sha256": sha256_file(
                environment_path
            ),
            "python": (
                environment["python"][
                    "version"
                ]
            ),
            "torch": (
                environment["runtime"][
                    "torch_version"
                ]
            ),
            "torchvision": (
                environment["runtime"][
                    "torchvision_version"
                ]
            ),
            "open_clip": (
                environment["packages"][
                    "open-clip-torch"
                ]
            ),
            "device": (
                environment["runtime"][
                    "canonical_device"
                ]
            ),
            "model_architecture": (
                environment["model"][
                    "architecture"
                ]
            ),
            "pretrained": (
                environment["model"][
                    "pretrained"
                ]
            ),
            "embedding_dimension": (
                environment["model"][
                    "embedding_dimension"
                ]
            ),
            "environment_valid": (
                environment[
                    "environment_valid"
                ]
            ),
        },
        "validation": {
            "validators_expected": 25,
            "validators_present": len(
                validator_static_records
            ),
            "validators": (
                validator_static_records
            ),
            "static_validation_complete": True,
            "final_execution_batch_required": True,
            "experimental_generators_executed": False,
            "evaluators_executed": False,
            "embedding_generator_executed": False,
            "delivery_manifest_generator_executed": True,
        },
        "freeze": contract[
            "freeze_contract"
        ],
        "counts": {
            "tracked_files_before_delivery_commit": (
                len(tracked_paths)
            ),
            "expected_final_tracked_files": (
                inventory_contract[
                    "expected_final_tracked_files"
                ]
            ),
            "inventory_entries": len(
                inventory
            ),
            "manifest_self_excluded": 1,
            "tracked_clean": (
                state_counts[
                    "tracked_clean"
                ]
            ),
            "tracked_modified": (
                state_counts[
                    "tracked_modified"
                ]
            ),
            "untracked_delivery": (
                state_counts[
                    "untracked_delivery"
                ]
            ),
            "artifact_families": len(
                artifact_families
            ),
            "historical_result_files": len(
                historical_results_inventory
            ),
            "historical_figure_files": len(
                historical_figures_inventory
            ),
        },
        "artifact_families": (
            artifact_families
        ),
        "protected_historical_artifacts": {
            "conclusion_tecnica": {
                "path": (
                    conclusion_relative_path
                ),
                "normalized_content_matches_index": (
                    conclusion_matches_index
                ),
                "sha256": sha256_file(
                    PROJECT_ROOT
                    / conclusion_relative_path
                ),
            },
            "historical_results": {
                "excluded_prefix": (
                    "results/v2/"
                ),
                "file_count": len(
                    historical_results_inventory
                ),
                "inventory": (
                    historical_results_inventory
                ),
                "inventory_sha256": (
                    inventory_digest(
                        historical_results_inventory
                    )
                ),
            },
            "historical_figures": {
                "path": "figures/",
                "file_count": len(
                    historical_figures_inventory
                ),
                "inventory": (
                    historical_figures_inventory
                ),
                "inventory_sha256": (
                    inventory_digest(
                        historical_figures_inventory
                    )
                ),
            },
        },
        "inventory": inventory,
        "inventory_digest": {
            "algorithm": (
                "sha256_canonical_json"
            ),
            "canonicalization": (
                "ensure_ascii_false_"
                "sort_keys_true_"
                "compact_separators"
            ),
            "entries": len(inventory),
            "sha256": inventory_sha256,
        },
        "manifest_valid": True,
    }

    assert list(payload) == (
        required_sections
    )

    assert len(payload) == 16

    assert payload[
        "counts"
    ]["inventory_entries"] == (
        inventory_contract[
            "expected_inventory_entries"
        ]
    )

    assert payload[
        "environment"
    ]["environment_valid"] is True

    write_json_atomic(
        OUTPUT_PATH,
        payload,
    )

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.is_file()

    output_raw = OUTPUT_PATH.read_bytes()

    assert not output_raw.startswith(
        UTF8_BOM
    )

    assert b"\r" not in output_raw
    assert output_raw.endswith(b"\n")

    final_git_paths = current_git_paths()

    assert (
        final_git_paths["staged"]
        == []
    )

    expected_after_generation = sorted(
        delivery_paths[
            "allowed_final_worktree_paths"
        ]
    )

    actual_after_generation = sorted(
        {
            *final_git_paths["unstaged"],
            *final_git_paths["untracked"],
        }
    )

    assert (
        actual_after_generation
        == expected_after_generation
    )

    print("=" * 92)
    print("MANIFIESTO DE ENTREGA V2 GENERADO")
    print("=" * 92)
    print(
        "Ruta:",
        manifest_relative_path,
    )
    print(
        "Entradas inventariadas:",
        len(inventory),
    )
    print(
        "Familias:",
        len(artifact_families),
    )
    print(
        "Tracked clean:",
        state_counts["tracked_clean"],
    )
    print(
        "Tracked modified:",
        state_counts[
            "tracked_modified"
        ],
    )
    print(
        "Untracked delivery:",
        state_counts[
            "untracked_delivery"
        ],
    )
    print(
        "SHA-256 del inventario:",
        inventory_sha256,
    )
    print(
        "SHA-256 del manifiesto:",
        sha256_file(OUTPUT_PATH),
    )
    print("Manifiesto válido: True")


if __name__ == "__main__":
    main()
