"Valida independientemente el manifiesto integral de la entrega v2."

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "config" / "entrega_final_v2.json"
MANIFEST_PATH = PROJECT_ROOT / "results" / "v2" / "manifiesto_entrega_v2.json"
UTF8_BOM = b"\xef\xbb\xbf"


def run_git(
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def run_git_bytes(
    arguments: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    assert not raw.startswith(UTF8_BOM), f"{path}: contiene BOM."
    assert b"\r" not in raw, f"{path}: no utiliza exclusivamente LF."
    assert raw.endswith(b"\n"), f"{path}: no termina con salto de línea."
    value = json.loads(raw.decode("utf-8"))
    assert isinstance(value, dict)
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob_worktree(relative_path: str) -> str:
    value = run_git(
        ["hash-object", f"--path={relative_path}", "--", relative_path]
    ).stdout.strip()
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)
    return value


def git_blob_index(relative_path: str) -> str | None:
    process = run_git(["rev-parse", f":{relative_path}"], check=False)
    if process.returncode != 0:
        return None
    value = process.stdout.strip()
    assert len(value) == 40
    assert all(character in "0123456789abcdef" for character in value)
    return value


def normalize_tracked_text(data: bytes) -> str:
    text = data.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def inspect_content(data: bytes) -> dict[str, Any]:
    bom = data.startswith(UTF8_BOM)
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {
            "encoding": "binary",
            "bom": False,
            "line_endings": "not_applicable",
        }

    without_crlf = data.replace(b"\r\n", b"")
    counts = (
        data.count(b"\r\n"),
        without_crlf.count(b"\n"),
        without_crlf.count(b"\r"),
    )

    if sum(count > 0 for count in counts) > 1:
        line_endings = "mixed"
    elif counts[0] > 0:
        line_endings = "crlf"
    elif counts[1] > 0:
        line_endings = "lf"
    elif counts[2] > 0:
        line_endings = "cr"
    else:
        line_endings = "none"

    return {
        "encoding": "utf-8-sig" if bom else "utf-8",
        "bom": bom,
        "line_endings": line_endings,
    }


def classify_family(
    relative_path: str,
    rules: list[dict[str, str]],
) -> str:
    matching = [
        rule
        for rule in rules
        if relative_path.startswith(rule["prefix"])
    ]
    if not matching:
        return "root"
    return max(matching, key=lambda rule: len(rule["prefix"]))["family"]


def inventory_digest(inventory: list[dict[str, Any]]) -> str:
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
    for relative_path in sorted(relative_paths):
        path = PROJECT_ROOT / relative_path
        assert path.exists() and path.is_file(), relative_path
        rows.append(
            {
                "path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def current_git_paths() -> dict[str, list[str]]:
    return {
        "unstaged": sorted(
            line.strip()
            for line in run_git(["diff", "--name-only"]).stdout.splitlines()
            if line.strip()
        ),
        "staged": sorted(
            line.strip()
            for line in run_git(
                ["diff", "--cached", "--name-only"]
            ).stdout.splitlines()
            if line.strip()
        ),
        "untracked": sorted(
            line.strip()
            for line in run_git(
                ["ls-files", "--others", "--exclude-standard"]
            ).stdout.splitlines()
            if line.strip()
        ),
    }



def git_blob_at_commit(
    commit: str,
    relative_path: str,
) -> str | None:
    process = run_git(
        [
            "rev-parse",
            f"{commit}:{relative_path}",
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


def git_tracked_paths_at_commit(
    commit: str,
) -> list[str]:
    paths = sorted(
        line.strip()
        for line in run_git(
            [
                "ls-tree",
                "-r",
                "--name-only",
                commit,
            ]
        ).stdout.splitlines()
        if line.strip()
    )

    assert len(paths) == len(
        set(paths)
    )

    return paths


def detect_validation_mode(
    *,
    git_paths: dict[str, list[str]],
    current_head: str,
    source_head_commit: str,
    current_tracked_paths: list[str],
    source_tracked_paths: list[str],
    expected_unstaged: list[str],
    expected_untracked: list[str],
    allowed_final_worktree_paths: list[str],
    final_tree_paths: list[str],
    release_tag_present: bool,
) -> str:
    if current_head == source_head_commit:
        if (
            git_paths["staged"] == []
            and git_paths["unstaged"]
            == expected_unstaged
            and git_paths["untracked"]
            == expected_untracked
            and not release_tag_present
        ):
            assert (
                current_tracked_paths
                == source_tracked_paths
            )
            return "pre_add"

        if (
            git_paths["staged"]
            == allowed_final_worktree_paths
            and git_paths["unstaged"] == []
            and git_paths["untracked"] == []
            and not release_tag_present
        ):
            assert (
                current_tracked_paths
                == final_tree_paths
            )
            return "staged"

        raise AssertionError(
            "Estado Git no permitido antes "
            "del commit final."
        )

    assert git_paths == {
        "unstaged": [],
        "staged": [],
        "untracked": [],
    }

    assert (
        current_tracked_paths
        == final_tree_paths
    )

    if release_tag_present:
        return "tagged"

    return "committed"


def validator_static_record(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    assert path.exists() and path.is_file(), relative_path

    source = path.read_text(encoding="utf-8-sig")
    compile(
        source,
        relative_path,
        "exec",
        dont_inherit=True,
        optimize=0,
    )
    tree = ast.parse(source, filename=relative_path)

    main_count = sum(
        1
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    imported_roots = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    forbidden_imports = sorted(
        imported_roots
        & {"open_clip", "torch", "torchvision", "requests", "urllib3"}
    )

    assert main_count == 1, relative_path
    assert not forbidden_imports, f"{relative_path}: {forbidden_imports}"

    return {
        "path": relative_path,
        "main_count": main_count,
        "syntax_valid": True,
        "forbidden_imports": forbidden_imports,
    }


def build_inventory_row(
    relative_path: str,
    *,
    source_head_commit: str,
    source_tracked_set: set[str],
    family_rules: list[dict[str, str]],
) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    assert path.exists() and path.is_file(), relative_path
    data = path.read_bytes()

    worktree_blob = git_blob_worktree(
        relative_path
    )

    source_blob = git_blob_at_commit(
        source_head_commit,
        relative_path,
    )

    assert (
        relative_path in source_tracked_set
    ) == (source_blob is not None)

    if source_blob is None:
        index_blob = None
        git_state = "untracked_delivery"

    elif source_blob == worktree_blob:
        index_blob = source_blob
        git_state = "tracked_clean"

    else:
        index_blob = source_blob
        git_state = "tracked_modified"

    content = inspect_content(data)

    return {
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
        "encoding": content["encoding"],
        "bom": content["bom"],
        "line_endings": content[
            "line_endings"
        ],
    }


def main() -> None:
    contract = load_json(CONTRACT_PATH)
    manifest = load_json(MANIFEST_PATH)

    assert contract["schema_version"] == 1
    assert contract["contract_id"] == "DELIVERY-V2"
    assert contract["dataset_version"] == "v2"
    assert manifest["schema_version"] == 1
    assert manifest["contract_id"] == "DELIVERY-V2-MANIFEST"
    assert manifest["dataset_version"] == "v2"
    assert manifest["stage"] == "final_delivery_manifest"

    required_sections = contract["required_manifest_sections"]
    assert len(required_sections) == 16
    assert len(set(required_sections)) == 16
    assert list(manifest) == required_sections

    delivery_paths = contract["delivery_paths"]
    manifest_relative_path = delivery_paths["manifest"]
    assert MANIFEST_PATH == PROJECT_ROOT / manifest_relative_path

    git_paths = current_git_paths()
    expected_unstaged_before_manifest = sorted(
        delivery_paths[
            "preexisting_modified_paths"
        ]
    )
    expected_untracked_before_manifest = sorted(
        delivery_paths[
            "delivery_source_paths"
        ]
    )
    allowed_final_worktree_paths = sorted(
        delivery_paths[
            "allowed_final_worktree_paths"
        ]
    )

    git_contract = contract["git_contract"]

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

    source_head_commit = manifest[
        "git"
    ]["source_head_commit"]

    assert len(source_head_commit) == 40
    assert all(
        character in "0123456789abcdef"
        for character in source_head_commit
    )

    source_commit_exists = run_git(
        [
            "cat-file",
            "-e",
            f"{source_head_commit}^{{commit}}",
        ],
        check=False,
    )

    assert (
        source_commit_exists.returncode == 0
    )

    source_head_short = run_git(
        [
            "rev-parse",
            "--short=12",
            source_head_commit,
        ]
    ).stdout.strip()

    source_head_tree = run_git(
        [
            "rev-parse",
            f"{source_head_commit}^{{tree}}",
        ]
    ).stdout.strip()

    source_head_commit_time = run_git(
        [
            "show",
            "-s",
            "--format=%cI",
            source_head_commit,
        ]
    ).stdout.strip()

    current_head = run_git(
        [
            "rev-parse",
            "HEAD",
        ]
    ).stdout.strip()

    release_tag = git_contract[
        "expected_release_tag"
    ]

    release_tag_present = bool(
        run_git(
            [
                "tag",
                "--list",
                release_tag,
            ]
        ).stdout.strip()
    )

    source_tracked_paths = (
        git_tracked_paths_at_commit(
            source_head_commit
        )
    )

    expected_source_tracked_files = (
        contract["inventory_contract"][
            "expected_final_tracked_files"
        ]
    )

    assert len(source_tracked_paths) == (
        expected_source_tracked_files
    )
    assert len(set(source_tracked_paths)) == (
        expected_source_tracked_files
    )

    source_tracked_path_set = set(
        source_tracked_paths
    )
    expected_unstaged_after_manifest = sorted(
        path
        for path in allowed_final_worktree_paths
        if path in source_tracked_path_set
    )
    expected_untracked_after_manifest = sorted(
        path
        for path in allowed_final_worktree_paths
        if path not in source_tracked_path_set
    )

    current_tracked_paths = sorted(
        line.strip()
        for line in run_git(
            [
                "ls-files",
            ]
        ).stdout.splitlines()
        if line.strip()
    )

    assert len(
        current_tracked_paths
    ) == len(
        set(current_tracked_paths)
    )

    final_tree_paths = sorted(
        {
            *source_tracked_paths,
            *delivery_paths[
                "delivery_source_paths"
            ],
            manifest_relative_path,
        }
    )

    inventory_contract = contract[
        "inventory_contract"
    ]

    assert len(final_tree_paths) == (
        inventory_contract[
            "expected_final_tracked_files"
        ]
    )

    validation_mode = (
        detect_validation_mode(
            git_paths=git_paths,
            current_head=current_head,
            source_head_commit=(
                source_head_commit
            ),
            current_tracked_paths=(
                current_tracked_paths
            ),
            source_tracked_paths=(
                source_tracked_paths
            ),
            expected_unstaged=(
                expected_unstaged_after_manifest
            ),
            expected_untracked=(
                expected_untracked_after_manifest
            ),
            allowed_final_worktree_paths=(
                allowed_final_worktree_paths
            ),
            final_tree_paths=(
                final_tree_paths
            ),
            release_tag_present=(
                release_tag_present
            ),
        )
    )

    if current_head != source_head_commit:
        ancestor_process = run_git(
            [
                "merge-base",
                "--is-ancestor",
                source_head_commit,
                current_head,
            ],
            check=False,
        )

        assert (
            ancestor_process.returncode == 0
        )

        commit_distance_text = run_git(
            [
                "rev-list",
                "--count",
                (
                    f"{source_head_commit}"
                    f"..{current_head}"
                ),
            ]
        ).stdout.strip()

        assert commit_distance_text.isdigit()
        assert int(commit_distance_text) == 1

    if validation_mode == "tagged":
        tag_type = run_git(
            [
                "cat-file",
                "-t",
                release_tag,
            ]
        ).stdout.strip()

        assert tag_type == "tag"

        tagged_commit = run_git(
            [
                "rev-list",
                "-n",
                "1",
                release_tag,
            ]
        ).stdout.strip()

        assert tagged_commit == current_head

        tag_message = run_git(
            [
                "for-each-ref",
                f"refs/tags/{release_tag}",
                "--format=%(contents)",
            ]
        ).stdout

        assert (
            sha256_file(MANIFEST_PATH)
            in tag_message
        )

    else:
        assert not release_tag_present

    assert branch == git_contract["branch"]
    assert upstream == git_contract["upstream"]
    assert origin_url == git_contract["origin_url"]

    expected_git = {
        "branch": branch,
        "upstream": upstream,
        "origin_url": origin_url,
        "source_head_commit": (
            source_head_commit
        ),
        "source_head_short": (
            source_head_short
        ),
        "source_head_tree": source_head_tree,
        "source_head_commit_time": (
            source_head_commit_time
        ),
        "generation_time_basis": (
            "source_head_commit_time"
        ),
        "expected_release_tag": release_tag,
        "staged_paths": [],
        "unstaged_paths": (
            expected_unstaged_before_manifest
        ),
        "untracked_paths_before_manifest": (
            sorted(
                delivery_paths[
                    "delivery_source_paths"
                ]
            )
        ),
    }

    assert manifest["git"] == expected_git

    assert (
        manifest["generated_at"]
        == source_head_commit_time
    )

    tracked_paths = source_tracked_paths

    inventory_paths = sorted(
        {
            *source_tracked_paths,
            *delivery_paths[
                "delivery_source_paths"
            ],
        }
        - {manifest_relative_path}
    )

    assert (
        manifest_relative_path
        not in inventory_paths
    )

    assert len(inventory_paths) == (
        inventory_contract[
            "expected_inventory_entries"
        ]
    )

    if validation_mode == "pre_add":
        final_delivery_paths = sorted(
            {
                *source_tracked_paths,
                *git_paths["untracked"],
            }
        )
    else:
        final_delivery_paths = (
            current_tracked_paths
        )

    assert (
        final_delivery_paths
        == final_tree_paths
    )

    source_tracked_set = set(
        source_tracked_paths
    )

    family_rules = contract[
        "artifact_family_rules"
    ]

    expected_inventory = [
        build_inventory_row(
            relative_path,
            source_head_commit=(
                source_head_commit
            ),
            source_tracked_set=(
                source_tracked_set
            ),
            family_rules=family_rules,
        )
        for relative_path in inventory_paths
    ]

    assert (
        manifest["inventory"]
        == expected_inventory
    )

    assert [
        row["path"]
        for row in expected_inventory
    ] == inventory_paths

    assert len(
        {
            row["path"]
            for row in expected_inventory
        }
    ) == len(expected_inventory)

    required_fields = set(
        inventory_contract[
            "required_fields"
        ]
    )

    assert all(
        set(row) == required_fields
        for row in expected_inventory
    )

    expected_inventory_sha256 = (
        inventory_digest(
            expected_inventory
        )
    )

    assert manifest[
        "inventory_digest"
    ] == {
        "algorithm": (
            "sha256_canonical_json"
        ),
        "canonicalization": (
            "ensure_ascii_false_"
            "sort_keys_true_"
            "compact_separators"
        ),
        "entries": len(
            expected_inventory
        ),
        "sha256": (
            expected_inventory_sha256
        ),
    }

    state_counts = Counter(
        row["git_state"]
        for row in expected_inventory
    )

    inventory_path_set = set(
        inventory_paths
    )
    expected_tracked_modified = len(
        set(
            expected_unstaged_before_manifest
        )
        & inventory_path_set
    )
    expected_untracked_delivery = len(
        set(
            expected_untracked_before_manifest
        )
        & inventory_path_set
    )
    expected_tracked_clean = (
        len(inventory_paths)
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

    family_counts = Counter(row["family"] for row in expected_inventory)
    family_sizes = Counter()
    for row in expected_inventory:
        family_sizes[row["family"]] += row["size_bytes"]

    expected_families = [
        {
            "family": family,
            "file_count": family_counts[family],
            "size_bytes": family_sizes[family],
        }
        for family in sorted(family_counts)
    ]
    assert manifest["artifact_families"] == expected_families

    historical_results_paths = [
        path
        for path in tracked_paths
        if path.startswith("results/")
        and not path.startswith("results/v2/")
    ]
    historical_figures_paths = [
        path for path in tracked_paths if path.startswith("figures/")
    ]
    historical_results_inventory = directory_inventory(
        historical_results_paths
    )
    historical_figures_inventory = directory_inventory(
        historical_figures_paths
    )

    conclusion_path = "results/conclusion_tecnica.md"
    conclusion_matches_index = (
        normalize_tracked_text(
            (
                PROJECT_ROOT
                / conclusion_path
            ).read_bytes()
        )
        == normalize_tracked_text(
            run_git_bytes(
                [
                    "show",
                    (
                        f"{source_head_commit}:"
                        f"{conclusion_path}"
                    ),
                ]
            ).stdout
        )
    )
    assert conclusion_matches_index

    expected_protected = {
        "conclusion_tecnica": {
            "path": conclusion_path,
            "normalized_content_matches_index": True,
            "sha256": sha256_file(PROJECT_ROOT / conclusion_path),
        },
        "historical_results": {
            "excluded_prefix": "results/v2/",
            "file_count": len(historical_results_inventory),
            "inventory": historical_results_inventory,
            "inventory_sha256": inventory_digest(
                historical_results_inventory
            ),
        },
        "historical_figures": {
            "path": "figures/",
            "file_count": len(historical_figures_inventory),
            "inventory": historical_figures_inventory,
            "inventory_sha256": inventory_digest(
                historical_figures_inventory
            ),
        },
    }
    assert manifest["protected_historical_artifacts"] == expected_protected

    environment_contract = contract["environment_contract"]
    environment_path = PROJECT_ROOT / environment_contract["source"]
    environment = load_json(environment_path)
    expected_environment = {
        "source": environment_contract["source"],
        "source_sha256": sha256_file(environment_path),
        "python": environment["python"]["version"],
        "torch": environment["runtime"]["torch_version"],
        "torchvision": environment["runtime"]["torchvision_version"],
        "open_clip": environment["packages"]["open-clip-torch"],
        "device": environment["runtime"]["canonical_device"],
        "model_architecture": environment["model"]["architecture"],
        "pretrained": environment["model"]["pretrained"],
        "embedding_dimension": environment["model"]["embedding_dimension"],
        "environment_valid": environment["environment_valid"],
    }
    assert manifest["environment"] == expected_environment
    assert expected_environment["python"] == environment_contract["python"]
    assert expected_environment["torch"] == environment_contract["torch"]
    assert expected_environment["torchvision"] == environment_contract[
        "torchvision"
    ]
    assert expected_environment["open_clip"] == environment_contract[
        "open_clip"
    ]
    assert expected_environment["device"] == environment_contract[
        "canonical_device"
    ]
    assert expected_environment["model_architecture"] == environment_contract[
        "model_architecture"
    ]
    assert expected_environment["pretrained"] == environment_contract[
        "pretrained"
    ]
    assert expected_environment["embedding_dimension"] == environment_contract[
        "embedding_dimension"
    ]
    assert expected_environment["environment_valid"] is True

    validators = contract["validation_contract"]["validators"]
    assert len(validators) == 25
    assert len(set(validators)) == 25
    assert validators == sorted(validators)
    validator_records = [
        validator_static_record(validator) for validator in validators
    ]

    expected_validation = {
        "validators_expected": 25,
        "validators_present": len(validator_records),
        "validators": validator_records,
        "static_validation_complete": True,
        "final_execution_batch_required": True,
        "experimental_generators_executed": False,
        "evaluators_executed": False,
        "embedding_generator_executed": False,
        "delivery_manifest_generator_executed": True,
    }
    assert manifest["validation"] == expected_validation
    assert manifest["project"] == contract["project"]
    assert manifest["freeze"] == contract["freeze_contract"]

    expected_counts = {
        "tracked_files_before_delivery_commit": len(tracked_paths),
        "expected_final_tracked_files": inventory_contract[
            "expected_final_tracked_files"
        ],
        "inventory_entries": len(expected_inventory),
        "manifest_self_excluded": 1,
        "tracked_clean": state_counts["tracked_clean"],
        "tracked_modified": state_counts["tracked_modified"],
        "untracked_delivery": state_counts["untracked_delivery"],
        "artifact_families": len(expected_families),
        "historical_result_files": len(historical_results_inventory),
        "historical_figure_files": len(historical_figures_inventory),
    }
    assert manifest["counts"] == expected_counts
    assert manifest["manifest_valid"] is True

    print("=" * 92)
    print(
        "VALIDACIÓN INDEPENDIENTE DEL "
        "MANIFIESTO DE ENTREGA V2 SUPERADA"
    )
    print("=" * 92)
    print("Modo Git validado:", validation_mode)
    print("Entradas verificadas:", len(expected_inventory))
    print("Familias verificadas:", len(expected_families))
    print("Validadores verificados:", len(validator_records))
    print("Tracked clean:", state_counts["tracked_clean"])
    print("Tracked modified:", state_counts["tracked_modified"])
    print("Untracked delivery:", state_counts["untracked_delivery"])
    print("Resultados históricos:", len(historical_results_inventory))
    print("Figuras históricas:", len(historical_figures_inventory))
    print("SHA-256 del inventario:", expected_inventory_sha256)
    print("SHA-256 del manifiesto:", sha256_file(MANIFEST_PATH))
    print("Manifiesto válido: True")


if __name__ == "__main__":
    main()
