"""Valida las métricas de recuperación con casos sintéticos controlados."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from metricas_retrieval_v2 import (
    aggregate_query_metrics,
    evaluate_query,
    ndcg_at_k,
    positive_margin,
    rank_descending,
    recall_at_k,
    reciprocal_rank,
    relevant_ranks,
)


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


def expect_error(
    function,
    expected_exception: type[Exception],
    context: str,
) -> None:
    try:
        function()
    except expected_exception:
        return

    raise AssertionError(
        f"{context}: no produjo "
        f"{expected_exception.__name__}."
    )


def validate_perfect_single_positive() -> dict:
    scores = np.asarray(
        [0.95, 0.40, 0.30, 0.20, 0.10],
        dtype=np.float64,
    )

    result = evaluate_query(
        scores=scores,
        relevant_indices={0},
        candidate_keys=(
            "A",
            "B",
            "C",
            "D",
            "E",
        ),
    )

    assert result["first_relevant_rank"] == 1
    assert_close(
        result["recall_at_1"],
        1.0,
        "perfect.recall_at_1",
    )
    assert_close(
        result["recall_at_5"],
        1.0,
        "perfect.recall_at_5",
    )
    assert_close(
        result["mrr"],
        1.0,
        "perfect.mrr",
    )
    assert_close(
        result["ndcg_at_10"],
        1.0,
        "perfect.ndcg",
    )
    assert_close(
        result["positive_margin"],
        0.55,
        "perfect.margin",
    )

    return result


def validate_positive_at_rank_three() -> dict:
    scores = np.asarray(
        [0.90, 0.80, 0.60, 0.40, 0.20],
        dtype=np.float64,
    )

    result = evaluate_query(
        scores=scores,
        relevant_indices={2},
        candidate_keys=(
            "A",
            "B",
            "C",
            "D",
            "E",
        ),
    )

    assert result["first_relevant_rank"] == 3

    assert_close(
        result["recall_at_1"],
        0.0,
        "rank3.recall_at_1",
    )

    assert_close(
        result["recall_at_5"],
        1.0,
        "rank3.recall_at_5",
    )

    assert_close(
        result["mrr"],
        1.0 / 3.0,
        "rank3.mrr",
    )

    assert_close(
        result["ndcg_at_10"],
        1.0 / math.log2(4.0),
        "rank3.ndcg",
    )

    assert_close(
        result["positive_margin"],
        -0.30,
        "rank3.margin",
    )

    return result


def validate_multiple_relevant() -> dict:
    scores = np.asarray(
        [0.90, 0.80, 0.70, 0.60, 0.50],
        dtype=np.float64,
    )

    result = evaluate_query(
        scores=scores,
        relevant_indices={1, 3},
        candidate_keys=(
            "A",
            "B",
            "C",
            "D",
            "E",
        ),
    )

    ranking = np.asarray(
        result["ranking_indices"],
        dtype=np.int64,
    )

    assert relevant_ranks(
        ranking,
        {1, 3},
    ) == [2, 4]

    assert_close(
        result["recall_at_1"],
        0.0,
        "multi.recall_at_1",
    )

    assert_close(
        result["recall_at_5"],
        1.0,
        "multi.recall_at_5",
    )

    assert_close(
        result["mrr"],
        0.5,
        "multi.mrr",
    )

    actual_dcg = (
        1.0 / math.log2(3.0)
        + 1.0 / math.log2(5.0)
    )

    ideal_dcg = (
        1.0
        + 1.0 / math.log2(3.0)
    )

    assert_close(
        result["ndcg_at_10"],
        actual_dcg / ideal_dcg,
        "multi.ndcg",
    )

    assert "positive_margin" not in result

    return result


def validate_tie_breaking() -> None:
    ranking = rank_descending(
        scores=[0.5, 0.5, 0.5, 0.5],
        candidate_keys=[
            "B",
            "A",
            "D",
            "C",
        ],
    )

    assert ranking.tolist() == [
        1,
        0,
        3,
        2,
    ]

    assert_close(
        recall_at_k(
            ranking,
            {1},
            1,
        ),
        1.0,
        "tie.recall",
    )

    assert_close(
        reciprocal_rank(
            ranking,
            {1},
        ),
        1.0,
        "tie.mrr",
    )


def validate_cutoff_larger_than_gallery() -> None:
    ranking = np.asarray(
        [0, 1, 2],
        dtype=np.int64,
    )

    assert_close(
        recall_at_k(
            ranking,
            {2},
            10,
        ),
        1.0,
        "large_k.recall",
    )

    expected_ndcg = (
        1.0 / math.log2(4.0)
    )

    assert_close(
        ndcg_at_k(
            ranking,
            {2},
            10,
        ),
        expected_ndcg,
        "large_k.ndcg",
    )


def validate_aggregate(
    perfect: dict,
    rank_three: dict,
) -> None:
    aggregate = aggregate_query_metrics(
        [
            perfect,
            rank_three,
        ]
    )

    assert_close(
        aggregate["recall_at_1"],
        0.5,
        "aggregate.recall_at_1",
    )

    assert_close(
        aggregate["recall_at_5"],
        1.0,
        "aggregate.recall_at_5",
    )

    assert_close(
        aggregate["mrr"],
        (
            1.0
            + 1.0 / 3.0
        )
        / 2.0,
        "aggregate.mrr",
    )

    assert_close(
        aggregate["positive_margin"],
        (
            0.55
            - 0.30
        )
        / 2.0,
        "aggregate.margin",
    )

    assert_close(
        aggregate[
            "hard_negative_accuracy"
        ],
        0.5,
        "aggregate.hard_accuracy",
    )

    assert_close(
        aggregate["query_count"],
        2.0,
        "aggregate.query_count",
    )


def validate_errors() -> None:
    expect_error(
        lambda: evaluate_query(
            scores=[],
            relevant_indices={0},
            candidate_keys=[],
        ),
        ValueError,
        "empty_scores",
    )

    expect_error(
        lambda: evaluate_query(
            scores=[0.1, 0.2],
            relevant_indices=set(),
            candidate_keys=["A", "B"],
        ),
        ValueError,
        "empty_relevance",
    )

    expect_error(
        lambda: rank_descending(
            scores=[0.1, 0.2],
            candidate_keys=["A", "A"],
        ),
        ValueError,
        "duplicate_keys",
    )

    expect_error(
        lambda: positive_margin(
            scores=[0.1, 0.2, 0.3],
            relevant_indices={0, 1},
        ),
        ValueError,
        "multiple_positive_margin",
    )

    expect_error(
        lambda: relevant_ranks(
            ranking=[0, 0, 2],
            relevant_indices={0},
        ),
        ValueError,
        "invalid_ranking",
    )


def validate_no_nested_comparisons() -> None:
    path = Path(
        __file__
    ).with_name(
        "metricas_retrieval_v2.py"
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    findings = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue

        nested = (
            isinstance(
                node.left,
                ast.Compare,
            )
            or any(
                isinstance(
                    comparator,
                    ast.Compare,
                )
                for comparator
                in node.comparators
            )
        )

        if nested:
            findings.append(
                node.lineno
            )

    assert not findings, (
        "Comparaciones anidadas detectadas: "
        f"{findings}."
    )


def main() -> None:
    validate_no_nested_comparisons()

    perfect = (
        validate_perfect_single_positive()
    )

    rank_three = (
        validate_positive_at_rank_three()
    )

    multiple = (
        validate_multiple_relevant()
    )

    validate_tie_breaking()
    validate_cutoff_larger_than_gallery()

    validate_aggregate(
        perfect,
        rank_three,
    )

    validate_errors()

    print("=" * 80)
    print("VALIDACIÓN DE MÉTRICAS DE RECUPERACIÓN V2 SUPERADA")
    print("=" * 80)
    print("Caso positivo en rango 1: correcto")
    print("Caso positivo en rango 3: correcto")
    print("Caso con múltiples relevantes: correcto")
    print("Desempate determinista: correcto")
    print("Cutoff mayor que galería: correcto")
    print("Agregación por consultas: correcta")
    print("Validaciones de errores: correctas")
    print("Comparaciones anidadas: 0")
    print()
    print("Valores del caso con múltiples relevantes:")
    print(
        "- MRR:",
        multiple["mrr"],
    )
    print(
        "- nDCG@10:",
        multiple["ndcg_at_10"],
    )
    print()
    print(
        "El validador no utilizó ni modificó "
        "los resultados experimentales."
    )


if __name__ == "__main__":
    main()
