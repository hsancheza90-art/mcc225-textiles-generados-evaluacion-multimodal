"""Funciones deterministas para evaluar recuperación multimodal v2."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np


METRIC_KEYS = (
    "recall_at_1",
    "recall_at_5",
    "mrr",
    "ndcg_at_10",
)


def validate_scores(
    scores: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convierte y valida un vector unidimensional de similitudes."""

    array = np.asarray(
        scores,
        dtype=np.float64,
    )

    if array.ndim != 1:
        raise ValueError(
            "scores debe ser un vector unidimensional."
        )

    if array.size == 0:
        raise ValueError(
            "scores no puede estar vacío."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "scores contiene valores no finitos."
        )

    return array


def validate_candidate_keys(
    candidate_keys: Sequence[Hashable],
    expected_count: int,
) -> list[Hashable]:
    """Valida claves únicas para desempate reproducible."""

    keys = list(candidate_keys)

    if len(keys) != expected_count:
        raise ValueError(
            "La cantidad de candidate_keys no coincide "
            "con la cantidad de scores."
        )

    if len(set(keys)) != len(keys):
        raise ValueError(
            "candidate_keys debe contener valores únicos."
        )

    return keys


def validate_relevant_indices(
    relevant_indices: Iterable[int],
    candidate_count: int,
) -> frozenset[int]:
    """Valida los índices de los elementos relevantes."""

    relevant = frozenset(
        int(index)
        for index in relevant_indices
    )

    if not relevant:
        raise ValueError(
            "Debe existir al menos un elemento relevante."
        )

    invalid = sorted(
        index
        for index in relevant
        if index < 0 or index >= candidate_count
    )

    if invalid:
        raise ValueError(
            "Índices relevantes fuera de rango: "
            f"{invalid}."
        )

    return relevant


def rank_descending(
    scores: Sequence[float] | np.ndarray,
    candidate_keys: Sequence[Hashable],
) -> np.ndarray:
    """Ordena por score descendente y desempata por clave textual."""

    score_array = validate_scores(scores)

    keys = validate_candidate_keys(
        candidate_keys,
        score_array.size,
    )

    ranking = sorted(
        range(score_array.size),
        key=lambda index: (
            -float(score_array[index]),
            str(keys[index]),
        ),
    )

    return np.asarray(
        ranking,
        dtype=np.int64,
    )


def relevant_ranks(
    ranking: Sequence[int] | np.ndarray,
    relevant_indices: Iterable[int],
) -> list[int]:
    """Devuelve posiciones base 1 de todos los relevantes recuperados."""

    ranking_array = np.asarray(
        ranking,
        dtype=np.int64,
    )

    if ranking_array.ndim != 1:
        raise ValueError(
            "ranking debe ser un vector unidimensional."
        )

    candidate_count = ranking_array.size

    if candidate_count == 0:
        raise ValueError(
            "ranking no puede estar vacío."
        )

    expected = set(
        range(candidate_count)
    )

    actual = set(
        int(index)
        for index in ranking_array
    )

    if actual != expected:
        raise ValueError(
            "ranking debe ser una permutación completa "
            "de los candidatos."
        )

    relevant = validate_relevant_indices(
        relevant_indices,
        candidate_count,
    )

    return [
        rank
        for rank, candidate_index in enumerate(
            ranking_array,
            start=1,
        )
        if int(candidate_index) in relevant
    ]


def recall_at_k(
    ranking: Sequence[int] | np.ndarray,
    relevant_indices: Iterable[int],
    k: int,
) -> float:
    """Calcula éxito@K: al menos un relevante en los primeros K."""

    if k <= 0:
        raise ValueError(
            "k debe ser mayor que cero."
        )

    ranks = relevant_ranks(
        ranking,
        relevant_indices,
    )

    effective_k = min(
        k,
        len(ranking),
    )

    return float(
        min(ranks) <= effective_k
    )


def reciprocal_rank(
    ranking: Sequence[int] | np.ndarray,
    relevant_indices: Iterable[int],
) -> float:
    """Calcula el recíproco de la posición del primer relevante."""

    ranks = relevant_ranks(
        ranking,
        relevant_indices,
    )

    return 1.0 / float(min(ranks))


def dcg_at_k(
    ranking: Sequence[int] | np.ndarray,
    relevant_indices: Iterable[int],
    k: int,
) -> float:
    """Calcula DCG@K con relevancia binaria."""

    if k <= 0:
        raise ValueError(
            "k debe ser mayor que cero."
        )

    ranking_array = np.asarray(
        ranking,
        dtype=np.int64,
    )

    relevant = validate_relevant_indices(
        relevant_indices,
        ranking_array.size,
    )

    effective_k = min(
        k,
        ranking_array.size,
    )

    dcg = 0.0

    for rank, candidate_index in enumerate(
        ranking_array[:effective_k],
        start=1,
    ):
        if int(candidate_index) in relevant:
            dcg += 1.0 / math.log2(
                rank + 1.0
            )

    return dcg


def ndcg_at_k(
    ranking: Sequence[int] | np.ndarray,
    relevant_indices: Iterable[int],
    k: int,
) -> float:
    """Calcula nDCG@K con relevancia binaria."""

    ranking_array = np.asarray(
        ranking,
        dtype=np.int64,
    )

    relevant = validate_relevant_indices(
        relevant_indices,
        ranking_array.size,
    )

    actual_dcg = dcg_at_k(
        ranking_array,
        relevant,
        k,
    )

    effective_relevant = min(
        len(relevant),
        k,
        ranking_array.size,
    )

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1.0)
        for rank in range(
            1,
            effective_relevant + 1,
        )
    )

    if ideal_dcg <= 0.0:
        raise AssertionError(
            "IDCG debe ser positivo."
        )

    return actual_dcg / ideal_dcg


def positive_margin(
    scores: Sequence[float] | np.ndarray,
    relevant_indices: Iterable[int],
) -> float:
    """Calcula score positivo menos el mayor score negativo."""

    score_array = validate_scores(scores)

    relevant = validate_relevant_indices(
        relevant_indices,
        score_array.size,
    )

    if len(relevant) != 1:
        raise ValueError(
            "positive_margin requiere exactamente "
            "un elemento relevante."
        )

    positive_index = next(
        iter(relevant)
    )

    negative_indices = [
        index
        for index in range(score_array.size)
        if index != positive_index
    ]

    if not negative_indices:
        raise ValueError(
            "positive_margin requiere al menos un negativo."
        )

    return float(
        score_array[positive_index]
        - np.max(
            score_array[negative_indices]
        )
    )


def evaluate_query(
    scores: Sequence[float] | np.ndarray,
    relevant_indices: Iterable[int],
    candidate_keys: Sequence[Hashable],
) -> dict[str, Any]:
    """Evalúa una consulta individual de recuperación."""

    score_array = validate_scores(scores)

    relevant = validate_relevant_indices(
        relevant_indices,
        score_array.size,
    )

    ranking = rank_descending(
        score_array,
        candidate_keys,
    )

    ranks = relevant_ranks(
        ranking,
        relevant,
    )

    result: dict[str, Any] = {
        "candidate_count": int(
            score_array.size
        ),
        "relevant_count": len(relevant),
        "first_relevant_rank": min(ranks),
        "recall_at_1": recall_at_k(
            ranking,
            relevant,
            1,
        ),
        "recall_at_5": recall_at_k(
            ranking,
            relevant,
            5,
        ),
        "mrr": reciprocal_rank(
            ranking,
            relevant,
        ),
        "ndcg_at_10": ndcg_at_k(
            ranking,
            relevant,
            10,
        ),
        "ranking_indices": [
            int(index)
            for index in ranking
        ],
    }

    if len(relevant) == 1 and score_array.size >= 2:
        result["positive_margin"] = (
            positive_margin(
                score_array,
                relevant,
            )
        )

    return result


def aggregate_query_metrics(
    query_results: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Promedia métricas escalares sobre una colección de consultas."""

    rows = list(query_results)

    if not rows:
        raise ValueError(
            "Debe existir al menos un resultado."
        )

    aggregate: dict[str, float] = {}

    for metric in METRIC_KEYS:
        values = [
            float(row[metric])
            for row in rows
        ]

        if not np.isfinite(values).all():
            raise ValueError(
                f"{metric} contiene valores no finitos."
            )

        aggregate[metric] = float(
            np.mean(values)
        )

    margins = [
        float(row["positive_margin"])
        for row in rows
        if "positive_margin" in row
    ]

    if margins:
        aggregate["positive_margin"] = float(
            np.mean(margins)
        )

        aggregate[
            "hard_negative_accuracy"
        ] = aggregate["recall_at_1"]

    aggregate["query_count"] = float(
        len(rows)
    )

    return aggregate
