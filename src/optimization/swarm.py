"""Ant Colony Optimization próprio para construção de progressões harmônicas."""

from __future__ import annotations

import random

from .problem import (
    DOMINANT_DEGREE,
    SEQUENCE_LENGTH,
    TRANSITION_PREFERENCES,
    DEFAULT_TRANSITION_PREFERENCE,
    EvaluationBudget,
    MusicInstance,
    SearchResult,
    repair_sequence,
)


def _weighted_choice(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return len(weights) - 1


def _construct_ant(
    instance: MusicInstance,
    pheromone: list[list[float]],
    rng: random.Random,
    alpha: float,
    beta: float,
) -> list[int]:
    preferences = TRANSITION_PREFERENCES[instance.mode]
    sequence = [0]
    for position in range(1, SEQUENCE_LENGTH - 1):
        previous = sequence[-1]
        remaining_slots_after = (SEQUENCE_LENGTH - 2) - position
        if DOMINANT_DEGREE not in sequence and remaining_slots_after == 0:
            choices = [DOMINANT_DEGREE]
        else:
            choices = [degree for degree in range(7) if degree != previous]
        weights = []
        for degree in choices:
            pheromone_weight = max(0.01, pheromone[previous][degree]) ** alpha
            heuristic = preferences[previous].get(degree, DEFAULT_TRANSITION_PREFERENCE)
            weights.append(pheromone_weight * max(0.03, heuristic) ** beta)
        sequence.append(choices[_weighted_choice(weights, rng)])
    sequence.append(0)
    return repair_sequence(sequence, rng)


def run_aco(
    instance: MusicInstance,
    evaluation_budget: int = 2000,
    seed: int = 42,
    ants_per_update: int = 24,
    elite_ants: int = 4,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.20,
    deposit_scale: float = 1.0,
) -> SearchResult:
    """Executa ACO discreto; cada formiga consome uma avaliação do orçamento."""

    if ants_per_update < 1 or elite_ants < 1:
        raise ValueError("ants_per_update e elite_ants devem ser positivos.")
    if not 0.0 < evaporation_rate <= 1.0:
        raise ValueError("A evaporação deve estar no intervalo (0, 1].")

    rng = random.Random(seed)
    tracker = EvaluationBudget(instance, evaluation_budget)
    pheromone = [[1.0 for _ in range(7)] for _ in range(7)]
    while tracker.evaluations < evaluation_budget:
        batch: list[tuple[tuple[int, ...], float]] = []
        batch_size = min(ants_per_update, evaluation_budget - tracker.evaluations)
        for _ in range(batch_size):
            candidate = tuple(_construct_ant(instance, pheromone, rng, alpha, beta))
            result = tracker.evaluate(candidate)
            batch.append((candidate, result.objective))

        # Evaporação e reforço das melhores trilhas da colônia recém-avaliada.
        for source in range(7):
            for target in range(7):
                pheromone[source][target] *= 1.0 - evaporation_rate
        batch.sort(key=lambda item: item[1], reverse=True)
        for sequence, score in batch[:elite_ants]:
            deposit = deposit_scale * (0.5 + score / 100.0)
            for source, target in zip(sequence, sequence[1:]):
                pheromone[source][target] += deposit

        # Reforço global modesto para estabilizar a melhor solução encontrada.
        if tracker.best_sequence is not None and tracker.best_evaluation is not None:
            deposit = 0.25 * deposit_scale * (0.5 + tracker.best_evaluation.objective / 100.0)
            for source, target in zip(tracker.best_sequence, tracker.best_sequence[1:]):
                pheromone[source][target] += deposit
        for source in range(7):
            for target in range(7):
                pheromone[source][target] = min(20.0, max(0.01, pheromone[source][target]))

    return tracker.result()

