"""CLONALG próprio com clonagem proporcional e hipermutação adaptativa."""

from __future__ import annotations

import random

from .problem import (
    Evaluation,
    EvaluationBudget,
    MusicInstance,
    SearchResult,
    random_feasible_sequence,
    repair_sequence,
)


def _mutate_clone(sequence: tuple[int, ...], rng: random.Random, rate: float) -> list[int]:
    clone = list(sequence)
    changed_positions: list[int] = []
    for index in range(1, len(clone) - 1):
        if rng.random() < rate:
            changed_positions.append(index)
    if not changed_positions:
        changed_positions.append(rng.randrange(1, len(clone) - 1))
    for index in changed_positions:
        alternatives = [degree for degree in range(7) if degree != clone[index]]
        clone[index] = rng.choice(alternatives)
    return repair_sequence(clone, rng)


def run_clonalg(
    instance: MusicInstance,
    evaluation_budget: int = 2000,
    seed: int = 42,
    population_size: int = 30,
    selected_count: int = 10,
    clone_factor: int = 6,
    random_replacements: int = 4,
) -> SearchResult:
    """Executa seleção clonal até consumir exatamente o orçamento definido."""

    if population_size < 2:
        raise ValueError("A população do CLONALG deve ter pelo menos 2 anticorpos.")
    if selected_count < 1 or clone_factor < 1 or random_replacements < 0:
        raise ValueError("Seleção e clonagem devem ser positivas; reposições não negativas.")

    rng = random.Random(seed)
    tracker = EvaluationBudget(instance, evaluation_budget)
    population: list[tuple[tuple[int, ...], Evaluation]] = []
    for _ in range(min(population_size, evaluation_budget)):
        candidate = tuple(random_feasible_sequence(rng))
        population.append((candidate, tracker.evaluate(candidate)))

    while tracker.evaluations < evaluation_budget:
        population.sort(key=lambda item: item[1].objective, reverse=True)
        selected = population[: min(selected_count, len(population))]
        planned_clones: list[tuple[int, ...]] = []

        # Anticorpos de maior afinidade recebem mais clones e taxa de mutação menor.
        denominator = max(1, len(selected) - 1)
        for rank, (antibody, evaluation) in enumerate(selected):
            clone_count = max(1, round(clone_factor * (len(selected) - rank) / denominator))
            affinity = max(0.0, min(1.0, evaluation.objective / 100.0))
            mutation_rate = 0.05 + 0.45 * (1.0 - affinity)
            for _ in range(clone_count):
                planned_clones.append(_mutate_clone(antibody, rng, mutation_rate))

        new_population = list(selected)
        clone_index = 0
        while clone_index < len(planned_clones) and tracker.evaluations < evaluation_budget:
            candidate = planned_clones[clone_index]
            result = tracker.evaluate(candidate)
            new_population.append((tuple(candidate), result))
            clone_index += 1

        # Introduz diversidade por anticorpos aleatórios a cada rodada clonal.
        immigrant_count = min(random_replacements, evaluation_budget - tracker.evaluations)
        for _ in range(immigrant_count):
            candidate = tuple(random_feasible_sequence(rng))
            new_population.append((candidate, tracker.evaluate(candidate)))

        new_population.sort(key=lambda item: item[1].objective, reverse=True)
        population = new_population[:population_size]

    return tracker.result()

