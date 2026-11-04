"""Algoritmo Genético próprio com codificação inteira por grau diatônico."""

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


def _tournament(
    population: list[tuple[tuple[int, ...], Evaluation]], rng: random.Random, size: int
) -> tuple[int, ...]:
    contestants = [rng.choice(population) for _ in range(min(size, len(population)))]
    return max(contestants, key=lambda item: item[1].objective)[0]


def _crossover(first: tuple[int, ...], second: tuple[int, ...], rng: random.Random) -> list[int]:
    cut = rng.randint(1, len(first) - 2)
    return [0, *first[1 : 1 + cut], *second[1 + cut : -1], 0]


def _mutate(sequence: list[int], rng: random.Random, rate: float) -> list[int]:
    mutated = list(sequence)
    for index in range(1, len(mutated) - 1):
        if rng.random() < rate:
            choices = [degree for degree in range(7) if degree != mutated[index]]
            mutated[index] = rng.choice(choices)
    return mutated


def run_genetic(
    instance: MusicInstance,
    evaluation_budget: int = 2000,
    seed: int = 42,
    population_size: int = 40,
    tournament_size: int = 3,
    elitism: int = 2,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.12,
) -> SearchResult:
    """Executa AG até consumir exatamente o orçamento de avaliações."""

    if population_size < 2:
        raise ValueError("A população do AG deve ter pelo menos 2 indivíduos.")
    if elitism < 0 or elitism >= population_size:
        raise ValueError("O elitismo deve estar entre 0 e population_size - 1.")
    if not 0.0 <= crossover_rate <= 1.0 or not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("As probabilidades devem estar no intervalo [0, 1].")

    rng = random.Random(seed)
    tracker = EvaluationBudget(instance, evaluation_budget)
    population: list[tuple[tuple[int, ...], Evaluation]] = []

    for _ in range(min(population_size, evaluation_budget)):
        candidate = tuple(random_feasible_sequence(rng))
        population.append((candidate, tracker.evaluate(candidate)))

    while tracker.evaluations < evaluation_budget:
        population.sort(key=lambda item: item[1].objective, reverse=True)
        survivor_count = min(elitism, len(population))
        next_population = list(population[:survivor_count])

        while len(next_population) < population_size and tracker.evaluations < evaluation_budget:
            parent_a = _tournament(population, rng, tournament_size)
            parent_b = _tournament(population, rng, tournament_size)
            if rng.random() < crossover_rate:
                child = _crossover(parent_a, parent_b, rng)
            else:
                child = list(parent_a)
            child = _mutate(child, rng, mutation_rate)
            child = repair_sequence(child, rng)
            evaluated = tracker.evaluate(child)
            next_population.append((tuple(child), evaluated))

        if not next_population:
            raise RuntimeError("O AG terminou uma geração sem indivíduos.")
        next_population.sort(key=lambda item: item[1].objective, reverse=True)
        population = next_population[:population_size]

    return tracker.result()

