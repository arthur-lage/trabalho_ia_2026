"""Formulação comum do problema 40: progressões de acordes sob regras."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import random
from typing import Sequence


SEQUENCE_LENGTH = 8
CHORD_DEGREES = 7
DOMINANT_DEGREE = 4  # V
MIN_DISTINCT_DEGREES = 3

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MODE_DATA = {
    "major": {
        "scale": (0, 2, 4, 5, 7, 9, 11),
        "qualities": ("", "m", "m", "", "", "m", "°"),
        "roman": ("I", "ii", "iii", "IV", "V", "vi", "vii°"),
        "cadence": {4: 1.0, 6: 0.95, 3: 0.78, 1: 0.65, 5: 0.48, 2: 0.3, 0: 0.15},
    },
    "harmonic_minor": {
        "scale": (0, 2, 3, 5, 7, 8, 11),
        "qualities": ("m", "°", "+", "m", "", "", "°"),
        "roman": ("i", "ii°", "III+", "iv", "V", "VI", "vii°"),
        "cadence": {4: 1.0, 6: 0.98, 3: 0.78, 1: 0.64, 5: 0.48, 2: 0.3, 0: 0.15},
    },
    "natural_minor": {
        "scale": (0, 2, 3, 5, 7, 8, 10),
        "qualities": ("m", "°", "", "m", "m", "", ""),
        "roman": ("i", "ii°", "III", "iv", "v", "VI", "VII"),
        "cadence": {6: 0.95, 4: 0.88, 3: 0.78, 1: 0.62, 5: 0.5, 2: 0.42, 0: 0.15},
    },
}


@dataclass(frozen=True)
class MusicInstance:
    """Uma tonalidade/modo fixa usada como instância experimental."""

    name: str
    tonic: str
    tonic_pitch: int
    mode: str

    @property
    def scale(self) -> tuple[int, ...]:
        return MODE_DATA[self.mode]["scale"]

    @property
    def chord_qualities(self) -> tuple[str, ...]:
        return MODE_DATA[self.mode]["qualities"]

    @property
    def roman_degrees(self) -> tuple[str, ...]:
        return MODE_DATA[self.mode]["roman"]


INSTANCES = (
    MusicInstance("C_major", "C", 0, "major"),
    MusicInstance("A_harmonic_minor", "A", 9, "harmonic_minor"),
    MusicInstance("A_natural_minor", "A", 9, "natural_minor"),
)


# Preferências harmônicas fixas. Transições não listadas continuam permitidas,
# mas recebem uma preferência menor. As mesmas tabelas são usadas por todos os
# métodos e também como heurística declarada pelo ACO.
TRANSITION_PREFERENCES = {
    "major": {
        0: {1: 0.78, 2: 0.32, 3: 0.92, 4: 1.00, 5: 0.72},
        1: {0: 0.38, 3: 0.55, 4: 1.00, 6: 0.88},
        2: {0: 0.28, 1: 0.48, 3: 0.72, 5: 0.92},
        3: {0: 0.82, 1: 0.82, 2: 0.52, 4: 0.98},
        4: {0: 1.00, 3: 0.48, 5: 0.72},
        5: {0: 0.52, 1: 0.88, 3: 0.78, 4: 0.76},
        6: {0: 0.98, 4: 0.84},
    },
    "harmonic_minor": {
        0: {2: 0.38, 3: 0.94, 4: 0.96, 5: 0.72},
        1: {0: 0.36, 3: 0.72, 4: 1.00},
        2: {0: 0.32, 3: 0.74, 5: 0.92},
        3: {0: 0.84, 1: 0.55, 4: 1.00},
        4: {0: 1.00, 3: 0.5, 5: 0.72},
        5: {0: 0.55, 1: 0.78, 3: 0.84},
        6: {0: 1.00, 3: 0.62, 4: 0.84},
    },
    "natural_minor": {
        0: {2: 0.58, 3: 0.94, 4: 0.78, 5: 0.78, 6: 0.86},
        1: {3: 0.8, 4: 0.9, 0: 0.4},
        2: {5: 0.94, 3: 0.72, 6: 0.82, 0: 0.32},
        3: {0: 0.88, 4: 0.92, 5: 0.68, 1: 0.55},
        4: {0: 0.96, 5: 0.72, 2: 0.52},
        5: {3: 0.88, 1: 0.76, 6: 0.78, 0: 0.54},
        6: {0: 0.92, 2: 0.84, 4: 0.72},
    },
}
DEFAULT_TRANSITION_PREFERENCE = 0.12


@dataclass(frozen=True)
class Evaluation:
    """Resultado objetivo e diagnóstico de viabilidade de uma progressão."""

    objective: float
    conformity: float
    diversity: float
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    """Saída comum dos otimizadores, com convergência por avaliação."""

    best_sequence: tuple[int, ...]
    best_evaluation: Evaluation
    evaluations: int
    convergence: tuple[tuple[int, float], ...]


class EvaluationBudget:
    """Conta cada chamada à função objetivo e mantém o melhor candidato."""

    def __init__(self, instance: MusicInstance, limit: int):
        if limit < 1:
            raise ValueError("O orçamento de avaliações deve ser positivo.")
        self.instance = instance
        self.limit = limit
        self.evaluations = 0
        self.best_sequence: tuple[int, ...] | None = None
        self.best_evaluation: Evaluation | None = None
        self.convergence: list[tuple[int, float]] = []

    def evaluate(self, sequence: Sequence[int]) -> Evaluation:
        if self.evaluations >= self.limit:
            raise RuntimeError("O algoritmo tentou exceder o orçamento de avaliações.")
        result = evaluate_sequence(self.instance, sequence)
        self.evaluations += 1
        if result.feasible and (
            self.best_evaluation is None or result.objective > self.best_evaluation.objective
        ):
            self.best_sequence = tuple(int(value) for value in sequence)
            self.best_evaluation = result
        best_score = self.best_evaluation.objective if self.best_evaluation else float("-inf")
        self.convergence.append((self.evaluations, best_score))
        return result

    def result(self) -> SearchResult:
        if self.best_sequence is None or self.best_evaluation is None:
            raise RuntimeError("Nenhuma solução viável foi avaliada.")
        return SearchResult(
            self.best_sequence,
            self.best_evaluation,
            self.evaluations,
            tuple(self.convergence),
        )


def _check_representation(sequence: Sequence[int]) -> tuple[int, ...]:
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(f"A solução deve conter exatamente {SEQUENCE_LENGTH} acordes.")
    values: list[int] = []
    for value in sequence:
        if not isinstance(value, Integral):
            raise ValueError("Cada acorde deve ser codificado por um inteiro de 0 a 6.")
        degree = int(value)
        if not 0 <= degree < CHORD_DEGREES:
            raise ValueError("Cada acorde deve ser codificado por um inteiro de 0 a 6.")
        values.append(degree)
    return tuple(values)


def feasibility_violations(sequence: Sequence[int]) -> tuple[str, ...]:
    """Retorna restrições violadas sem incorporar penalidade ao objetivo."""

    values = _check_representation(sequence)
    violations: list[str] = []
    if values[0] != 0:
        violations.append("A progressão deve começar na tônica (I/i).")
    if values[-1] != 0:
        violations.append("A progressão deve terminar na tônica (I/i).")
    if DOMINANT_DEGREE not in values[1:-1]:
        violations.append("A progressão deve incluir a dominante (V) no interior.")
    if len(set(values)) < MIN_DISTINCT_DEGREES:
        violations.append(f"A progressão deve usar ao menos {MIN_DISTINCT_DEGREES} graus distintos.")
    if any(left == right for left, right in zip(values, values[1:])):
        violations.append("Não são permitidas repetições de acorde em posições adjacentes.")
    return tuple(violations)


def is_feasible(sequence: Sequence[int]) -> bool:
    return not feasibility_violations(sequence)


def evaluate_sequence(instance: MusicInstance, sequence: Sequence[int]) -> Evaluation:
    """Maximiza conformidade harmônica e diversidade em escala 0–100.

    O custo não contém penalidade: candidatos que violam restrições continuam
    recebendo seu valor objetivo original e são marcados como inviáveis.
    """

    values = _check_representation(sequence)
    preferences = TRANSITION_PREFERENCES[instance.mode]
    transition_scores = [
        preferences[left].get(right, DEFAULT_TRANSITION_PREFERENCE)
        for left, right in zip(values, values[1:])
    ]
    transition_mean = sum(transition_scores) / len(transition_scores)
    cadence = MODE_DATA[instance.mode]["cadence"][values[-2]]
    conformity = 0.85 * transition_mean + 0.15 * cadence
    diversity = len(set(values)) / CHORD_DEGREES
    objective = 100.0 * (0.80 * conformity + 0.20 * diversity)
    violations = feasibility_violations(values)
    return Evaluation(objective, conformity, diversity, not violations, violations)


def decode_sequence(instance: MusicInstance, sequence: Sequence[int]) -> tuple[str, ...]:
    """Converte graus internos em símbolos de acordes transpostos à tonalidade."""

    values = _check_representation(sequence)
    mode_data = MODE_DATA[instance.mode]
    chords: list[str] = []
    for degree in values:
        pitch = (instance.tonic_pitch + mode_data["scale"][degree]) % 12
        chords.append(PITCH_NAMES[pitch] + mode_data["qualities"][degree])
    return tuple(chords)


def repair_sequence(sequence: Sequence[int], rng: random.Random) -> list[int]:
    """Repara as restrições comuns preservando, sempre que possível, os genes."""

    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(f"A solução deve conter exatamente {SEQUENCE_LENGTH} acordes.")
    if any(not isinstance(value, Integral) for value in sequence):
        raise ValueError("O reparo recebe uma sequência de inteiros.")

    repaired = [int(value) % CHORD_DEGREES for value in sequence]
    repaired[0] = 0
    repaired[-1] = 0

    # Remove repetições adjacentes sem alterar os extremos fixos.
    for index in range(1, SEQUENCE_LENGTH - 1):
        forbidden = {repaired[index - 1]}
        if index == SEQUENCE_LENGTH - 2:
            forbidden.add(0)
        if repaired[index] in forbidden:
            if index < SEQUENCE_LENGTH - 2:
                forbidden.add(repaired[index + 1])
            options = [degree for degree in range(CHORD_DEGREES) if degree not in forbidden]
            repaired[index] = rng.choice(options)

    # Garante a dominante em uma posição que não crie repetição adjacente.
    if DOMINANT_DEGREE not in repaired[1:-1]:
        positions = [
            index
            for index in range(1, SEQUENCE_LENGTH - 1)
            if repaired[index - 1] != DOMINANT_DEGREE
            and repaired[index + 1] != DOMINANT_DEGREE
        ]
        if positions:
            repaired[rng.choice(positions)] = DOMINANT_DEGREE

    # Garante diversidade mínima adicionando um terceiro grau sem conflito local.
    if len(set(repaired)) < MIN_DISTINCT_DEGREES:
        existing = set(repaired)
        alternatives = [degree for degree in range(CHORD_DEGREES) if degree not in existing]
        placements = [
            (index, degree)
            for index in range(1, SEQUENCE_LENGTH - 1)
            for degree in alternatives
            if repaired[index - 1] != degree and repaired[index + 1] != degree
        ]
        if placements:
            index, degree = rng.choice(placements)
            repaired[index] = degree

    if not is_feasible(repaired):
        # Fallback válido fixo para qualquer caso limite; o espaço continua
        # diversificado pelas demais posições da população e pelas mutações.
        repaired = [0, 1, 4, 5, 3, 4, 1, 0]
    return repaired


def random_feasible_sequence(rng: random.Random) -> list[int]:
    candidate = [0] + [rng.randrange(CHORD_DEGREES) for _ in range(SEQUENCE_LENGTH - 2)] + [0]
    return repair_sequence(candidate, rng)


def validate_domain() -> dict[str, str]:
    """Checagens locais exigidas: avaliador, restrição, reparo e caso controlado."""

    known = [0, 3, 4, 0, 5, 3, 4, 0]  # I–IV–V–I–vi–IV–V–I
    invalid = [0] * SEQUENCE_LENGTH
    if not is_feasible(known):
        raise AssertionError("A progressão de referência deveria ser viável.")
    if is_feasible(invalid):
        raise AssertionError("A sequência sem dominante deve ser inviável.")

    rng = random.Random(0)
    repaired = repair_sequence(invalid, rng)
    if not is_feasible(repaired):
        raise AssertionError("O reparador não recuperou uma solução viável.")

    expected = {
        "C_major": ("C", "F", "G", "C", "A m", "F", "G", "C"),
        "A_harmonic_minor": ("A m", "D m", "E", "A m", "F", "D m", "E", "A m"),
        "A_natural_minor": ("A m", "D m", "E m", "A m", "F", "D m", "E m", "A m"),
    }
    for instance in INSTANCES:
        decoded = decode_sequence(instance, known)
        formatted = tuple(value.replace("m", " m") for value in decoded)
        if formatted != expected[instance.name]:
            raise AssertionError(f"Decodificação incorreta na instância {instance.name}.")
        evaluation = evaluate_sequence(instance, known)
        if not 0.0 <= evaluation.objective <= 100.0:
            raise AssertionError("O objetivo saiu do intervalo declarado [0, 100].")

    return {
        "known_solution": "viável e decodificada nas três instâncias",
        "invalid_solution": "detectada pelas restrições musicais",
        "repair": "converteu a solução inválida em solução viável",
        "objective": "conformidade e diversidade na escala 0–100",
    }
