"""Protocolo reproduzível de comparação para os métodos da Parte 2."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .genetic import run_genetic
from .immune import run_clonalg
from .problem import (
    INSTANCES,
    EvaluationBudget,
    MusicInstance,
    SearchResult,
    decode_sequence,
    random_feasible_sequence,
    validate_domain,
)
from .swarm import run_aco


DEFAULT_RUNS = 10
DEFAULT_BUDGET = 2000
DEFAULT_SEED = 42
ALGORITHM_NAMES = ("AG", "ACO", "CLONALG", "Busca aleatória")


def _run_random_search(
    instance: MusicInstance, evaluation_budget: int, seed: int
) -> SearchResult:
    rng = random.Random(seed)
    tracker = EvaluationBudget(instance, evaluation_budget)
    while tracker.evaluations < evaluation_budget:
        tracker.evaluate(random_feasible_sequence(rng))
    return tracker.result()


def _run_method(
    algorithm: str, instance: MusicInstance, budget: int, seed: int
) -> SearchResult:
    if algorithm == "AG":
        return run_genetic(instance, evaluation_budget=budget, seed=seed)
    if algorithm == "ACO":
        return run_aco(instance, evaluation_budget=budget, seed=seed)
    if algorithm == "CLONALG":
        return run_clonalg(instance, evaluation_budget=budget, seed=seed)
    if algorithm == "Busca aleatória":
        return _run_random_search(instance, budget, seed)
    raise ValueError(f"Algoritmo desconhecido: {algorithm}")


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (instance, algorithm), group in runs.groupby(["instance", "algorithm"], sort=False):
        scores = group["objective"]
        records.append(
            {
                "instance": instance,
                "algorithm": algorithm,
                "best_objective": float(scores.max()),
                "mean_objective": float(scores.mean()),
                "std_objective": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                "feasible_runs": int(group["feasible"].sum()),
                "feasibility_rate_percent": float(100.0 * group["feasible"].mean()),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                "std_runtime_seconds": float(group["runtime_seconds"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "mean_objective_evaluations": float(group["objective_evaluations"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _aggregate_curves(
    histories: dict[tuple[str, str], list[tuple[tuple[int, float], ...]]],
    budget: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (instance, algorithm), runs in histories.items():
        values = np.asarray([[score for _, score in run] for run in runs], dtype=float)
        means = values.mean(axis=0)
        for evaluation, mean_score in enumerate(means, start=1):
            records.append(
                {
                    "instance": instance,
                    "algorithm": algorithm,
                    "objective_evaluations": evaluation,
                    "mean_best_objective": float(mean_score),
                }
            )
        if len(means) != budget:
            raise RuntimeError("A curva não corresponde ao orçamento de avaliações informado.")
    return pd.DataFrame.from_records(records)


def _plot_convergence(curves: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(INSTANCES), 1, figsize=(11, 14), sharex=True, sharey=True, constrained_layout=True
    )
    colors = {"AG": "#2563eb", "ACO": "#16a34a", "CLONALG": "#9333ea", "Busca aleatória": "#64748b"}
    for axis, instance in zip(axes, INSTANCES):
        subset = curves[curves["instance"] == instance.name]
        for algorithm in ALGORITHM_NAMES:
            method = subset[subset["algorithm"] == algorithm]
            axis.plot(
                method["objective_evaluations"],
                method["mean_best_objective"],
                label=algorithm,
                color=colors[algorithm],
                linewidth=1.8,
            )
        axis.set_title(instance.name.replace("_", " "))
        axis.set_ylabel("Melhor objetivo médio (0–100)")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", ncol=2)
    axes[-1].set_xlabel("Avaliações da função objetivo")
    fig.suptitle("Convergência média por orçamento de avaliações", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_experiments(
    runs_per_method: int = DEFAULT_RUNS,
    evaluation_budget: int = DEFAULT_BUDGET,
    seed_base: int = DEFAULT_SEED,
    output_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if runs_per_method < 2:
        raise ValueError("Use pelo menos 2 execuções independentes por método e instância.")
    if evaluation_budget < 1:
        raise ValueError("O orçamento de avaliações deve ser positivo.")
    if output_root is None:
        output_root = Path(__file__).resolve().parents[2] / "results"

    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_domain()
    run_records: list[dict[str, object]] = []
    histories: dict[tuple[str, str], list[tuple[tuple[int, float], ...]]] = {}

    for instance in INSTANCES:
        for algorithm in ALGORITHM_NAMES:
            key = (instance.name, algorithm)
            histories[key] = []
            for run_index in range(runs_per_method):
                seed = seed_base + run_index
                start = perf_counter()
                result = _run_method(algorithm, instance, evaluation_budget, seed)
                runtime = perf_counter() - start
                if result.evaluations != evaluation_budget:
                    raise RuntimeError(
                        f"{algorithm} consumiu {result.evaluations} de {evaluation_budget} avaliações."
                    )
                if not result.best_evaluation.feasible:
                    raise RuntimeError(f"{algorithm} retornou uma solução inviável.")

                histories[key].append(result.convergence)
                run_records.append(
                    {
                        "instance": instance.name,
                        "mode": instance.mode,
                        "algorithm": algorithm,
                        "run": run_index + 1,
                        "seed": seed,
                        "objective": result.best_evaluation.objective,
                        "conformity": result.best_evaluation.conformity,
                        "diversity": result.best_evaluation.diversity,
                        "feasible": result.best_evaluation.feasible,
                        "objective_evaluations": result.evaluations,
                        "runtime_seconds": runtime,
                        "degree_sequence": json.dumps(result.best_sequence),
                        "chord_progression": json.dumps(
                            decode_sequence(instance, result.best_sequence), ensure_ascii=False
                        ),
                    }
                )
                print(
                    f"{instance.name:20s} | {algorithm:16s} | execução {run_index + 1:02d}/"
                    f"{runs_per_method} | objetivo {result.best_evaluation.objective:6.2f} | "
                    f"{runtime:7.3f}s"
                )

    runs_frame = pd.DataFrame.from_records(run_records)
    summary_frame = _summarize(runs_frame)
    curves_frame = _aggregate_curves(histories, evaluation_budget)

    runs_frame.to_csv(tables_dir / "optimization_runs.csv", index=False)
    summary_frame.to_csv(tables_dir / "optimization_summary.csv", index=False)
    curves_frame.to_csv(tables_dir / "optimization_convergence.csv", index=False)
    _plot_convergence(curves_frame, figures_dir / "optimization_convergence.png")

    metadata = {
        "problem": 40,
        "problem_name": "Geração de escalas musicais ou progressões de acordes sob regras",
        "implemented_variant": "progressões de acordes diatônicos de oito posições",
        "runs_per_method_instance": runs_per_method,
        "objective_evaluation_budget_per_run": evaluation_budget,
        "seeds": list(range(seed_base, seed_base + runs_per_method)),
        "instances": [instance.name for instance in INSTANCES],
        "methods": {
            "AG": {
                "population_size": 40,
                "tournament_size": 3,
                "elitism": 2,
                "crossover_rate": 0.9,
                "mutation_rate_per_gene": 0.12,
            },
            "ACO": {
                "ants_per_update": 24,
                "elite_ants": 4,
                "alpha": 1.0,
                "beta": 2.0,
                "evaporation_rate": 0.2,
            },
            "CLONALG": {
                "population_size": 30,
                "selected_antibodies": 10,
                "clone_factor": 6,
                "adaptive_mutation_rate": "0.05 + 0.45 * (1 - affinity)",
                "random_replacements_per_round": 4,
            },
        },
        "reference": "busca aleatória de soluções viáveis com o mesmo orçamento",
        "validation_checks": validation,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or "não informado pelo sistema",
            "logical_cpu_count": __import__("os").cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (tables_dir / "optimization_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    print("\nResumo comparativo:")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nTabelas: {tables_dir}")
    print(f"Figura: {figures_dir / 'optimization_convergence.png'}")
    return runs_frame, summary_frame, curves_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Execuções independentes por método/instância.")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="Avaliações da função objetivo por execução.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Primeira semente da sequência registrada.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Pasta-base dos resultados (padrão: results/).")
    parser.add_argument("--validate-only", action="store_true", help="Valida o avaliador, as restrições e o reparo e encerra.")
    args = parser.parse_args()

    if args.validate_only:
        print(json.dumps(validate_domain(), ensure_ascii=False, indent=2))
        return
    run_experiments(args.runs, args.budget, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
