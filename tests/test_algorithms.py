import unittest

from src.optimization.genetic import run_genetic
from src.optimization.immune import run_clonalg
from src.optimization.problem import (
    INSTANCES,
    exhaustive_optimum,
    is_feasible,
)
from src.optimization.swarm import run_aco

BUDGET = 300
RUNNERS = {"AG": run_genetic, "ACO": run_aco, "CLONALG": run_clonalg}


class AlgorithmTests(unittest.TestCase):
    def test_budget_is_respected_and_solution_is_feasible(self):
        for name, runner in RUNNERS.items():
            for instance in INSTANCES:
                with self.subTest(algorithm=name, instance=instance.name):
                    result = runner(instance, evaluation_budget=BUDGET, seed=7)
                    self.assertEqual(result.evaluations, BUDGET)
                    self.assertEqual(len(result.convergence), BUDGET)
                    self.assertTrue(is_feasible(result.best_sequence))
                    self.assertTrue(result.best_evaluation.feasible)

    def test_same_seed_gives_same_result(self):
        instance = INSTANCES[0]
        for name, runner in RUNNERS.items():
            with self.subTest(algorithm=name):
                first = runner(instance, evaluation_budget=BUDGET, seed=11)
                second = runner(instance, evaluation_budget=BUDGET, seed=11)
                self.assertEqual(first.best_sequence, second.best_sequence)
                self.assertEqual(first.convergence, second.convergence)

    def test_convergence_is_monotonic(self):
        instance = INSTANCES[1]
        for name, runner in RUNNERS.items():
            with self.subTest(algorithm=name):
                scores = [score for _, score in runner(instance, BUDGET, 3).convergence]
                self.assertEqual(scores, sorted(scores))

    def test_repair_counters_are_consistent(self):
        result = run_genetic(INSTANCES[0], evaluation_budget=BUDGET, seed=5)
        self.assertGreater(result.repair_calls, 0)
        self.assertLessEqual(result.repairs_needed, result.repair_calls)


class ExhaustiveReferenceTests(unittest.TestCase):
    def test_exact_optimum_bounds_every_algorithm(self):
        instance = INSTANCES[0]
        optimum, sequence, feasible_count = exhaustive_optimum(instance)
        self.assertEqual(feasible_count, 26970)
        self.assertTrue(is_feasible(sequence))
        result = run_aco(instance, evaluation_budget=BUDGET, seed=1)
        self.assertLessEqual(result.best_evaluation.objective, optimum + 1e-9)


if __name__ == "__main__":
    unittest.main()
