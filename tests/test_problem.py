import random
import unittest

from src.optimization.problem import (
    INSTANCES,
    decode_sequence,
    evaluate_sequence,
    feasibility_violations,
    is_feasible,
    repair_sequence,
    validate_domain,
)


class Problem40Tests(unittest.TestCase):
    KNOWN_SEQUENCE = (0, 3, 4, 0, 5, 3, 4, 0)
    CONTROLLED_SEQUENCE = (0, 5, 3, 4, 0, 5, 3, 0)
    INVALID_SEQUENCE = (0, 0, 0, 0, 0, 0, 0, 0)
    EXPECTED_SCORES = {
        "C_major": 85.40571428571429,
        "A_harmonic_minor": 86.57142857142857,
        "A_natural_minor": 83.77142857142857,
    }

    def test_known_progression_is_feasible_and_repeats_are_allowed(self):
        self.assertTrue(is_feasible(self.KNOWN_SEQUENCE))
        self.assertEqual(feasibility_violations(self.KNOWN_SEQUENCE), ())
        self.assertLess(len(set(self.KNOWN_SEQUENCE)), len(self.KNOWN_SEQUENCE))

    def test_controlled_progression_is_feasible(self):
        self.assertTrue(is_feasible(self.CONTROLLED_SEQUENCE))

    def test_known_objective_matches_hand_calculation_for_each_instance(self):
        for instance in INSTANCES:
            with self.subTest(instance=instance.name):
                evaluation = evaluate_sequence(instance, self.KNOWN_SEQUENCE)
                self.assertTrue(evaluation.feasible)
                self.assertEqual(evaluation.violations, ())
                self.assertAlmostEqual(
                    evaluation.objective,
                    self.EXPECTED_SCORES[instance.name],
                    places=10,
                )

    def test_invalid_progression_reports_violations_and_repair_recovers_it(self):
        evaluation = evaluate_sequence(INSTANCES[0], self.INVALID_SEQUENCE)
        self.assertFalse(evaluation.feasible)
        self.assertTrue(evaluation.violations)

        repaired = repair_sequence(list(self.INVALID_SEQUENCE), random.Random(0))
        self.assertTrue(is_feasible(repaired))

    def test_decoding_known_progression_in_c_major(self):
        self.assertEqual(
            decode_sequence(INSTANCES[0], self.KNOWN_SEQUENCE),
            ("C", "F", "G", "C", "Am", "F", "G", "C"),
        )

    def test_domain_validation_reports_controlled_checks(self):
        result = validate_domain()
        self.assertIn("known_solution", result)
        self.assertIn("controlled_solution", result)
        self.assertIn("invalid_solution", result)
        self.assertIn("repair", result)


if __name__ == "__main__":
    unittest.main()
