import unittest

import numpy as np

from src.tasks import (
    TASK_REGISTRY,
    alt_mod,
    build_targets,
    shift_alt_mod,
    shift_mod,
)


class TaskTests(unittest.TestCase):
    def test_alt_mod_alternates_modulo_rank_ends(self):
        self.assertEqual(alt_mod([1, 2, 3, 4, 5], modulus=3), [3, 5, 1, 2, 4])

    def test_shift_mod_uses_first_input_as_offset(self):
        self.assertEqual(shift_mod([2, 3, 5], modulus=6), [1, 4, 5])
        self.assertEqual(shift_mod([3, 2, 5], modulus=6), [0, 2, 5])

    def test_shift_alt_mod_alternates_transformed_values(self):
        self.assertEqual(shift_alt_mod([2, 3, 5], modulus=6), [1, 5, 4])

    def test_shift_mod_preserves_duplicate_transformed_tokens(self):
        self.assertEqual(shift_mod([3, 0, 6], modulus=6), [0, 3, 3])

    def test_tasks_do_not_mutate_input(self):
        values = [5, 1, 4, 2, 3]
        original = values.copy()
        for task in TASK_REGISTRY.values():
            task(values, modulus=3)
            self.assertEqual(values, original)

    def test_modulus_validation(self):
        for task in (alt_mod, shift_mod, shift_alt_mod):
            with self.assertRaises(ValueError):
                task([1, 2], modulus=None)
            with self.assertRaises(ValueError):
                task([1, 2], modulus=0)
        with self.assertRaises(ValueError):
            shift_mod([], modulus=3)

    def test_build_targets_supports_shift_tasks(self):
        inputs = np.asarray([[2, 3, 5], [3, 2, 5]], dtype=np.int64)
        expected = np.asarray([[1, 4, 5], [0, 2, 5]], dtype=np.int64)
        np.testing.assert_array_equal(
            build_targets("shift_mod", inputs, modulus=6), expected
        )


if __name__ == "__main__":
    unittest.main()
