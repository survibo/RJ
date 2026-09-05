import itertools
import json
import math
import os
import tempfile
import unittest

import numpy as np

from src.data import (
    MAX_GENERATED_ROWS,
    generate_dataset,
    read_split,
    sample_combinations,
    unrank_combination,
)


class CombinationSamplingTests(unittest.TestCase):
    def test_unrank_matches_lexicographic_enumeration(self):
        expected = list(itertools.combinations(range(6), 3))
        actual = [tuple(unrank_combination(i, 6, 3)) for i in range(len(expected))]
        self.assertEqual(actual, expected)

    def test_large_space_sampling_is_unique_and_reproducible(self):
        self.assertGreater(math.comb(100, 20), np.iinfo(np.int64).max)
        first = sample_combinations(100, 20, 100, seed=7)
        second = sample_combinations(100, 20, 100, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len({tuple(row) for row in first}), 100)
        self.assertTrue(np.all(first[:, 1:] > first[:, :-1]))

    def test_random_dataset_does_not_enumerate_large_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = generate_dataset(
                n=50,
                m=10,
                train_count=20,
                n_test=10,
                split_strategy="random",
                seed=3,
                out_dir=tmp,
            )
            train = read_split(os.path.join(out, "train.txt"), 10)
            test = read_split(os.path.join(out, "test.txt"), 10)
            with open(os.path.join(out, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)

        self.assertEqual(train.shape, (20, 10))
        self.assertEqual(test.shape, (10, 10))
        self.assertEqual(meta["train_size"], 20)
        train_sets = {tuple(sorted(row)) for row in train}
        test_sets = {tuple(sorted(row)) for row in test}
        self.assertTrue(train_sets.isdisjoint(test_sets))

    def test_generated_row_limit_is_separate_from_combination_count(self):
        with self.assertRaisesRegex(ValueError, "MAX_GENERATED_ROWS"):
            generate_dataset(
                n=100,
                m=5,
                train_count=MAX_GENERATED_ROWS + 1,
                n_test=0,
                split_strategy="random",
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
