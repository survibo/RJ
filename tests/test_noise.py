import json
import os
import tempfile
import unittest

import numpy as np
import torch

from src.metrics import noise_metrics
from src.noise import corrupt_targets, keep_rows, pick_rows


class NoiseTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.random.default_rng(0).integers(70, size=(1000, 7))

    def test_corrupts_exact_count_one_cell_per_row(self):
        out, rows, pos = corrupt_targets(self.targets, 0.1, 70, seed=3)

        self.assertEqual(len(rows), 100)
        diff = out != self.targets
        self.assertEqual(int(diff.sum()), 100)  # 행마다 정확히 한 칸, 반드시 다른 값
        self.assertTrue(diff[rows, pos].all())
        self.assertFalse(diff[np.setdiff1d(np.arange(1000), rows)].any())
        self.assertTrue(((out >= 0) & (out < 70)).all())

    def test_does_not_modify_input(self):
        before = self.targets.copy()
        corrupt_targets(self.targets, 0.2, 70, seed=1)
        np.testing.assert_array_equal(self.targets, before)

    def test_same_seed_same_rows_and_rows_are_nested(self):
        a = corrupt_targets(self.targets, 0.1, 70, seed=5)
        b = corrupt_targets(self.targets, 0.1, 70, seed=5)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        small = set(pick_rows(1000, 0.05, seed=5).tolist())
        large = set(pick_rows(1000, 0.10, seed=5).tolist())
        self.assertTrue(small <= large)

    def test_drop_removes_exactly_the_noisy_rows(self):
        _, rows, _ = corrupt_targets(self.targets, 0.2, 70, seed=7)
        keep = keep_rows(1000, 0.2, seed=7)

        self.assertEqual(len(keep), 800)
        self.assertEqual(set(keep.tolist()) | set(rows.tolist()), set(range(1000)))
        self.assertFalse(set(keep.tolist()) & set(rows.tolist()))

    def test_zero_frac_is_noop(self):
        out, rows, _ = corrupt_targets(self.targets, 0.0, 70, seed=0)
        self.assertEqual(len(rows), 0)
        np.testing.assert_array_equal(out, self.targets)

    def test_rejects_bad_frac(self):
        with self.assertRaises(ValueError):
            pick_rows(10, 1.0, seed=0)


class NoiseMetricTests(unittest.TestCase):
    def test_memorize_vs_rule(self):
        clean = torch.tensor([[1, 2, 3], [4, 5, 6]])
        wrong = torch.tensor([[1, 9, 3], [4, 5, 8]])
        pos = torch.tensor([1, 2])
        # 1행은 오답을 따라 하고, 2행은 규칙대로 답한다.
        preds = torch.tensor([[1, 9, 3], [4, 5, 6]])

        out = noise_metrics(preds, wrong, clean, pos)

        self.assertAlmostEqual(out["mem_exact"], 0.5)
        self.assertAlmostEqual(out["rule_exact"], 0.5)
        self.assertAlmostEqual(out["mem_tok"], 0.5)
        self.assertAlmostEqual(out["rule_tok"], 0.5)


class TrainNoiseSmokeTests(unittest.TestCase):
    """CPU 에서 몇 step 만 돌려 오답/삭제 run 이 끝까지 돌고 기록이 남는지 본다."""

    DATA = os.path.join("data", "n30_m5_tr1000_te4096_random_s42")

    def run_train(self, runs_dir, *extra):
        from train import main

        args = [
            "--data-dir", self.DATA, "--rules", "Q:first > R:add",
            "--n-embd", "32", "--n-head", "2", "--n-layer", "1",
            "--batch-size", "64", "--steps", "4", "--eval-every", "2",
            "--n-eval", "64", "--device", "cpu", "--runs-dir", runs_dir, *extra,
        ]
        self.assertEqual(main(args), 0)
        (run,) = os.listdir(runs_dir)
        return os.path.join(runs_dir, run)

    @unittest.skipUnless(os.path.isdir(DATA), "sample data not generated")
    def test_label_noise_run_logs_noisy_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_train(tmp, "--label-noise", "0.1", "--noise-seed", "3")
            self.assertIn("ln0.1-ns3", run)
            with open(os.path.join(run, "noise.json"), encoding="utf-8") as f:
                info = json.load(f)
            self.assertEqual(len(info["noisy_rows"]), 100)
            with open(os.path.join(run, "metrics.csv"), encoding="utf-8") as f:
                header, first = f.readline().strip().split(","), f.readline().strip().split(",")
            self.assertNotEqual(first[header.index("noisy_mem_exact")], "")

    @unittest.skipUnless(os.path.isdir(DATA), "sample data not generated")
    def test_drop_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = self.run_train(tmp, "--drop-frac", "0.2")
            with open(os.path.join(run, "noise.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["train_size"], 800)


if __name__ == "__main__":
    unittest.main()
