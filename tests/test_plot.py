import csv
import os
import tempfile
import unittest

from plot import compact_run_labels, main, read_metrics


class PlotTests(unittest.TestCase):
    def test_compacts_common_run_suffix(self):
        labels, context = compact_run_labels(
            [
                "ascending_random_n30m5_tr1000_s42",
                "shift_mod_random_n30m5_tr1000_s42",
            ]
        )
        self.assertEqual(labels, ["ascending", "shift_mod"])
        self.assertEqual(context, "random / n30m5 / tr1000 / s42")

    def test_reads_invalid_and_missing_values_as_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss"])
                writer.writerow([0, ""])
                writer.writerow([1, "bad"])
            cols = read_metrics(path)
            self.assertEqual(cols["step"], [0.0, 1.0])
            self.assertTrue(all(value != value for value in cols["train_loss"]))

    def test_generates_plot_from_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for task in ("ascending", "shift_mod"):
                run_dir = os.path.join(tmp, f"{task}_random_n10m3_tr60_s1")
                os.makedirs(run_dir)
                path = os.path.join(run_dir, "metrics.csv")
                paths.append(path)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "step",
                            "train_gen_exact_acc",
                            "test_gen_exact_acc",
                            "test_gen_valid_acc",
                            "train_loss",
                            "test_loss",
                            "param_norm",
                            "embd_norm",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "step": 1,
                            "train_gen_exact_acc": 0.5,
                            "test_gen_exact_acc": 0.2,
                            "test_gen_valid_acc": 0.6,
                            "train_loss": 1.0,
                            "test_loss": 1.4,
                            "param_norm": 10.0,
                            "embd_norm": 2.0,
                        }
                    )
            output = os.path.join(tmp, "plot.png")
            self.assertEqual(main([*paths, "--out", output]), 0)
            self.assertGreater(os.path.getsize(output), 0)


if __name__ == "__main__":
    unittest.main()
