import unittest

import torch

from src.metrics import generation_metrics, symmetry_metrics, unique_frac


class FixedGenerationModel(torch.nn.Module):
    def __init__(self, generated, prefix_length, vocab_size):
        super().__init__()
        self.generated = generated
        self.prefix_length = prefix_length
        self.vocab_size = vocab_size

    def forward(self, tokens):
        logits = torch.zeros(
            tokens.shape[0], tokens.shape[1], self.vocab_size, device=tokens.device
        )
        step = tokens.shape[1] - self.prefix_length
        logits[:, -1, self.generated[step]] = 1.0
        return logits


class GenerationMetricTests(unittest.TestCase):
    def test_valid_accuracy_compares_target_multiset(self):
        m = 3
        seqs = torch.tensor([[6, 0, 3, 5, 7, 1, 1, 2]])
        model = FixedGenerationModel([1, 2, 1], prefix_length=m + 2, vocab_size=8)

        metrics = generation_metrics(model, seqs, m, batch_size=1, device=torch.device("cpu"))

        self.assertEqual(metrics["gen_exact_acc"], 0.0)
        self.assertEqual(metrics["gen_valid_acc"], 1.0)


class InputFnModel(torch.nn.Module):
    """입력 x1..xm 에 fn 을 적용한 결과를 차례로 생성하는 가짜 모델."""

    def __init__(self, fn, m, vocab_size):
        super().__init__()
        self.fn, self.m, self.vocab_size = fn, m, vocab_size

    def forward(self, tokens):
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab_size)
        step = tokens.shape[1] - (self.m + 2)
        out = self.fn(tokens[:, 1 : self.m + 1])
        logits[torch.arange(tokens.shape[0]), -1, out[:, step]] = 1.0
        return logits


class SymmetryMetricTests(unittest.TestCase):
    def setUp(self):
        self.m = 3
        self.seqs = torch.tensor([[6, 0, 3, 5, 7, 0, 0, 0], [6, 4, 1, 2, 7, 0, 0, 0]])
        self.perms = torch.tensor([[1, 2, 0], [2, 0, 1]])
        self.cpu = torch.device("cpu")

    def score(self, fn, kind):
        model = InputFnModel(fn, self.m, vocab_size=8)
        preds = generation_metrics(
            model, self.seqs, self.m, 2, self.cpu, return_preds=True
        )["preds"]
        return symmetry_metrics(model, self.seqs, preds, self.perms, kind, self.m, 2, self.cpu)

    def test_sorting_model_is_invariant(self):
        sort = lambda x: x.sort(dim=1).values
        self.assertEqual(self.score(sort, "invariant")["sym_exact_viol"], 0.0)
        self.assertEqual(self.score(sort, "equivariant")["sym_exact_viol"], 1.0)

    def test_copying_model_is_equivariant(self):
        copy = lambda x: x
        self.assertEqual(self.score(copy, "equivariant")["sym_token_viol"], 0.0)
        self.assertEqual(self.score(copy, "invariant")["sym_exact_viol"], 1.0)

    def test_unique_frac(self):
        preds = torch.tensor([[1, 2], [1, 2], [3, 4], [5, 6]])
        self.assertEqual(unique_frac(preds), 0.75)


if __name__ == "__main__":
    unittest.main()
