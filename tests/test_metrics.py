import unittest

import torch

from src.metrics import generation_metrics


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


if __name__ == "__main__":
    unittest.main()
