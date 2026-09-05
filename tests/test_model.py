import unittest

import torch

from src.model import GPT, GPTConfig


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = GPT(
            GPTConfig(
                vocab_size=12,
                block_size=8,
                n_embd=8,
                n_head=2,
                n_layer=1,
                dropout=0.0,
            )
        ).eval()

    def test_future_token_does_not_change_earlier_logits(self):
        first = torch.tensor([[10, 1, 4, 7, 11, 2, 3, 4]])
        second = first.clone()
        second[:, -1] = 9

        with torch.no_grad():
            first_logits = self.model(first)
            second_logits = self.model(second)

        torch.testing.assert_close(first_logits[:, :-1], second_logits[:, :-1])

    def test_token_embedding_and_lm_head_are_tied(self):
        self.assertIs(self.model.wte.weight, self.model.lm_head.weight)

    def test_rejects_sequence_longer_than_block_size(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1, 9), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
