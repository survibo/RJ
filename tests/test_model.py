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


class PrefixLMTests(unittest.TestCase):
    # m = 3: BOS x1 x2 x3 SEP y1 y2 y3 -> prefix_len 5
    def make(self, input_pos):
        torch.manual_seed(1)
        cfg = GPTConfig(
            vocab_size=12, block_size=8, n_embd=8, n_head=2, n_layer=2,
            prefix_len=5, input_pos=input_pos,
        )
        return GPT(cfg).eval()

    def test_inputs_attend_bidirectionally_outputs_stay_causal(self):
        model = self.make(input_pos=True)
        seq = torch.tensor([[10, 1, 4, 7, 11, 2, 3, 4]])
        later_input = seq.clone()
        later_input[:, 3] = 9
        later_output = seq.clone()
        later_output[:, -1] = 9
        with torch.no_grad():
            base = model(seq)
            self.assertFalse(torch.allclose(base[:, 1], model(later_input)[:, 1]))
            torch.testing.assert_close(base[:, :-1], model(later_output)[:, :-1])

    def test_no_input_pos_is_invariant_to_input_order(self):
        model = self.make(input_pos=False)
        seq = torch.tensor([[10, 1, 4, 7, 11, 2, 3, 4]])
        shuffled = seq[:, [0, 3, 1, 2, 4, 5, 6, 7]]
        with torch.no_grad():
            torch.testing.assert_close(model(seq)[:, 4:], model(shuffled)[:, 4:])

    def test_no_input_pos_requires_prefix_lm(self):
        with self.assertRaises(ValueError):
            GPT(GPTConfig(vocab_size=12, block_size=8, input_pos=False))


if __name__ == "__main__":
    unittest.main()
