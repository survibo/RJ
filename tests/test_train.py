import unittest

from train import build_parser


class TrainCliTests(unittest.TestCase):
    def test_cuda_optimizations_are_enabled_by_default(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.precision, "auto")
        self.assertTrue(args.compile)
        self.assertTrue(args.fused_adamw)
        self.assertEqual(args.checkpoint_every, 2500)

    def test_cuda_optimizations_can_be_disabled(self):
        args = build_parser().parse_args(
            ["--precision", "fp32", "--no-compile", "--no-fused-adamw"]
        )

        self.assertEqual(args.precision, "fp32")
        self.assertFalse(args.compile)
        self.assertFalse(args.fused_adamw)


if __name__ == "__main__":
    unittest.main()
