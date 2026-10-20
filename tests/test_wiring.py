import unittest

from src.symmetry import describe
from src.wiring import make_wire_fn, parse_wiring, wiring_name

# RESEARCH_PLAN.md 실험 설계 1절 표: (배선, 대칭 유형, |G|)
PLAN_TABLE = [
    ("sortnbr", "invariant", 120),
    ("sum", "equivariant", 120),
    ("first", "equivariant", 24),
    ("t2", "equivariant", 12),
    ("t22", "equivariant", 8),
    ("c3", "equivariant", 6),
    ("c32", "equivariant", 6),
    ("nbr", "equivariant", 5),
    ("c5r", "equivariant", 5),
    ("c4", "equivariant", 4),
    ("nbrpos", "none", 1),
]


class WiringTests(unittest.TestCase):
    def test_values(self):
        xs = [3, 10, 7, 1, 20]
        add = lambda spec: make_wire_fn(spec, "add", 50, 5, 50)(xs)
        self.assertEqual(add("nbr"), [13, 17, 8, 21, 23])
        self.assertEqual(add("first"), [6, 13, 10, 4, 23])
        self.assertEqual(add("sum"), [44, 1, 48, 42, 11])  # Σ = 41
        self.assertEqual(add("sortnbr"), [4, 10, 17, 30, 21])  # s = 1 3 7 10 20
        self.assertEqual(add("nbrpos"), [13, 18, 10, 24, 27])
        self.assertEqual(add("t2"), [13, 13, 14, 2, 40])

    def test_table_op_is_deterministic_and_in_range(self):
        f = make_wire_fn("nbr", "table", 50, 5, 50, op_seed=3)
        g = make_wire_fn("nbr", "table", 50, 5, 50, op_seed=3)
        ys = f([3, 10, 7, 1, 20])
        self.assertEqual(ys, g([3, 10, 7, 1, 20]))
        self.assertTrue(all(0 <= y < 50 for y in ys))

    def test_symmetry_matches_plan_table(self):
        for op in ("add", "table"):
            for spec, kind, size in PLAN_TABLE:
                with self.subTest(op=op, spec=spec):
                    d = describe(make_wire_fn(spec, op, 50, 5, 50), 50, 5, n_samples=64)
                    self.assertEqual((d["kind"], d["group_size"]), (kind, size))

    def test_parse_rejects_bad_specs(self):
        for spec in ("nope", "perm:0,1,2", "perm:0,0,1,2,3", "perm:a,b"):
            with self.assertRaises(ValueError):
                parse_wiring(spec, 5)

    def test_wiring_name(self):
        self.assertEqual(wiring_name("perm:1,0,2,3,4"), "perm10234")
        self.assertEqual(wiring_name("c3"), "c3")


if __name__ == "__main__":
    unittest.main()
