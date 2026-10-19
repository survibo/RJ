import json
import os
import unittest

import numpy as np

from src.rules import (
    format_rules,
    make_step,
    parse_rules,
    rules_hash,
    run_rules,
)
from src.tasks import build_rule_targets, build_targets

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "rule_lab_golden.json")


class RuleLabParityTests(unittest.TestCase):
    """rule-lab.html 의 RuleCore 결과와 완전히 같은지 확인한다 (tests/make_rule_golden.js)."""

    def test_matches_rule_lab(self):
        with open(GOLDEN, "r", encoding="utf-8") as f:
            cases = json.load(f)
        for i, case in enumerate(cases):
            steps = [make_step(s["code"], s["mode"], s) for s in case["steps"]]
            with self.subTest(case=i, rules=format_rules(steps), values=case["values"]):
                if case.get("error"):
                    with self.assertRaises(ValueError):
                        run_rules(case["values"], steps, case["n"])
                else:
                    self.assertEqual(run_rules(case["values"], steps, case["n"]), case["final"])

    def test_copied_spec_parses_to_same_rules(self):
        # rule-lab 의 "명령어 복사" 문자열이 train.py --rules 에서 같은 조합이 되는지
        with open(GOLDEN, "r", encoding="utf-8") as f:
            cases = json.load(f)
        for i, case in enumerate(cases):
            steps = [make_step(s["code"], s["mode"], s) for s in case["steps"]]
            with self.subTest(case=i, spec=case["spec"]):
                self.assertEqual(case["spec"], format_rules(steps))
                self.assertEqual(format_rules(parse_rules(case["spec"])), case["spec"])

    def test_golden_covers_every_rule(self):
        from src.rules import RULE_MODES

        with open(GOLDEN, "r", encoding="utf-8") as f:
            cases = json.load(f)
        seen = {s["code"] for c in cases if not c.get("error") for s in c["steps"]}
        self.assertEqual(seen, set(RULE_MODES))


class RuleSpecTests(unittest.TestCase):
    def test_defaults_are_filled_and_canonical(self):
        steps = parse_rules("q > i:c=3 > B:left,k=8 > R")
        self.assertEqual(
            format_rules(steps), "Q:first > I:add,c=3 > B:left,k=8 > R:add"
        )
        self.assertEqual(format_rules(parse_rules(format_rules(steps))), format_rules(steps))
        self.assertEqual(rules_hash(steps), rules_hash(parse_rules(format_rules(steps))))

    def test_condition_and_period_params(self):
        steps = parse_rules("AJ:threshold,t=5,fn=pow,fc=2 > AO:mul,period=2/3 > Z:odd,fn=mod")
        self.assertEqual(
            format_rules(steps),
            "AJ:threshold,t=5,fn=pow,fc=2 > AO:mul,period=2/3 > Z:odd,fn=mod",
        )

    def test_invalid_specs(self):
        for spec in (
            "",
            "Q >",
            "NOPE",
            "A:sideways",
            "I:add,k=3",  # I 는 k 를 쓰지 않는다
            "C:keep,c=1",
            "I:add,c=0",
            "B:left,k=-1",
            "Z:even,fn=cube",
            "I:add,c",
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_rules(spec)

    def test_runtime_errors(self):
        with self.assertRaises(ValueError):
            run_rules([1, 2], parse_rules("R:add"), 5)
        with self.assertRaises(ValueError):
            run_rules([1, 2], parse_rules("Q:kth,k=3"), 5)


class RuleExecutionTests(unittest.TestCase):
    def test_rule_lab_demo(self):
        # rule-lab 예시 조합: n=50, [7, 2, 49, 13, 25, 1]
        steps = parse_rules("Q:first > I:add,c=3 > B:left,k=8 > R:add")
        # Q 가 7 을 저장, +3, 왼쪽 2칸, +7 -> [59, 23, 35, 11, 17, 12]
        # 마지막 ((x-1) mod 50)+1 은 50 을 넘은 59 만 9 로 감는다.
        self.assertEqual(
            run_rules([7, 2, 49, 13, 25, 1], steps, 50), [9, 23, 35, 11, 17, 12]
        )

    def test_separate_final_modulus(self):
        steps = parse_rules("A:asc")
        self.assertEqual(run_rules([9, 3, 7], steps, 10, modulus=4), [3, 3, 1])

    def test_values_in_range_are_unchanged(self):
        # 1..n 안의 값은 마지막 모듈러와 M 을 거쳐도 그대로다.
        values = [7, 1, 10, 4]
        self.assertEqual(run_rules(values, parse_rules("C:keep"), 10), values)
        self.assertEqual(run_rules(values, parse_rules("M:mod"), 10), values)
        self.assertEqual(run_rules(values, parse_rules("I:add,c=10"), 10), values)

    def test_build_rule_targets_maps_tokens(self):
        # token 0..n-1 -> 값 1..n, 결과 값 1..p -> token 0..p-1
        # 정렬만 하면 기존 ascending task 와 같고, p < n 이면 token mod p 가 된다.
        inputs = np.asarray([[2, 0, 1], [4, 3, 0]], dtype=np.int64)
        steps = parse_rules("A:asc")
        np.testing.assert_array_equal(
            build_rule_targets(steps, inputs, n=5), build_targets("ascending", inputs)
        )
        np.testing.assert_array_equal(
            build_rule_targets(steps, inputs, n=5, modulus=3), [[0, 1, 2], [0, 0, 1]]
        )


if __name__ == "__main__":
    unittest.main()
