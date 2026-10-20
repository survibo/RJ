"""Task definitions.

모든 task 함수는 입력을 수정하지 않는 순수 함수다.

공통 인터페이스:
    target = TASK_REGISTRY[task](input_values, modulus=modulus)

규칙 조합 task (src/rules.py):
    targets = build_rule_targets(parse_rules(spec), inputs, n, modulus)

배선 규칙 task (src/wiring.py):
    targets = build_fn_targets(make_wire_fn(spec, op, n, m, modulus), inputs)
"""

from typing import Callable, List, Optional, Sequence

import numpy as np

from src.rules import Step, parse_rules, run_rules
from src.wiring import make_wire_fn


def _alternate(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    lo, hi = 0, len(values) - 1
    while lo <= hi:
        out.append(values[lo])
        if lo != hi:
            out.append(values[hi])
        lo += 1
        hi -= 1
    return out


def _require_modulus(modulus: Optional[int]) -> int:
    if modulus is None:
        raise ValueError("task requires --modulus")
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    return modulus


def ascending(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    return sorted(xs)


def mod_sort(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    modulus = _require_modulus(modulus)
    return sorted(xs, key=lambda x: (x % modulus, x))


def alternating(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    """a0, a(m-1), a1, a(m-2), ... (정렬된 값 기준 바깥에서 안쪽으로)"""
    return _alternate(sorted(xs))


def alt_mod(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    """(x % modulus, x) 순위의 최솟값과 최댓값부터 교번 출력한다."""
    modulus = _require_modulus(modulus)
    return _alternate(sorted(xs, key=lambda x: (x % modulus, x)))


def _shifted_values(xs: Sequence[int], modulus: Optional[int]) -> List[int]:
    modulus = _require_modulus(modulus)
    if not xs:
        raise ValueError("shifted-mod tasks require at least one input value")
    offset = xs[0]
    return [(x + offset) % modulus for x in xs]


def shift_mod(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    """첫 입력값을 offset으로 더한 나머지 token들을 오름차순 출력한다."""
    return sorted(_shifted_values(xs, modulus))


def shift_alt_mod(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    """첫 입력값을 offset으로 더한 나머지 token들을 교번 출력한다."""
    return _alternate(sorted(_shifted_values(xs, modulus)))


def sum_shift(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    """전체 입력의 합을 offset으로 더한 나머지를 입력 순서 그대로 출력한다."""
    modulus = _require_modulus(modulus)
    if not xs:
        raise ValueError("sum_shift requires at least one input value")
    offset = sum(xs)
    return [(x + offset) % modulus for x in xs]


TASK_REGISTRY = {
    "ascending": ascending,
    "mod": mod_sort,
    "alternating": alternating,
    "alt_mod": alt_mod,
    "shift_mod": shift_mod,
    "shift_alt_mod": shift_alt_mod,
    "sum_shift": sum_shift,
}

MODULUS_TASKS = frozenset({"mod", "alt_mod", "shift_mod", "shift_alt_mod", "sum_shift"})
# 출력이 [0, modulus) 값이라 modulus <= n 이어야 하는 task
SHIFT_TASKS = frozenset({"shift_mod", "shift_alt_mod", "sum_shift"})
# --rules 로 만든 규칙 조합 task 의 이름. 출력은 [0, modulus) 라 modulus <= n.
RULES_TASK = "rules"
# --wiring 으로 만든 배선 규칙 task 의 이름. 출력은 [0, modulus) 라 modulus <= n.
WIRE_TASK = "wire"


def build_rule_targets(
    steps: Sequence[Step], inputs: np.ndarray, n: int, modulus: Optional[int] = None
) -> np.ndarray:
    """규칙 조합 task. inputs: [N, m] token(0..n-1) -> targets: [N, m] token(0..p-1)

    rule-lab 은 값 1..n 을 쓰므로 값 = token + 1 로 계산하고 결과 값 1..p 를 token 0..p-1 로 돌린다.
    마지막이 ((x-1) mod p)+1 이라 규칙이 값을 바꾸지 않으면 token 도 (p = n 일 때) 그대로다.
    """
    out = np.empty_like(inputs)
    for i, row in enumerate(inputs):
        values = [int(v) + 1 for v in row]
        try:
            ys = run_rules(values, steps, n, modulus)
        except ValueError as e:
            raise ValueError(f"input {values}: {e}") from e
        out[i] = [y - 1 for y in ys]
    return out


def make_target_fn(
    task: str,
    n: int,
    m: int,
    modulus: Optional[int] = None,
    rules: Optional[str] = None,
    wiring: Optional[str] = None,
    op: str = "add",
    op_seed: int = 0,
) -> Callable[[Sequence[int]], List[int]]:
    """모든 task 를 token 수열 (0..n-1) -> target token 수열 함수로 통일한다 (대칭 계산용)."""
    if task == WIRE_TASK:
        return make_wire_fn(wiring, op, n, m, modulus, op_seed)
    if task == RULES_TASK:
        steps = parse_rules(rules)
        return lambda xs: [y - 1 for y in run_rules([int(v) + 1 for v in xs], steps, n, modulus)]
    if task not in TASK_REGISTRY:
        raise ValueError(f"unknown task '{task}', expected one of {sorted(TASK_REGISTRY)}")
    fn = TASK_REGISTRY[task]
    return lambda xs: fn([int(v) for v in xs], modulus=modulus)


def build_fn_targets(fn: Callable[[Sequence[int]], List[int]], inputs: np.ndarray) -> np.ndarray:
    out = np.empty_like(inputs)
    for i, row in enumerate(inputs):
        try:
            out[i] = fn(row.tolist())
        except ValueError as e:
            raise ValueError(f"input {row.tolist()}: {e}") from e
    return out


def build_targets(task: str, inputs: np.ndarray, modulus: Optional[int] = None) -> np.ndarray:
    """inputs: [N, m] int array -> targets: [N, m] int array"""
    if task not in TASK_REGISTRY:
        raise ValueError(f"unknown task '{task}', expected one of {sorted(TASK_REGISTRY)}")
    fn = TASK_REGISTRY[task]
    out = np.empty_like(inputs)
    for i, row in enumerate(inputs):
        out[i] = fn(row.tolist(), modulus=modulus)
    return out
