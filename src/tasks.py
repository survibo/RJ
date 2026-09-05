"""Task definitions.

모든 task 함수는 입력을 수정하지 않는 순수 함수다.

공통 인터페이스:
    target = TASK_REGISTRY[task](input_values, modulus=modulus)
"""

from typing import List, Optional, Sequence

import numpy as np


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


TASK_REGISTRY = {
    "ascending": ascending,
    "mod": mod_sort,
    "alternating": alternating,
    "alt_mod": alt_mod,
    "shift_mod": shift_mod,
    "shift_alt_mod": shift_alt_mod,
}

MODULUS_TASKS = frozenset({"mod", "alt_mod", "shift_mod", "shift_alt_mod"})
SHIFT_TASKS = frozenset({"shift_mod", "shift_alt_mod"})


def build_targets(task: str, inputs: np.ndarray, modulus: Optional[int] = None) -> np.ndarray:
    """inputs: [N, m] int array -> targets: [N, m] int array"""
    if task not in TASK_REGISTRY:
        raise ValueError(f"unknown task '{task}', expected one of {sorted(TASK_REGISTRY)}")
    fn = TASK_REGISTRY[task]
    out = np.empty_like(inputs)
    for i, row in enumerate(inputs):
        out[i] = fn(row.tolist(), modulus=modulus)
    return out
