"""Task definitions.

모든 task 함수는 입력을 수정하지 않는 순수 함수다.

공통 인터페이스:
    target = TASK_REGISTRY[task](input_values, modulus=modulus)
"""

from typing import List, Optional, Sequence

import numpy as np


def ascending(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    return sorted(xs)


def mod_sort(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    if modulus is None:
        raise ValueError("task 'mod' requires --modulus")
    if modulus <= 0:
        raise ValueError(f"modulus must be positive, got {modulus}")
    return sorted(xs, key=lambda x: (x % modulus, x))


def alternating(xs: Sequence[int], modulus: Optional[int] = None) -> List[int]:
    """a0, a(m-1), a1, a(m-2), ... (정렬된 값 기준 바깥에서 안쪽으로)"""
    a = sorted(xs)
    out: List[int] = []
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        out.append(a[lo])
        if lo != hi:
            out.append(a[hi])
        lo += 1
        hi -= 1
    return out


TASK_REGISTRY = {
    "ascending": ascending,
    "mod": mod_sort,
    "alternating": alternating,
}


def build_targets(task: str, inputs: np.ndarray, modulus: Optional[int] = None) -> np.ndarray:
    """inputs: [N, m] int array -> targets: [N, m] int array"""
    if task not in TASK_REGISTRY:
        raise ValueError(f"unknown task '{task}', expected one of {sorted(TASK_REGISTRY)}")
    fn = TASK_REGISTRY[task]
    out = np.empty_like(inputs)
    for i, row in enumerate(inputs):
        out[i] = fn(row.tolist(), modulus=modulus)
    return out
