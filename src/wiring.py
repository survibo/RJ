"""배선 규칙 가족 (RESEARCH_PLAN.md 실험 설계 1절).

모든 규칙은 y_i = T(x_i, 참조값_i) 꼴이다. 연산 T 는 고정하고
각 칸이 무엇을 참조하는지(배선)만 바꾼다. token 공간(0..n-1)에서 계산하고 출력은 [0, p).

배선 spec:
    sortnbr        정렬 후 오른쪽 이웃 (순환): T(s_i, s_{i+1}), s = sorted(x)
    sum            전체 합: T(x_i, Σx mod n)
    first          첫 칸: T(x_i, x_0)
    nbr            오른쪽 이웃 (순환): T(x_i, x_{i+1})
    nbrpos         오른쪽 이웃 + 위치 번호: T(x_i, x_{i+1}) + i
    perm:a,b,...   고정 배선 π: T(x_i, x_{π(i)}), π(i) 는 0-based
    t2, t22, c3, c32, c4, c5r   m=5 용 π preset (WIRING_PRESETS)

연산:
    add     T(a, b) = (a + b) mod p
    table   T(a, b) = M[a][b], M 은 op_seed 로 뽑은 [n, n] 무작위 표 (값 [0, p))
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

WIRE_KINDS = ("sortnbr", "sum", "first", "nbr", "nbrpos")

# m=5 전용 π preset. 대칭군 크기 = π 의 중심화군 크기 (주석 값은 tests/test_wiring.py 가 확인).
WIRING_PRESETS = {
    "t2": "perm:1,0,2,3,4",  # (0 1)          |G| = 12
    "t22": "perm:1,0,3,2,4",  # (0 1)(2 3)     |G| = 8
    "c3": "perm:1,2,0,3,4",  # (0 1 2)        |G| = 6
    "c32": "perm:1,2,0,4,3",  # (0 1 2)(3 4)   |G| = 6
    "c4": "perm:1,2,3,0,4",  # (0 1 2 3)      |G| = 4
    "c5r": "perm:3,0,4,2,1",  # (0 3 2 4 1)    |G| = 5, 이웃이 아닌 5-순환
}

OPS = ("add", "table")


def parse_wiring(spec: str, m: int) -> Tuple[str, Optional[Tuple[int, ...]]]:
    """spec -> (kind, perm). kind 는 WIRE_KINDS 또는 'perm'."""
    resolved = WIRING_PRESETS.get(spec, spec)
    if resolved in WIRE_KINDS:
        return resolved, None
    if not resolved.startswith("perm:"):
        raise ValueError(
            f"unknown wiring '{spec}', expected one of "
            f"{list(WIRE_KINDS) + sorted(WIRING_PRESETS)} or perm:a,b,..."
        )
    try:
        perm = tuple(int(v) for v in resolved[len("perm:"):].split(","))
    except ValueError:
        raise ValueError(f"wiring '{spec}': perm entries must be integers")
    if sorted(perm) != list(range(m)):
        raise ValueError(f"wiring '{spec}': perm must be a permutation of 0..{m - 1}")
    return "perm", perm


def wiring_name(spec: str) -> str:
    """run 이름용. 'perm:1,0,2,3,4' -> 'perm10234'"""
    return spec.replace(":", "").replace(",", "")


def make_op(op: str, n: int, p: int, op_seed: int = 0) -> Callable[[int, int], int]:
    if p <= 0:
        raise ValueError(f"modulus must be positive, got {p}")
    if op == "add":
        return lambda a, b: (a + b) % p
    if op == "table":
        table = np.random.default_rng(op_seed).integers(0, p, size=(n, n))
        return lambda a, b: int(table[a, b])
    raise ValueError(f"unknown op '{op}', expected one of {list(OPS)}")


def make_wire_fn(
    spec: str, op: str, n: int, m: int, p: int, op_seed: int = 0
) -> Callable[[Sequence[int]], List[int]]:
    """token 수열 (0..n-1) -> 출력 token 수열 (0..p-1)"""
    kind, perm = parse_wiring(spec, m)
    T = make_op(op, n, p, op_seed)

    def apply(xs: Sequence[int]) -> List[int]:
        xs = [int(v) for v in xs]
        if len(xs) != m:
            raise ValueError(f"expected {m} values, got {len(xs)}")
        if kind == "sortnbr":
            s = sorted(xs)
            return [T(s[i], s[(i + 1) % m]) for i in range(m)]
        if kind == "sum":
            total = sum(xs) % n
            return [T(x, total) for x in xs]
        if kind == "first":
            return [T(x, xs[0]) for x in xs]
        if kind == "nbr":
            return [T(xs[i], xs[(i + 1) % m]) for i in range(m)]
        if kind == "nbrpos":
            return [(T(xs[i], xs[(i + 1) % m]) + i) % p for i in range(m)]
        return [T(xs[i], xs[perm[i]]) for i in range(m)]

    return apply
