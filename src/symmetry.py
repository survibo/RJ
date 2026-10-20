"""규칙의 대칭군 / 의존 범위 계산 (RESEARCH_PLAN.md 핵심 개념 정의).

순열은 index 배열 p 로 나타낸다: 섞인 입력은 x[p] (새 i번째 칸 = 원래 p[i]번째 칸).
    불변: f(x[p]) == f(x)
    등변: f(x[p]) == f(x)[p]
순열 m! 개를 모두 검사하고, 입력은 n_samples 개를 무작위로 뽑아 확인한다.
"""

import itertools
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Fn = Callable[[Sequence[int]], List[int]]


def sample_inputs(n: int, m: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(n, size=m, replace=False) for _ in range(count)])


def symmetry_groups(
    fn: Fn, n: int, m: int, n_samples: int = 256, seed: int = 0
) -> Dict[str, List[Tuple[int, ...]]]:
    """{'invariant': [p, ...], 'equivariant': [p, ...]} (둘 다 항등 순열 포함)"""
    xs = sample_inputs(n, m, n_samples, seed)
    ys = [list(fn(x.tolist())) for x in xs]
    invariant, equivariant = [], []
    for p in itertools.permutations(range(m)):
        idx = list(p)
        inv = eqv = True
        for x, y in zip(xs, ys):
            out = list(fn(x[idx].tolist()))
            inv = inv and out == y
            eqv = eqv and out == [y[j] for j in idx]
            if not (inv or eqv):
                break
        if inv:
            invariant.append(p)
        if eqv:
            equivariant.append(p)
    return {"invariant": invariant, "equivariant": equivariant}


def classify(groups: Dict[str, List[Tuple[int, ...]]]) -> Tuple[str, List[Tuple[int, ...]]]:
    """더 큰 쪽을 규칙의 대칭으로 본다. 둘 다 항등뿐이면 ('none', [항등])."""
    inv, eqv = groups["invariant"], groups["equivariant"]
    if len(inv) <= 1 and len(eqv) <= 1:
        return "none", eqv or inv
    if len(inv) >= len(eqv):
        return "invariant", inv
    return "equivariant", eqv


def dependency_range(fn: Fn, n: int, m: int, n_samples: int = 256, seed: int = 0) -> float:
    """입력 한 칸을 안 쓰인 다른 값으로 바꿨을 때 바뀌는 출력 칸 수의 평균."""
    rng = np.random.default_rng(seed)
    xs = sample_inputs(n, m, n_samples, seed)
    changed = 0
    for x in xs:
        y = fn(x.tolist())
        unused = np.setdiff1d(np.arange(n), x)
        for i in range(m):
            x2 = x.copy()
            x2[i] = rng.choice(unused)
            y2 = fn(x2.tolist())
            changed += sum(a != b for a, b in zip(y, y2))
    return changed / (n_samples * m)


def describe(fn: Fn, n: int, m: int, n_samples: int = 256, seed: int = 0) -> Dict:
    groups = symmetry_groups(fn, n, m, n_samples, seed)
    kind, group = classify(groups)
    return {
        "kind": kind,
        "group_size": len(group),
        "group": [list(p) for p in group],
        "invariant_size": len(groups["invariant"]),
        "equivariant_size": len(groups["equivariant"]),
        "dependency_range": dependency_range(fn, n, m, n_samples, seed),
    }
