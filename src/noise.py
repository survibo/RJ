"""오답 섞기 (label noise) / 삭제 대조군.

같은 (frac, seed) 면 오답으로 바꿀 행과 삭제할 행이 정확히 같다.
비율이 커져도 앞쪽 행은 그대로라서 5% 의 오답 행은 10% 의 오답 행에 포함된다.
"""

from typing import Tuple

import numpy as np


def pick_rows(size: int, frac: float, seed: int) -> np.ndarray:
    """[0, size) 에서 round(frac * size) 개 행을 고른다 (오름차순)."""
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"frac must satisfy 0 <= frac < 1, got {frac}")
    k = int(round(frac * size))
    order = np.random.default_rng(seed).permutation(size)
    return np.sort(order[:k]).astype(np.int64)


def corrupt_targets(
    targets: np.ndarray, frac: float, out_range: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """고른 행마다 출력 칸 하나를 정답과 다른 token (0..out_range-1) 으로 바꾼다.

    돌려주는 값: (바뀐 targets, 바뀐 행 index, 행마다 바뀐 칸 위치)
    """
    if out_range < 2:
        raise ValueError(f"out_range must be >= 2, got {out_range}")
    rows = pick_rows(targets.shape[0], frac, seed)
    rng = np.random.default_rng(seed + 1)
    pos = rng.integers(targets.shape[1], size=rows.shape[0]).astype(np.int64)
    # 0 이 아닌 offset 을 더해서 원래 값과 반드시 다르게 만든다.
    offset = rng.integers(1, out_range, size=rows.shape[0])
    out = targets.copy()
    out[rows, pos] = (targets[rows, pos] + offset) % out_range
    return out, rows, pos


def keep_rows(size: int, frac: float, seed: int) -> np.ndarray:
    """삭제 대조군: corrupt_targets 가 바꿀 행을 빼고 남은 행 index."""
    mask = np.ones(size, dtype=bool)
    mask[pick_rows(size, frac, seed)] = False
    return np.flatnonzero(mask).astype(np.int64)
