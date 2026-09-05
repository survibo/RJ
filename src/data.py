"""Dataset 생성 / 로딩 / sequence 구성."""

import itertools
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# 전수 열거 및 실제 생성 행 수의 독립적인 상한.
MAX_COMBINATIONS = 5_000_000
MAX_GENERATED_ROWS = 5_000_000


# ---------------------------------------------------------------- vocabulary

def bos_id(n: int) -> int:
    return n


def sep_id(n: int) -> int:
    return n + 1


def vocab_size(n: int) -> int:
    return n + 2


def seq_len(m: int) -> int:
    return 2 * m + 2


# ------------------------------------------------------------- combinations

def enumerate_combinations(n: int, m: int) -> np.ndarray:
    """[0, n) 에서 중복 없이 고른 m 개 combination 전수. shape [C(n,m), m], 오름차순."""
    total = math.comb(n, m)
    if total > MAX_COMBINATIONS:
        raise ValueError(
            f"C({n},{m}) = {total} exceeds MAX_COMBINATIONS = {MAX_COMBINATIONS}. "
            "exhaustive enumeration is not feasible. Use a random split to sample "
            "without enumerating the full combination space."
        )
    combos = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n), m)),
        dtype=np.int32,
        count=total * m,
    )
    return combos.reshape(total, m)


def unrank_combination(rank: int, n: int, m: int) -> List[int]:
    """Return the lexicographic combination at zero-based rank without enumeration."""
    total = math.comb(n, m)
    if rank < 0 or rank >= total:
        raise ValueError(f"combination rank must satisfy 0 <= rank < {total}, got {rank}")

    result: List[int] = []
    start = 0
    for i in range(m):
        slots = m - i
        prefix_total = math.comb(n - start, slots)
        lo, hi = start, n - slots
        while lo < hi:
            candidate = (lo + hi + 1) // 2
            skipped = prefix_total - math.comb(n - candidate, slots)
            if skipped <= rank:
                lo = candidate
            else:
                hi = candidate - 1
        skipped = prefix_total - math.comb(n - lo, slots)
        rank -= skipped
        result.append(lo)
        start = lo + 1
    return result


def sample_combinations(n: int, m: int, count: int, seed: int) -> np.ndarray:
    """Uniformly sample unique combinations without materializing C(n,m) rows."""
    total = math.comb(n, m)
    if count < 0 or count > total:
        raise ValueError(f"sample count must satisfy 0 <= count <= {total}, got {count}")

    # Floyd's algorithm is O(count), including when total exceeds int64.
    rng = random.Random(seed)
    selected = set()
    ranks: List[int] = []
    for upper in range(total - count, total):
        candidate = rng.randrange(upper + 1)
        rank = upper if candidate in selected else candidate
        selected.add(rank)
        ranks.append(rank)
    rng.shuffle(ranks)

    dtype = np.int32 if n - 1 <= np.iinfo(np.int32).max else np.int64
    return np.asarray(
        [unrank_combination(rank, n, m) for rank in ranks], dtype=dtype
    ).reshape(count, m)


def _pair_ids(combos: np.ndarray, n: int) -> np.ndarray:
    """각 combination 이 포함하는 unordered pair 의 index. shape [C, C(m,2)]"""
    m = combos.shape[1]
    cols = list(itertools.combinations(range(m), 2))
    out = np.empty((combos.shape[0], len(cols)), dtype=np.int64)
    for k, (a, b) in enumerate(cols):
        i = combos[:, a].astype(np.int64)
        j = combos[:, b].astype(np.int64)
        # combos 는 오름차순이므로 i < j
        out[:, k] = i * (2 * n - i - 1) // 2 + (j - i - 1)
    return out


# ------------------------------------------------------------------- splits

def random_split(
    combos: np.ndarray, train_count: int, n_test: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    order = rng.permutation(combos.shape[0])
    train_idx = order[:train_count]
    test_idx = order[train_count : train_count + n_test]
    return train_idx, test_idx, {}


def relation_complete_split(
    combos: np.ndarray, n: int, train_count: int, n_test: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """train set 이 모든 unordered pair 를 최소 한 번 포함하도록 greedy 구성."""
    m = combos.shape[1]
    n_pairs = math.comb(n, 2)
    per_combo = math.comb(m, 2)
    lower_bound = math.ceil(n_pairs / per_combo)
    if train_count < lower_bound:
        raise ValueError(
            f"train_count={train_count} < lower bound ceil(C({n},2)/C({m},2)) = {lower_bound}. "
            "relation-complete split is impossible."
        )

    order = rng.permutation(combos.shape[0])
    pair_ids = _pair_ids(combos, n)[order]  # order 기준으로 재배열

    covered = np.zeros(n_pairs, dtype=bool)
    alive = np.ones(order.shape[0], dtype=bool)
    picked: List[int] = []
    while not covered.all():
        gains = (~covered[pair_ids]).sum(axis=1)
        gains[~alive] = -1
        k = int(np.argmax(gains))
        if gains[k] <= 0:
            raise RuntimeError("greedy coverage stalled; cannot cover every pair.")
        covered[pair_ids[k]] = True
        alive[k] = False
        picked.append(k)
        if len(picked) > train_count:
            break

    basis_size = len(picked)
    if basis_size > train_count:
        raise ValueError(
            f"greedy relation basis size > train_count ({basis_size} > {train_count}). "
            f"raise --train-count to at least {basis_size}."
        )

    picked_arr = np.asarray(picked, dtype=np.int64)
    picked_mask = np.zeros(order.shape[0], dtype=bool)
    picked_mask[picked_arr] = True
    basis = order[picked_arr]
    rest = order[~picked_mask]  # seed 기반 random 순서 유지

    fill = train_count - basis_size
    if fill + n_test > rest.shape[0]:
        raise ValueError(
            f"train_count + n_test = {train_count + n_test} > C({n},{m}) = {combos.shape[0]}"
        )
    train_idx = np.concatenate([basis, rest[:fill]])
    test_idx = rest[fill : fill + n_test]
    return train_idx, test_idx, {"relation_basis_size": basis_size}


# ---------------------------------------------------------------- generation

def default_out_dir(
    n: int, m: int, train_count: int, n_test: int, split_strategy: str, seed: int
) -> str:
    return os.path.join(
        "data", f"n{n}_m{m}_tr{train_count}_te{n_test}_{split_strategy}_s{seed}"
    )


def generate_dataset(
    n: int,
    m: int,
    train_count: int,
    n_test: int,
    split_strategy: str,
    seed: int,
    out_dir: Optional[str] = None,
) -> str:
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if m <= 0:
        raise ValueError(f"m must be > 0, got {m}")
    if m > n:
        raise ValueError(f"m must be <= n, got m={m}, n={n}")
    if train_count <= 0:
        raise ValueError(f"train_count must be > 0, got {train_count}")
    if n_test < 0:
        raise ValueError(f"n_test must be >= 0, got {n_test}")
    total = math.comb(n, m)
    requested = train_count + n_test
    if requested > total:
        raise ValueError(
            f"train_count + n_test = {requested} > C({n},{m}) = {total}"
        )
    if requested > MAX_GENERATED_ROWS:
        raise ValueError(
            f"train_count + n_test = {requested} exceeds MAX_GENERATED_ROWS = "
            f"{MAX_GENERATED_ROWS}"
        )

    rng = np.random.default_rng(seed)

    if split_strategy == "random":
        if total <= MAX_COMBINATIONS:
            combos = enumerate_combinations(n, m)
            train_idx, test_idx, extra = random_split(
                combos, train_count, n_test, rng
            )
            train_combos = combos[train_idx]
            test_combos = combos[test_idx]
        else:
            combos = sample_combinations(n, m, requested, seed)
            train_combos = combos[:train_count]
            test_combos = combos[train_count:]
            extra = {}
    elif split_strategy == "relation-complete":
        combos = enumerate_combinations(n, m)
        train_idx, test_idx, extra = relation_complete_split(
            combos, n, train_count, n_test, rng
        )
        train_combos = combos[train_idx]
        test_combos = combos[test_idx]
    else:
        raise ValueError(f"unknown split strategy '{split_strategy}'")

    # 각 combination 은 한 번만 무작위 shuffle 해서 입력으로 저장한다.
    train_inputs = rng.permuted(train_combos, axis=1)
    test_inputs = rng.permuted(test_combos, axis=1)

    if out_dir is None:
        out_dir = default_out_dir(n, m, train_count, n_test, split_strategy, seed)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"[warn] {out_dir} already exists; overwriting "
              "(same seed reproduces the same content)")
    os.makedirs(out_dir, exist_ok=True)

    write_split(os.path.join(out_dir, "train.txt"), train_inputs)
    write_split(os.path.join(out_dir, "test.txt"), test_inputs)

    meta = {
        "n": n,
        "m": m,
        "train_size": int(train_inputs.shape[0]),
        "test_size": int(test_inputs.shape[0]),
        "split_strategy": split_strategy,
        "seed": seed,
    }
    meta.update(extra)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return out_dir


# ------------------------------------------------------------------------ io

def write_split(path: str, inputs: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in inputs:
            f.write(" ".join(str(int(v)) for v in row))
            f.write("\n")


def read_split(path: str, m: int) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [int(v) for v in line.split()]
            if len(vals) != m:
                raise ValueError(f"{path}: expected {m} values per line, got {len(vals)}")
            rows.append(vals)
    if not rows:
        return np.zeros((0, m), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def load_meta(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_sequences(inputs: np.ndarray, targets: np.ndarray, n: int) -> np.ndarray:
    """BOS x1..xm SEP y1..ym  ->  [N, 2m+2] int64"""
    N, m = inputs.shape
    seqs = np.empty((N, 2 * m + 2), dtype=np.int64)
    seqs[:, 0] = bos_id(n)
    seqs[:, 1 : m + 1] = inputs
    seqs[:, m + 1] = sep_id(n)
    seqs[:, m + 2 :] = targets
    return seqs
