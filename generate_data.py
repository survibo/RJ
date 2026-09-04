"""Dataset 생성 CLI.

예:
    python generate_data.py \
      --n 30 --m 5 --train-count 1000 --n-test 4096 \
      --split-strategy random --seed 42
"""

import argparse
import json
import os
import sys

from src.data import generate_dataset, load_meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sorting-grokking dataset generator")
    p.add_argument("--n", type=int, required=True, help="token 범위 [0, n)")
    p.add_argument("--m", type=int, required=True, help="한 sample 의 token 개수")
    p.add_argument("--train-count", type=int, required=True)
    p.add_argument("--n-test", type=int, required=True)
    p.add_argument(
        "--split-strategy", choices=["random", "relation-complete"], default="random"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="출력 디렉토리 (기본: data/...)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out_dir = generate_dataset(
            n=args.n,
            m=args.m,
            train_count=args.train_count,
            n_test=args.n_test,
            split_strategy=args.split_strategy,
            seed=args.seed,
            out_dir=args.out,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    meta = load_meta(out_dir)
    print(f"wrote {out_dir}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
