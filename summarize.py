"""run 디렉토리들을 훑어 grokking 지표를 한 표로 모은다 (RESEARCH_PLAN.md grokking 지표).

    t_fit : train 생성 정확도(수열 전체 일치)가 --fit 에 처음 도달한 step
    t_gen : test 생성 정확도가 --gen 에 처음 도달한 step (예산 안에 못 가면 censored)
    t_sym : test 대칭 위반 점수가 --sym 이하이고 출력 다양성이 --uniq 이상인 첫 step
            (학습 초반 상수 출력은 위반 점수가 0 이라 다양성 조건으로 걸러낸다)
    delay = t_gen - t_fit, lead = t_gen - t_sym
    t_mem : 오답 행에서 오답을 그대로 따라 하는 비율이 --mem 에 처음 도달한 step (Plan.md)
    t_flip: t_mem 이후, 오답 행에서 규칙대로 답하는 비율이 오답 따라 하기보다 처음 커진 step

예:
    python summarize.py runs_pilot --out runs_pilot/summary.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional

FIELDS = [
    "run",
    "task",
    "rules",
    "label_noise",
    "drop_frac",
    "wiring",
    "op",
    "prefix_lm",
    "input_pos",
    "n_layer",
    "n_embd",
    "train_size",
    "seed",
    "sym_kind",
    "group_size",
    "dependency_range",
    "t_fit",
    "t_gen",
    "t_sym",
    "delay",
    "delay_ratio",
    "lead",
    "censored",
    "t_mem",
    "t_flip",
    "final_noisy_mem",
    "final_noisy_rule",
    "last_step",
    "final_test_acc",
    "wall_time",
]


def _float(v: str) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def first_step(rows: List[Dict], cond) -> Optional[int]:
    for r in rows:
        if cond(r):
            return int(r["step"])
    return None


def summarize_run(
    run_dir: str, fit: float, gen: float, sym: float, uniq: float, mem: float = 0.5
) -> Optional[Dict]:
    metrics = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(metrics):
        return None
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sym_info = {}
    sym_path = os.path.join(run_dir, "symmetry.json")
    if os.path.isfile(sym_path):
        with open(sym_path, encoding="utf-8") as f:
            sym_info = json.load(f)
    with open(metrics, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def val(r, k):
        return _float(r.get(k, ""))

    t_fit = first_step(rows, lambda r: (val(r, "train_gen_exact_acc") or 0) >= fit)
    t_gen = first_step(rows, lambda r: (val(r, "test_gen_exact_acc") or 0) >= gen)
    t_sym = first_step(
        rows,
        lambda r: val(r, "test_sym_exact_viol") is not None
        and val(r, "test_sym_exact_viol") <= sym
        and (val(r, "test_pred_unique_frac") or 0) >= uniq,
    )
    t_mem = first_step(rows, lambda r: (val(r, "noisy_mem_exact") or 0) >= mem)
    t_flip = None
    if t_mem is not None:
        t_flip = first_step(
            [r for r in rows if int(r["step"]) > t_mem],
            lambda r: (val(r, "noisy_rule_exact") or 0) > (val(r, "noisy_mem_exact") or 0),
        )
    out = {
        "run": os.path.basename(run_dir),
        "task": cfg["task"],
        "rules": cfg.get("rules") or "",
        "label_noise": cfg.get("label_noise") or 0,
        "drop_frac": cfg.get("drop_frac") or 0,
        "wiring": cfg.get("wiring") or "",
        "op": cfg.get("op") or "",
        "prefix_lm": cfg.get("prefix_lm", False),
        "input_pos": cfg.get("input_pos", True),
        "n_layer": cfg["n_layer"],
        "n_embd": cfg["n_embd"],
        "train_size": cfg["train_size"],
        "seed": cfg["seed"],
        "sym_kind": sym_info.get("kind", ""),
        "group_size": sym_info.get("group_size", ""),
        "dependency_range": round(sym_info["dependency_range"], 3)
        if "dependency_range" in sym_info
        else "",
        "t_fit": t_fit,
        "t_gen": t_gen,
        "t_sym": t_sym,
        "delay": t_gen - t_fit if t_gen is not None and t_fit is not None else None,
        "delay_ratio": round(t_gen / t_fit, 3) if t_gen is not None and t_fit else None,
        "lead": t_gen - t_sym if t_gen is not None and t_sym is not None else None,
        "censored": t_gen is None,
        "t_mem": t_mem,
        "t_flip": t_flip,
        "final_noisy_mem": val(rows[-1], "noisy_mem_exact"),
        "final_noisy_rule": val(rows[-1], "noisy_rule_exact"),
        "last_step": int(rows[-1]["step"]),
        "final_test_acc": val(rows[-1], "test_gen_exact_acc"),
        "wall_time": round(val(rows[-1], "wall_time") or 0),
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="grokking 지표 요약")
    p.add_argument("runs_dirs", nargs="+", help="run 디렉토리들의 부모 (하위까지 재귀 탐색)")
    p.add_argument("--fit", type=float, default=0.99)
    p.add_argument("--gen", type=float, default=0.95)
    p.add_argument("--sym", type=float, default=0.05)
    p.add_argument("--uniq", type=float, default=0.5)
    p.add_argument("--mem", type=float, default=0.5, help="t_mem 기준 (오답 따라 하기 비율)")
    p.add_argument("--out", type=str, default=None, help="CSV 경로 (없으면 표만 출력)")
    args = p.parse_args(argv)

    results = []
    for root in args.runs_dirs:
        for cfg_path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            r = summarize_run(
                os.path.dirname(cfg_path), args.fit, args.gen, args.sym, args.uniq, args.mem
            )
            if r is not None:
                results.append(r)
    if not results:
        print("no runs found", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in results:
                w.writerow({k: "" if r[k] is None else r[k] for k in FIELDS})
        print(f"wrote {args.out} ({len(results)} runs)")

    if any(r["rules"] for r in results):
        cols = ["rules", "label_noise", "drop_frac", "seed", "t_fit", "t_gen", "t_mem",
                "t_flip", "final_test_acc", "final_noisy_mem", "final_noisy_rule", "wall_time"]
    else:
        cols = ["wiring", "op", "n_layer", "n_embd", "train_size", "seed", "group_size",
                "t_fit", "t_gen", "t_sym", "delay", "lead", "final_test_acc", "wall_time"]
    print("\t".join(cols))
    for r in results:
        print("\t".join("-" if r[c] is None else str(r[c]) for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
