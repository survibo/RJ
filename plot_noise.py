"""오답 섞기 실험 (Plan.md) 그래프 3종.

    1. grok_vs_noise.png   : 오답/삭제 비율 -> 이해 시점 (과제별 칸, seed 마다 점 + 평균 선)
    2. noisy_dynamics.png  : 오답 행에서 오답 따라 하기 vs 규칙대로 답하기 (과제별 칸, --noise 하나)
    3. test_curves.png     : 오답 비율별 test 정확도 곡선 (과제별 칸, --seed 하나)

예:
    python plot_noise.py runs_noise/main --out-dir runs_noise/plots
"""

import argparse
import glob
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot import read_metrics  # noqa: E402
from summarize import summarize_run  # noqa: E402

INK = "#2b2b2a"
MUTED = "#6b6a64"
GRID = "#e4e3de"
NOISE_COLOR = "#2a78d6"  # 오답 (categorical 1)
DROP_COLOR = "#eb6834"  # 삭제 대조군 (categorical 2)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # 오답 비율 (ordinal)


def load_runs(roots: List[str], gen: float, mem: float) -> List[Dict]:
    runs = []
    for root in roots:
        for cfg_path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            run_dir = os.path.dirname(cfg_path)
            r = summarize_run(run_dir, fit=0.99, gen=gen, sym=0.05, uniq=0.5, mem=mem)
            if r is None or not r["rules"]:
                continue
            r["dir"] = run_dir
            r["label_noise"] = float(r["label_noise"])
            r["drop_frac"] = float(r["drop_frac"])
            runs.append(r)
    return runs


def by_task(runs: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        out[r["rules"]].append(r)
    return dict(out)


def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def panels(n: int, width: float = 4.2, height: float = 3.4):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    return fig, list(axes[0])


def finish(fig, axes, out: str, title: str, ncol: int, legend_title: Optional[str] = None) -> None:
    """범례는 그림 아래 한 줄로 (데이터를 가리지 않게)."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=8,
               title=legend_title, title_fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------- 그래프 1

def plot_grok_vs_noise(tasks: Dict[str, List[Dict]], out: str, gen: float) -> None:
    fig, axes = panels(len(tasks))
    for ax, (rules, runs) in zip(axes, tasks.items()):
        budget = max(r["last_step"] for r in runs)
        top = budget * 1.08  # 이해 못 한 run 을 그리는 줄
        for kind, color, key, marker in (
            ("label noise", NOISE_COLOR, "label_noise", "o"),
            ("drop (control)", DROP_COLOR, "drop_frac", "s"),
        ):
            # 0% 기준점은 두 조건이 같다: 오답 0% run 을 삭제 선의 시작점으로도 쓴다.
            pts = [r for r in runs if r[key] > 0 or (r["label_noise"] == 0 and r["drop_frac"] == 0)]
            if not pts:
                continue
            xs_line, ys_line = [], []
            for frac in sorted({r[key] for r in pts}):
                group = [r for r in pts if r[key] == frac]
                ok = [r["t_gen"] for r in group if r["t_gen"] is not None]
                for r in group:
                    censored = r["t_gen"] is None
                    ax.scatter(
                        frac * 100, top if censored else r["t_gen"],
                        s=36, marker="x" if censored else marker, color=color,
                        alpha=0.85, linewidths=1.5 if censored else 0.8,
                        edgecolors="white" if not censored else None, zorder=3,
                    )
                # 평균 선은 모든 seed 가 이해한 비율에서만 잇는다.
                if ok and len(ok) == len(group):
                    xs_line.append(frac * 100)
                    ys_line.append(mean(ok))
            ax.plot(xs_line, ys_line, color=color, linewidth=2, label=kind, zorder=2)
        ax.axhline(budget, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(1.0, budget, "step budget ", transform=ax.get_yaxis_transform(),
                fontsize=7, color=MUTED, ha="right", va="bottom")
        ax.set_ylim(0, top * 1.06)
        ax.set_xlim(-2, 42)
        ax.set_xlabel("fraction of train rows (%)", fontsize=9, color=INK)
        ax.set_ylabel(f"steps to test acc {gen:.0%}", fontsize=9, color=INK)
        style_axes(ax, rules)
    axes[0].scatter([], [], marker="x", color=MUTED, label="not reached within budget")
    finish(fig, axes, out, "When does the model understand the rule?", ncol=3)


# --------------------------------------------------------------- 그래프 2

def plot_noisy_dynamics(tasks: Dict[str, List[Dict]], out: str, noise: float, seed: int) -> None:
    fig, axes = panels(len(tasks))
    for ax, (rules, runs) in zip(axes, tasks.items()):
        match = [r for r in runs if r["label_noise"] == noise and r["seed"] == seed]
        style_axes(ax, f"{rules}  (noise {noise:.0%}, seed {seed})")
        if not match:
            ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        cols = read_metrics(os.path.join(match[0]["dir"], "metrics.csv"))
        steps = cols["step"]
        ax.plot(steps, cols["test_gen_exact_acc"], color=MUTED, linewidth=1.2,
                linestyle="--", label="test acc")
        ax.plot(steps, cols.get("noisy_mem_exact", []), color=DROP_COLOR, linewidth=2,
                label="noisy rows: copies wrong answer")
        ax.plot(steps, cols.get("noisy_rule_exact", []), color=NOISE_COLOR, linewidth=2,
                label="noisy rows: follows rule")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("step", fontsize=9, color=INK)
        ax.set_ylabel("fraction", fontsize=9, color=INK)
    finish(fig, axes, out, "Memorize the wrong answer, or follow the rule?", ncol=3)


# --------------------------------------------------------------- 그래프 3

def plot_test_curves(tasks: Dict[str, List[Dict]], out: str, seed: int) -> None:
    fig, axes = panels(len(tasks))
    for ax, (rules, runs) in zip(axes, tasks.items()):
        noisy = sorted(
            (r for r in runs if r["drop_frac"] == 0 and r["seed"] == seed),
            key=lambda r: r["label_noise"],
        )
        for i, r in enumerate(noisy):
            cols = read_metrics(os.path.join(r["dir"], "metrics.csv"))
            color = RAMP[min(i, len(RAMP) - 1)]
            ax.plot(cols["step"], cols["test_gen_exact_acc"], color=color, linewidth=2,
                    label=f"{r['label_noise']:.0%}")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("step", fontsize=9, color=INK)
        ax.set_ylabel("test exact acc", fontsize=9, color=INK)
        style_axes(ax, f"{rules}  (seed {seed})")
    finish(fig, axes, out, "Test accuracy by label-noise level", ncol=len(RAMP),
           legend_title="label noise")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="오답 섞기 실험 그래프")
    p.add_argument("runs_dirs", nargs="+")
    p.add_argument("--out-dir", type=str, default="runs_noise/plots")
    p.add_argument("--gen", type=float, default=0.95, help="이해 기준 test 정확도")
    p.add_argument("--mem", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=0.2, help="그래프 2 에 쓸 오답 비율")
    p.add_argument("--seed", type=int, default=0, help="그래프 2, 3 에 쓸 seed")
    args = p.parse_args(argv)

    runs = load_runs(args.runs_dirs, args.gen, args.mem)
    if not runs:
        print("no --rules runs found")
        return 1
    tasks = by_task(runs)
    os.makedirs(args.out_dir, exist_ok=True)
    plot_grok_vs_noise(tasks, os.path.join(args.out_dir, "grok_vs_noise.png"), args.gen)
    plot_noisy_dynamics(
        tasks, os.path.join(args.out_dir, "noisy_dynamics.png"), args.noise, args.seed
    )
    plot_test_curves(tasks, os.path.join(args.out_dir, "test_curves.png"), args.seed)
    print(f"wrote 3 plots to {args.out_dir} ({len(runs)} runs, {len(tasks)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
