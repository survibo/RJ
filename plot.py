"""Metrics CSV -> 3-panel grokking plot.

예:
    python plot.py "runs/*/metrics.csv" --out runs/grokking.png
"""

import argparse
import csv
import glob
import math
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PANELS = [
    ("Accuracy", ["train_gen_exact_acc", "test_gen_exact_acc", "test_gen_valid_acc"], "linear"),
    ("Loss", ["train_loss", "test_loss"], "log"),
    ("Norm", ["param_norm", "embd_norm"], "linear"),
]
LINESTYLES = ["-", "--", ":", "-."]


def read_metrics(path: str) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return cols
        for name in reader.fieldnames:
            cols[name] = []
        for row in reader:
            for name in reader.fieldnames:
                v = (row.get(name) or "").strip()
                try:
                    cols[name].append(float(v) if v != "" else math.nan)
                except ValueError:
                    cols[name].append(math.nan)
    return cols


def run_label(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="plot grokking curves")
    p.add_argument("pattern", nargs="+", help='예: "runs/*/metrics.csv"')
    p.add_argument("--out", type=str, default="runs/grokking.png")
    p.add_argument("--x-scale", choices=["log", "linear"], default="log")
    p.add_argument(
        "--norm-y-scale",
        choices=["linear", "log"],
        default="linear",
        help="param_norm 과 embd_norm 의 scale 차이가 클 때 log 로 보면 편하다",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args(argv)

    paths: List[str] = []
    for pat in args.pattern:
        matched = sorted(glob.glob(pat))
        paths.extend(matched if matched else ([pat] if os.path.isfile(pat) else []))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise SystemExit(f"error: no metrics.csv matched {args.pattern}")

    runs = [(run_label(path), read_metrics(path)) for path in paths]
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i % 10) for i, (label, _) in enumerate(runs)}

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, (title, metrics, yscale) in zip(axes, PANELS):
        if title == "Norm":
            yscale = args.norm_y_scale
        for label, cols in runs:
            steps = cols.get("step", [])
            for k, metric in enumerate(metrics):
                ys = cols.get(metric)
                if ys is None:
                    continue
                xs, yy = [], []
                for x, y in zip(steps, ys):
                    if math.isnan(y):
                        continue
                    if args.x_scale == "log" and x <= 0:
                        continue
                    if yscale == "log" and y <= 0:
                        continue
                    xs.append(x)
                    yy.append(y)
                if not xs:
                    continue
                ax.plot(
                    xs,
                    yy,
                    color=colors[label],
                    linestyle=LINESTYLES[k % len(LINESTYLES)],
                    linewidth=1.4,
                    label=f"{label} | {metric}" if len(runs) > 1 else metric,
                )
        ax.set_title(title)
        ax.set_xscale(args.x_scale)
        ax.set_yscale(yscale)
        ax.grid(True, which="both", alpha=0.25)
        if title == "Accuracy":
            ax.set_ylim(-0.02, 1.02)
        # metric 은 linestyle, run 은 color 로 구분한다.
        handles = [
            Line2D([], [], color="black", linestyle=LINESTYLES[k % len(LINESTYLES)], label=metric)
            for k, metric in enumerate(metrics)
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")
    axes[-1].set_xlabel("step")

    if len(runs) > 1:
        run_handles = [
            Line2D([], [], color=colors[label], linestyle="-", label=label)
            for label, _ in runs
        ]
        fig.legend(
            handles=run_handles,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(3, len(runs)),
        )
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.02 * math.ceil(len(runs) / 3) - 0.02))
    else:
        fig.suptitle(runs[0][0], fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out} ({len(runs)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
