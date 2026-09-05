"""Metrics CSV -> 3-panel grokking plot.

예:
    python plot.py "runs/*/metrics.csv" --out runs/grokking.png
"""

import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PANELS = [
    (
        "Generation accuracy",
        ["train_gen_exact_acc", "test_gen_exact_acc", "test_gen_valid_acc"],
        "linear",
        "accuracy",
    ),
    ("Cross-entropy loss", ["train_loss", "test_loss"], "log", "loss"),
    ("Parameter norms", ["param_norm", "embd_norm"], "linear", "L2 norm"),
]
METRIC_STYLES = {
    "train_gen_exact_acc": ("Train exact", "--", 1.5, 0.72),
    "test_gen_exact_acc": ("Test exact", "-", 2.3, 1.0),
    "test_gen_valid_acc": ("Test valid", ":", 1.9, 0.9),
    "train_loss": ("Train", "--", 1.5, 0.72),
    "test_loss": ("Test", "-", 2.2, 1.0),
    "param_norm": ("All parameters", "-", 2.0, 0.95),
    "embd_norm": ("Token embedding", "--", 1.6, 0.78),
}


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


def compact_run_labels(labels: Sequence[str]) -> Tuple[List[str], str]:
    """Move a shared underscore-delimited suffix out of a multi-run legend."""
    if len(labels) < 2:
        return list(labels), ""
    parts = [label.split("_") for label in labels]
    shared: List[str] = []
    while all(items for items in parts):
        candidate = parts[0][-1]
        if not all(items[-1] == candidate for items in parts):
            break
        if any(len(items) == 1 for items in parts):
            break
        shared.append(candidate)
        for items in parts:
            items.pop()
    compact = ["_".join(items) for items in parts]
    context = " / ".join(reversed(shared))
    return compact, context


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

    raw_labels = [run_label(path) for path in paths]
    labels, context = compact_run_labels(raw_labels)
    runs = [(label, read_metrics(path)) for label, path in zip(labels, paths)]
    cmap = plt.get_cmap("tab10")
    colors = {label: cmap(i % 10) for i, (label, _) in enumerate(runs)}

    plt.rcParams.update(
        {
            "axes.edgecolor": "#c7cbd1",
            "axes.labelcolor": "#343a40",
            "axes.titlecolor": "#20252b",
            "font.size": 10,
            "xtick.color": "#555b63",
            "ytick.color": "#555b63",
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.5, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0], "hspace": 0.28},
    )
    fig.patch.set_facecolor("#f7f8fa")
    for ax, (title, metrics, yscale, ylabel) in zip(axes, PANELS):
        if title == "Parameter norms":
            yscale = args.norm_y_scale
        ax.set_facecolor("white")
        for label, cols in runs:
            steps = cols.get("step", [])
            for metric in metrics:
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
                _, linestyle, linewidth, alpha = METRIC_STYLES[metric]
                ax.plot(
                    xs,
                    yy,
                    color=colors[label],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel)
        ax.set_xscale(args.x_scale)
        ax.set_yscale(yscale)
        ax.grid(axis="y", which="major", color="#dfe3e8", linewidth=0.8)
        ax.grid(axis="y", which="minor", color="#edf0f2", linewidth=0.5)
        ax.grid(axis="x", which="major", color="#edf0f2", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)
        if title == "Generation accuracy":
            ax.axhspan(0.9, 1.0, color="#e9f5ec", alpha=0.7, zorder=0)
            ax.axhline(0.95, color="#9ebca5", linewidth=0.8, zorder=0)
            ax.set_ylim(-0.015, 1.015)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
        # metric 은 linestyle, run 은 color 로 구분한다.
        handles = [
            Line2D(
                [],
                [],
                color="#343a40",
                linestyle=METRIC_STYLES[metric][1],
                linewidth=METRIC_STYLES[metric][2],
                alpha=METRIC_STYLES[metric][3],
                label=METRIC_STYLES[metric][0],
            )
            for metric in metrics
        ]
        ax.legend(
            handles=handles,
            fontsize=8.5,
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            borderaxespad=0,
            frameon=False,
            ncol=len(handles),
            handlelength=2.8,
            columnspacing=1.4,
        )
    axes[-1].set_xlabel("Training step")

    if len(runs) > 1:
        run_handles = [
            Line2D([], [], color=colors[label], linestyle="-", label=label)
            for label, _ in runs
        ]
        fig.legend(
            handles=run_handles,
            fontsize=8.5,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(4, len(runs)),
            frameon=False,
            title="Runs",
            title_fontsize=9,
        )
        fig.suptitle("Grokking training curves", fontsize=15, fontweight="bold", y=0.995)
        if context:
            fig.text(0.5, 0.972, context, ha="center", va="top", fontsize=9, color="#687079")
        legend_rows = math.ceil(len(runs) / 4)
        fig.subplots_adjust(
            left=0.09,
            right=0.975,
            bottom=0.075,
            top=0.90 - 0.025 * (legend_rows - 1),
        )
    else:
        fig.suptitle(runs[0][0], fontsize=14, fontweight="bold", y=0.99)
        fig.subplots_adjust(left=0.09, right=0.975, bottom=0.075, top=0.94)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} ({len(runs)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
