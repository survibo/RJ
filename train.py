"""Training CLI.

예:
    python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 \
                    --task ascending --steps 100000

    python train.py --data-dir data/n30_m5_tr1000_te4096_random_s42 \
                    --task mod --modulus 5 --steps 100000

resume:
    python train.py --resume runs/<run>/ckpt_last.pt --steps 500000
"""

import argparse
import csv
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch

from src.data import (
    build_sequences,
    load_meta,
    read_split,
    seq_len,
    vocab_size,
)
from src.metrics import (
    embd_norm,
    generation_metrics,
    output_loss,
    param_norm,
    teacher_forced_metrics,
)
from src.model import GPT, GPTConfig
from src.tasks import TASK_REGISTRY, build_targets

CSV_FIELDS = [
    "step",
    "wall_time",
    "lr",
    "train_loss",
    "test_loss",
    "train_token_acc",
    "test_token_acc",
    "train_exact_acc",
    "test_exact_acc",
    "train_gen_token_acc",
    "test_gen_token_acc",
    "train_gen_exact_acc",
    "test_gen_exact_acc",
    "train_gen_valid_acc",
    "test_gen_valid_acc",
    "param_norm",
    "embd_norm",
]

# resume 시 변경할 수 없는 항목 (--steps 만 연장 가능)
LOCKED_ARGS = [
    "data_dir",
    "task",
    "modulus",
    "n_embd",
    "n_head",
    "n_layer",
    "batch_size",
    "lr",
    "weight_decay",
    "warmup",
    "grokfast",
    "grokfast_alpha",
    "grokfast_lamb",
    "grokfast_start_step",
    "eval_every",
    "n_eval",
    "seed",
]


ARG_SPECS = [
    ("--data-dir", dict(type=str, default=None)),
    ("--task", dict(type=str, default="ascending", choices=sorted(TASK_REGISTRY))),
    ("--modulus", dict(type=int, default=5)),
    ("--n-embd", dict(type=int, default=128)),
    ("--n-head", dict(type=int, default=4)),
    ("--n-layer", dict(type=int, default=2)),
    ("--batch-size", dict(type=int, default=512)),
    ("--lr", dict(type=float, default=1e-3)),
    ("--weight-decay", dict(type=float, default=1.0)),
    ("--warmup", dict(type=int, default=10)),
    ("--grokfast", dict(action="store_true", default=False)),
    ("--grokfast-alpha", dict(type=float, default=0.98)),
    ("--grokfast-lamb", dict(type=float, default=2.0)),
    ("--grokfast-start-step", dict(type=int, default=0)),
    ("--steps", dict(type=int, default=100000)),
    ("--eval-every", dict(type=int, default=250)),
    ("--n-eval", dict(type=int, default=4096)),
    ("--seed", dict(type=int, default=42)),
    ("--runs-dir", dict(type=str, default="runs")),
    ("--resume", dict(type=str, default=None, help="path to ckpt_last.pt")),
    ("--device", dict(type=str, default="auto", choices=["auto", "cpu", "cuda"])),
]


def build_parser(suppress: bool = False) -> argparse.ArgumentParser:
    """suppress=True 면 명시적으로 준 인자만 namespace 에 담긴다 (resume 충돌 검사용)."""
    kwargs = {"argument_default": argparse.SUPPRESS} if suppress else {}
    p = argparse.ArgumentParser(description="sorting-grokking trainer", **kwargs)
    for flag, spec in ARG_SPECS:
        spec = dict(spec)
        if suppress:
            spec.pop("default", None)
        p.add_argument(flag, **spec)
    return p


def run_name(cfg: dict) -> str:
    parts = [
        cfg["task"],
        cfg["split_strategy"],
        f"n{cfg['n']}m{cfg['m']}",
    ]
    if cfg["task"] == "mod":
        parts.append(f"k{cfg['modulus']}")
    parts.append(f"tr{cfg['train_size']}")
    if cfg.get("grokfast", False):
        parts.append(
            f"gfema-a{cfg['grokfast_alpha']:g}-l{cfg['grokfast_lamb']:g}"
        )
        if cfg["grokfast_start_step"] > 0:
            parts.append(f"gfstart{cfg['grokfast_start_step']}")
    parts.append(f"s{cfg['seed']}")
    return "_".join(parts)


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("error: --device cuda requested but CUDA is not available.")
    return torch.device(choice)


def load_sequences(data_dir: str, task: str, modulus, meta: dict):
    m, n = meta["m"], meta["n"]
    out = {}
    for split in ("train", "test"):
        inputs = read_split(os.path.join(data_dir, f"{split}.txt"), m)
        targets = build_targets(task, inputs, modulus=modulus)
        out[split] = torch.from_numpy(build_sequences(inputs, targets, n))
    return out["train"], out["test"]


def pick_eval_indices(size: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if size <= k:
        return np.arange(size, dtype=np.int64)
    return np.sort(rng.choice(size, size=k, replace=False)).astype(np.int64)


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def append_csv_row(path: str, row: dict) -> None:
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: fmt(row.get(k)) for k in CSV_FIELDS})


def save_checkpoint(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def apply_grokfast_ema(
    model: torch.nn.Module,
    ema: Optional[dict],
    alpha: float,
    lamb: float,
) -> dict:
    """Amplify the low-frequency component of each parameter gradient."""
    if ema is None:
        ema = {}

    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if not parameter.requires_grad or grad is None:
            continue
        if name not in ema:
            ema[name] = grad.detach().clone()
        else:
            ema[name].mul_(alpha).add_(grad.detach(), alpha=1.0 - alpha)
        grad.add_(ema[name], alpha=lamb)

    return ema


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    provided = set(vars(build_parser(suppress=True).parse_args(argv)).keys())

    ckpt = None
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise SystemExit(f"error: checkpoint not found: {args.resume}")
        run_dir = os.path.dirname(os.path.abspath(args.resume))
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.isfile(cfg_path):
            raise SystemExit(f"error: config.json not found in {run_dir}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for key in LOCKED_ARGS:
            saved = config.get(key)
            if key in provided and getattr(args, key) != saved:
                raise SystemExit(
                    f"error: --{key.replace('_', '-')} cannot be changed on resume "
                    f"(saved={saved}, given={getattr(args, key)})"
                )
            if saved is not None:
                setattr(args, key, saved)
        if "steps" not in provided:
            args.steps = config["steps"]
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.steps < ckpt["global_step"]:
            raise SystemExit(
                f"error: --steps {args.steps} < already completed step {ckpt['global_step']}"
            )
        config["steps"] = args.steps
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
        meta = load_meta(config["data_dir"])
    else:
        if args.data_dir is None:
            raise SystemExit("error: --data-dir is required (or use --resume)")
        if not os.path.isdir(args.data_dir):
            raise SystemExit(f"error: data dir not found: {args.data_dir}")
        meta = load_meta(args.data_dir)
        if args.n_embd % args.n_head != 0:
            raise SystemExit(
                f"error: n_embd ({args.n_embd}) % n_head ({args.n_head}) != 0"
            )
        if args.task == "mod" and args.modulus <= 0:
            raise SystemExit(f"error: modulus must be > 0, got {args.modulus}")
        if not 0.0 <= args.grokfast_alpha < 1.0:
            raise SystemExit(
                f"error: grokfast alpha must satisfy 0 <= alpha < 1, "
                f"got {args.grokfast_alpha}"
            )
        if args.grokfast_lamb < 0.0:
            raise SystemExit(
                f"error: grokfast lambda must be >= 0, got {args.grokfast_lamb}"
            )
        if args.grokfast_start_step < 0:
            raise SystemExit(
                "error: grokfast start step must be >= 0, "
                f"got {args.grokfast_start_step}"
            )
        config = {
            "task": args.task,
            "modulus": args.modulus if args.task == "mod" else None,
            "n": meta["n"],
            "m": meta["m"],
            "train_size": meta["train_size"],
            "test_size": meta["test_size"],
            "split_strategy": meta["split_strategy"],
            "seed": args.seed,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.98],
            "warmup": args.warmup,
            "grokfast": args.grokfast,
            "grokfast_alpha": args.grokfast_alpha,
            "grokfast_lamb": args.grokfast_lamb,
            "grokfast_start_step": args.grokfast_start_step,
            "steps": args.steps,
            "eval_every": args.eval_every,
            "n_eval": args.n_eval,
            "data_dir": args.data_dir,
        }
        run_dir = os.path.join(args.runs_dir, run_name(config))
        if os.path.exists(run_dir):
            raise SystemExit(f"error: run directory already exists: {run_dir}")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    n, m = config["n"], config["m"]
    modulus = config["modulus"]
    device = pick_device(args.device)

    # ---------------------------------------------------------------- seeding
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------- data
    train_seqs, test_seqs = load_sequences(
        config["data_dir"], config["task"], modulus, meta
    )
    train_seqs = train_seqs.to(device)
    test_seqs = test_seqs.to(device)
    train_size = train_seqs.shape[0]

    # ------------------------------------------------------------------ model
    model_cfg = GPTConfig(
        vocab_size=vocab_size(n),
        block_size=seq_len(m),
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=0.0,
    )
    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=tuple(config["betas"]),
        weight_decay=config["weight_decay"],
    )
    warmup = config["warmup"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0,
    )

    # -------------------------------------------------- rng / eval subset 복원
    batch_rng = np.random.default_rng(seed)
    eval_rng = np.random.default_rng(seed + 1)
    global_step = 0
    elapsed = 0.0
    grokfast_ema = None

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt["global_step"]
        elapsed = ckpt.get("elapsed", 0.0)
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda" and ckpt.get("cuda_rng") is not None:
            states = ckpt["cuda_rng"]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)
        batch_rng.bit_generator.state = ckpt["batch_rng"]
        train_eval_idx = np.asarray(ckpt["train_eval_idx"], dtype=np.int64)
        test_eval_idx = np.asarray(ckpt["test_eval_idx"], dtype=np.int64)
        if config.get("grokfast", False):
            if "grokfast_ema" not in ckpt:
                raise SystemExit("error: Grokfast checkpoint is missing EMA state")
            saved_ema = ckpt["grokfast_ema"]
            if saved_ema is not None:
                grokfast_ema = {
                    name: value.to(device) for name, value in saved_ema.items()
                }
    else:
        train_eval_idx = pick_eval_indices(train_size, config["n_eval"], eval_rng)
        test_eval_idx = pick_eval_indices(test_seqs.shape[0], config["n_eval"], eval_rng)

    train_eval = train_seqs[torch.from_numpy(train_eval_idx).to(device)]
    test_eval = test_seqs[torch.from_numpy(test_eval_idx).to(device)]

    metrics_path = os.path.join(run_dir, "metrics.csv")
    ckpt_path = os.path.join(run_dir, "ckpt_last.pt")
    eval_bs = max(1, config["batch_size"])

    print(f"run_dir   : {run_dir}")
    print(f"device    : {device}")
    print(f"task      : {config['task']} (modulus={modulus})")
    print(f"data      : {config['data_dir']}  train={train_size} test={test_seqs.shape[0]}")
    print(f"eval subs : train={len(train_eval_idx)} test={len(test_eval_idx)}")
    print(f"params    : {model.num_params()}")
    print(f"steps     : {global_step} -> {config['steps']}")
    if config.get("grokfast", False):
        print(
            f"grokfast  : EMA alpha={config['grokfast_alpha']:g} "
            f"lambda={config['grokfast_lamb']:g} "
            f"start={config['grokfast_start_step']}"
        )

    start_time = time.time() - elapsed

    def do_eval() -> None:
        tf_tr = teacher_forced_metrics(model, train_eval, m, eval_bs, device)
        tf_te = teacher_forced_metrics(model, test_eval, m, eval_bs, device)
        gen_tr = generation_metrics(model, train_eval, m, eval_bs, device)
        gen_te = generation_metrics(model, test_eval, m, eval_bs, device)
        row = {
            "step": global_step,
            "wall_time": time.time() - start_time,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": tf_tr["loss"],
            "test_loss": tf_te["loss"],
            "train_token_acc": tf_tr["token_acc"],
            "test_token_acc": tf_te["token_acc"],
            "train_exact_acc": tf_tr["exact_acc"],
            "test_exact_acc": tf_te["exact_acc"],
            "train_gen_token_acc": gen_tr["gen_token_acc"],
            "test_gen_token_acc": gen_te["gen_token_acc"],
            "train_gen_exact_acc": gen_tr["gen_exact_acc"],
            "test_gen_exact_acc": gen_te["gen_exact_acc"],
            "train_gen_valid_acc": gen_tr["gen_valid_acc"],
            "test_gen_valid_acc": gen_te["gen_valid_acc"],
            "param_norm": param_norm(model),
            "embd_norm": embd_norm(model),
        }
        append_csv_row(metrics_path, row)
        print(
            f"step {global_step:>7d} | loss {row['train_loss']:.4f}/{row['test_loss']:.4f} "
            f"| gen_exact {row['train_gen_exact_acc']:.3f}/{row['test_gen_exact_acc']:.3f} "
            f"| gen_valid {row['train_gen_valid_acc']:.3f}/{row['test_gen_valid_acc']:.3f} "
            f"| pnorm {row['param_norm']:.2f} | {row['wall_time']:.0f}s",
            flush=True,
        )

    def save() -> None:
        save_checkpoint(
            ckpt_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "elapsed": time.time() - start_time,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
                "batch_rng": batch_rng.bit_generator.state,
                "train_eval_idx": train_eval_idx,
                "test_eval_idx": test_eval_idx,
                "grokfast_ema": grokfast_ema,
                "config": config,
            },
        )

    if ckpt is None:
        do_eval()
        save()

    model.train()
    batch_size = config["batch_size"]
    eval_every = config["eval_every"]
    total_steps = config["steps"]
    while global_step < total_steps:
        idx = batch_rng.integers(0, train_size, size=batch_size)  # replacement sampling
        batch = train_seqs[torch.from_numpy(idx).to(device)]
        logits = model(batch)
        loss = output_loss(logits, batch, m)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if (
            config.get("grokfast", False)
            and global_step >= config["grokfast_start_step"]
        ):
            grokfast_ema = apply_grokfast_ema(
                model,
                grokfast_ema,
                alpha=config["grokfast_alpha"],
                lamb=config["grokfast_lamb"],
            )
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % eval_every == 0 or global_step == total_steps:
            do_eval()
            save()

    print(f"done. run_dir = {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
