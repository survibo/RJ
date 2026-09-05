"""Teacher-forced / generation metric, norm metric."""

import math
from typing import Dict

import torch
import torch.nn.functional as F


def output_logits_and_targets(logits: torch.Tensor, seqs: torch.Tensor, m: int):
    """sequence: BOS x1..xm SEP y1..ym

    position m+1 (SEP) 부터 2m 까지의 prediction 만 사용한다.
    반환: out_logits [B, m, V], out_targets [B, m]
    """
    return logits[:, m + 1 : 2 * m + 1, :], seqs[:, m + 2 : 2 * m + 2]


def output_loss(logits: torch.Tensor, seqs: torch.Tensor, m: int) -> torch.Tensor:
    out_logits, out_targets = output_logits_and_targets(logits, seqs, m)
    return F.cross_entropy(
        out_logits.reshape(-1, out_logits.size(-1)), out_targets.reshape(-1)
    )


@torch.no_grad()
def teacher_forced_metrics(
    model, seqs: torch.Tensor, m: int, batch_size: int, device: torch.device
) -> Dict[str, float]:
    n = seqs.shape[0]
    if n == 0:
        return {"loss": math.nan, "token_acc": math.nan, "exact_acc": math.nan}
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_correct = 0
    exact_correct = 0
    for i in range(0, n, batch_size):
        batch = seqs[i : i + batch_size].to(device, non_blocking=True)
        logits = model(batch)
        out_logits, out_targets = output_logits_and_targets(logits, batch, m)
        loss = F.cross_entropy(
            out_logits.reshape(-1, out_logits.size(-1)),
            out_targets.reshape(-1),
            reduction="sum",
        )
        loss_sum += loss.item()
        pred = out_logits.argmax(dim=-1)
        correct = pred.eq(out_targets)
        token_correct += int(correct.sum().item())
        exact_correct += int(correct.all(dim=1).sum().item())
    if was_training:
        model.train()
    return {
        "loss": loss_sum / (n * m),
        "token_acc": token_correct / (n * m),
        "exact_acc": exact_correct / n,
    }


@torch.no_grad()
def generate(model, prefix: torch.Tensor, m: int) -> torch.Tensor:
    """prefix: [B, m+2] (BOS x1..xm SEP). greedy 로 정확히 m 개 token 생성."""
    cur = prefix
    for _ in range(m):
        logits = model(cur)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
    return cur[:, prefix.shape[1] :]


@torch.no_grad()
def generation_metrics(
    model, seqs: torch.Tensor, m: int, batch_size: int, device: torch.device
) -> Dict[str, float]:
    n = seqs.shape[0]
    if n == 0:
        return {
            "gen_token_acc": math.nan,
            "gen_exact_acc": math.nan,
            "gen_valid_acc": math.nan,
        }
    was_training = model.training
    model.eval()
    token_correct = 0
    exact_correct = 0
    valid_correct = 0
    for i in range(0, n, batch_size):
        batch = seqs[i : i + batch_size].to(device, non_blocking=True)
        prefix = batch[:, : m + 2]
        target = batch[:, m + 2 : 2 * m + 2]
        pred = generate(model, prefix, m)
        correct = pred.eq(target)
        token_correct += int(correct.sum().item())
        exact_correct += int(correct.all(dim=1).sum().item())
        # 변환 task는 중복 token을 출력할 수 있으므로 target multiset과 비교한다.
        valid = pred.sort(dim=1).values.eq(target.sort(dim=1).values).all(dim=1)
        valid_correct += int(valid.sum().item())
    if was_training:
        model.train()
    return {
        "gen_token_acc": token_correct / (n * m),
        "gen_exact_acc": exact_correct / n,
        "gen_valid_acc": valid_correct / n,
    }


@torch.no_grad()
def param_norm(model) -> float:
    seen = {}
    for p in model.parameters():
        if p.requires_grad:
            seen[id(p)] = p
    total = sum(float(p.detach().float().pow(2).sum().item()) for p in seen.values())
    return math.sqrt(total)


@torch.no_grad()
def embd_norm(model) -> float:
    return float(model.wte.weight.detach().float().norm().item())
