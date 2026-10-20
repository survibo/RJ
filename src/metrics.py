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


# inference_mode tensors conflict with the compiled CUDA training graph after step-0 eval.
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
    model,
    seqs: torch.Tensor,
    m: int,
    batch_size: int,
    device: torch.device,
    return_preds: bool = False,
) -> Dict[str, float]:
    """return_preds=True 면 생성 결과 [N, m] 를 'preds' 로 함께 돌려준다."""
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
    preds = []
    for i in range(0, n, batch_size):
        batch = seqs[i : i + batch_size].to(device, non_blocking=True)
        prefix = batch[:, : m + 2]
        target = batch[:, m + 2 : 2 * m + 2]
        pred = generate(model, prefix, m)
        if return_preds:
            preds.append(pred)
        correct = pred.eq(target)
        token_correct += int(correct.sum().item())
        exact_correct += int(correct.all(dim=1).sum().item())
        # 변환 task는 중복 token을 출력할 수 있으므로 target multiset과 비교한다.
        valid = pred.sort(dim=1).values.eq(target.sort(dim=1).values).all(dim=1)
        valid_correct += int(valid.sum().item())
    if was_training:
        model.train()
    out = {
        "gen_token_acc": token_correct / (n * m),
        "gen_exact_acc": exact_correct / n,
        "gen_valid_acc": valid_correct / n,
    }
    if return_preds:
        out["preds"] = torch.cat(preds)
    return out


@torch.no_grad()
def symmetry_metrics(
    model,
    seqs: torch.Tensor,
    preds: torch.Tensor,
    perms: torch.Tensor,
    kind: str,
    m: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """대칭 위반 점수 (정답 target 을 쓰지 않는다).

    seqs: [N, 2m+2], preds: seqs 입력에 대한 모델 생성 결과 [N, m],
    perms: 행마다 하나씩 고른 순열 index [N, m] (섞인 입력 = x[p]).
    불변이면 model(x[p]) 와 model(x), 등변이면 model(x[p]) 와 model(x)[p] 를 비교한다.
    """
    n = seqs.shape[0]
    if n == 0 or kind not in ("invariant", "equivariant"):
        return {"sym_token_viol": math.nan, "sym_exact_viol": math.nan}
    was_training = model.training
    model.eval()
    token_viol = 0
    exact_viol = 0
    for i in range(0, n, batch_size):
        batch = seqs[i : i + batch_size].to(device, non_blocking=True)
        p = perms[i : i + batch_size].to(device, non_blocking=True)
        prefix = batch[:, : m + 2].clone()
        prefix[:, 1 : m + 1] = batch[:, 1 : m + 1].gather(1, p)
        pred_perm = generate(model, prefix, m)
        ref = preds[i : i + batch_size].to(device, non_blocking=True)
        if kind == "equivariant":
            ref = ref.gather(1, p)
        diff = pred_perm.ne(ref)
        token_viol += int(diff.sum().item())
        exact_viol += int(diff.any(dim=1).sum().item())
    if was_training:
        model.train()
    return {"sym_token_viol": token_viol / (n * m), "sym_exact_viol": exact_viol / n}


def noise_metrics(
    preds: torch.Tensor, wrong: torch.Tensor, clean: torch.Tensor, pos: torch.Tensor
) -> Dict[str, float]:
    """오답 행에서 모델이 오답을 따라 하는지(외움), 규칙대로 답하는지(이해).

    preds / wrong / clean: [N, m] 생성 결과 / 학습에 쓴 오답 target / 원래 정답 target.
    pos: [N] 행마다 오답으로 바뀐 칸 위치.
    *_exact 는 수열 전체가 같은 비율, *_tok 는 바뀐 칸 하나만 본 비율.
    """
    if preds.shape[0] == 0:
        return {k: math.nan for k in ("mem_exact", "rule_exact", "mem_tok", "rule_tok")}
    rows = torch.arange(preds.shape[0], device=preds.device)
    at_pos = preds[rows, pos]
    return {
        "mem_exact": preds.eq(wrong).all(dim=1).float().mean().item(),
        "rule_exact": preds.eq(clean).all(dim=1).float().mean().item(),
        "mem_tok": at_pos.eq(wrong[rows, pos]).float().mean().item(),
        "rule_tok": at_pos.eq(clean[rows, pos]).float().mean().item(),
    }


def unique_frac(preds: torch.Tensor) -> float:
    """서로 다른 출력 수열의 비율. 상수 출력 같은 사소한 대칭 해를 가려내는 데 쓴다."""
    if preds.shape[0] == 0:
        return math.nan
    return torch.unique(preds, dim=0).shape[0] / preds.shape[0]


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
