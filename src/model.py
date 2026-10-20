"""GPT-2 style decoder-only Transformer (직접 구현)."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    # > 0 이면 prefix-LM: 앞 prefix_len 개 token (BOS x1..xm SEP) 끼리는 양방향 attention
    prefix_len: int = 0
    # False 면 입력 칸 x1..xm 이 모두 같은 위치 embedding 을 받는다 (prefix-LM 필요)
    input_pos: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError(
                f"n_embd ({cfg.n_embd}) must be divisible by n_head ({cfg.n_head})"
            )
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN -> attention -> residual, Pre-LN -> MLP -> residual"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        if not cfg.input_pos and cfg.prefix_len <= 2:
            raise ValueError("input_pos=False requires a prefix-LM (prefix_len > 2)")
        if cfg.prefix_len > cfg.block_size:
            raise ValueError(f"prefix_len {cfg.prefix_len} > block_size {cfg.block_size}")
        pos_ids = torch.arange(cfg.block_size)
        if not cfg.input_pos:
            pos_ids[1 : cfg.prefix_len - 1] = 1
        self.register_buffer("pos_ids", pos_ids, persistent=False)
        if cfg.prefix_len > 0:
            mask = torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool).tril()
            mask[: cfg.prefix_len, : cfg.prefix_len] = True
            self.register_buffer("attn_mask", mask, persistent=False)
        else:
            self.attn_mask = None
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # token embedding 과 LM head weight 공유
        self.lm_head.weight = self.wte.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        x = self.drop(self.wte(idx) + self.wpe(self.pos_ids[:T])[None, :, :])
        mask = None if self.attn_mask is None else self.attn_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_params(self) -> int:
        # weight tying 때문에 lm_head.weight 는 wte.weight 와 같은 tensor 다.
        seen = {}
        for p in self.parameters():
            seen[id(p)] = p.numel()
        return sum(seen.values())
