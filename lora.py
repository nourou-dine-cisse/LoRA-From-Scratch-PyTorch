"""
LoRA: Low-Rank Adaptation of Large Language Models (From Scratch in PyTorch)
Author: Mouhamed Nourou Dine Cisse (ENSIIE & Université Paris-Saclay M2 Artificial Intelligence)
Paper Reference: Hu et al., 2021 (https://arxiv.org/abs/2106.09685)

Mathematical Formulation:
    h = W_0 x + Delta W x
    Delta W = (alpha / r) * (B @ A)
    where:
        W_0 in R^{d_out x d_in} is frozen pretrained weight
        A in R^{r x d_in} is initialized with Gaussian N(0, sigma^2)
        B in R^{d_out x r} is initialized to zeros (so Delta W = 0 initially)
        r is the rank (r << min(d_in, d_out))
        alpha is the constant scaling hyperparameter
"""

import math
from typing import Optional
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear incorporating Low-Rank Adaptation (LoRA).
    
    Freezes the base weight W_0 and trains two low-rank matrices A and B.
    Optionally supports weight merging at inference time to introduce zero extra latency.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank if rank > 0 else 1.0
        self.merge_weights = merge_weights
        self.merged = False

        # Pretrained frozen base linear projection
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

        if rank > 0:
            # Low-rank decomposition matrices:
            # A projects down from in_dim to rank
            self.lora_a = nn.Linear(in_dim, rank, bias=False)
            # B projects up from rank to out_dim
            self.lora_b = nn.Linear(rank, out_dim, bias=False)

            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

            # Initialize:
            # A initialized with standard normal (Kaiming uniform)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            # B initialized to zero so Delta W = 0 at the start of training
            nn.init.zeros_(self.lora_b.weight)
        else:
            self.lora_a = None
            self.lora_b = None
            self.dropout = nn.Identity()

    def merge(self):
        """Merges LoRA adapter weights directly into the base linear layer for zero inference overhead."""
        if self.rank > 0 and not self.merged:
            delta_w = (self.lora_b.weight @ self.lora_a.weight) * self.scaling
            self.linear.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Subtracts merged LoRA adapter weights from the base linear layer to resume training."""
        if self.rank > 0 and self.merged:
            delta_w = (self.lora_b.weight @ self.lora_a.weight) * self.scaling
            self.linear.weight.data -= delta_w
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.rank == 0:
            return self.linear(x)

        # Regular forward: W_0(x) + (alpha / r) * B(A(dropout(x)))
        base_out = self.linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


def linear_to_lora(
    linear: nn.Linear,
    rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0
) -> LoRALinear:
    """
    Converts a standard PyTorch nn.Linear layer to a LoRALinear layer,
    preserving original pretrained weights and bias.
    """
    out_dim, in_dim = linear.weight.shape
    has_bias = linear.bias is not None

    lora = LoRALinear(
        in_dim=in_dim,
        out_dim=out_dim,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=has_bias
    )
    # Copy pretrained weights into the frozen linear component
    lora.linear.weight.data = linear.weight.data.clone()
    if has_bias:
        lora.linear.bias.data = linear.bias.data.clone()

    return lora


def apply_lora_to_model(
    model: nn.Module,
    target_modules=("key", "value", "query"),
    rank: int = 8,
    lora_alpha: float = 16.0
):
    """
    Recursively replaces target linear layers (e.g. self-attention Q, K, V)
    with LoRALinear layers and freezes non-LoRA parameters.
    """
    for name, module in list(model.named_children()):
        if any(target in name.lower() for target in target_modules) and isinstance(module, nn.Linear):
            lora_layer = linear_to_lora(module, rank=rank, lora_alpha=lora_alpha)
            setattr(model, name, lora_layer)
        else:
            apply_lora_to_model(module, target_modules, rank=rank, lora_alpha=lora_alpha)

    # Freeze base model parameters
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name or "classifier" in name or "score" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def count_parameters(model: nn.Module):
    """Returns total parameters and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, (trainable / total) * 100.0


if __name__ == "__main__":
    print("=" * 70)
    print("LoRA: Low-Rank Adaptation (From Scratch in PyTorch)")
    print("=" * 70)

    # Test LoRALinear module
    batch_size, seq_len, in_dim, out_dim = 2, 16, 768, 768
    x = torch.randn(batch_size, seq_len, in_dim)

    linear_std = nn.Linear(in_dim, out_dim)
    lora_layer = linear_to_lora(linear_std, rank=8, lora_alpha=16.0)

    # Freezing check
    lora_layer.linear.weight.requires_grad = False
    if lora_layer.linear.bias is not None:
        lora_layer.linear.bias.requires_grad = False

    out_std = linear_std(x)
    out_lora_init = lora_layer(x)

    # At step 0, Delta W = 0 because B=0, so outputs must match exactly
    diff = (out_std - out_lora_init).abs().max().item()
    print(f"Initial deviation between base linear and LoRA: {diff:.8f}")
    assert diff < 1e-6, "LoRA initialization failed: B must be 0 at start!"
    print("✅ Initialization verification passed (B=0 ensures exact base model equivalence).")

    # Trainable parameter verification
    total, trainable, pct = count_parameters(lora_layer)
    print(f"LoRALinear params: Total={total:,} | Trainable={trainable:,} ({pct:.2f}%)")
    print("=" * 70)
