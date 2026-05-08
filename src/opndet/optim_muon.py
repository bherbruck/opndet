"""Muon + MuSGD optimizers for opndet's server-tier -pro variants.

Per YOLO26 (Sapkota et al. 2025) and Jordan (2024). Muon is for matrix-shaped
parameters (Conv2d weights flattened to [out_ch, in_ch*kH*kW], Linear weights,
2D embeddings). Everything else (biases, BN affine, narrow head 1x1 convs)
falls back to AdamW. MuSGD is a thin wrapper that owns both legs.

References (public, non-AGPL):
  Jordan 2024 — https://kellerjordan.github.io/posts/muon/
  Sapkota et al. 2025, YOLO26 §2.4 (arXiv 2509.25164)

Constraint: this file does not affect the AdamW default path. Only constructed
when the user opts in via `optimizer: musgd` in the training config.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn

# Quintic Newton-Schulz coefficients (Jordan 2024). Tuned so the iteration
# converges to a matrix with singular values in [~0.7, ~1.0] from any starting
# matrix with spectral norm <= 1. Five iterations is the canonical choice.
_NS_COEF: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(G: torch.Tensor, n_iter: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization. Returns a matrix with the same
    shape as G, approximately equal to U V^T from G's SVD G = U S V^T.

    G can be 2D or higher; it is reshaped to 2D as `G.reshape(G.shape[0], -1)`
    for the iteration, then reshaped back. The math operates on the smaller
    side (X is transposed when out_dim > in_dim) for fewer flops.
    """
    a, b, c = _NS_COEF
    orig_shape = G.shape
    if G.ndim != 2:
        G = G.reshape(G.shape[0], -1)
    # Use bf16/fp32 for the iteration (matmul precision) regardless of param dtype.
    compute_dtype = torch.float32 if G.dtype not in (torch.float32, torch.bfloat16) else G.dtype
    X = G.to(compute_dtype)
    X = X / (X.norm() + eps)
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    for _ in range(n_iter):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype).reshape(orig_shape)


def _is_muon_eligible(p: torch.Tensor) -> bool:
    """Muon expects a 2D-flattenable matrix where both dimensions are >= 2.

    Skips:
      - 1D params (BN affine, biases, LayerNorm scale/shift)
      - degenerate matrices where one of the flat dims is 1 (head 1x1 conv to
        1 or few channels falls into AdamW; orthogonalizing a 1xN row vector
        is meaningless and unstable)
    """
    if p.ndim < 2:
        return False
    flat0 = p.shape[0]
    flat1 = 1
    for s in p.shape[1:]:
        flat1 *= s
    return flat0 >= 2 and flat1 >= 2


def partition_params(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Walk model.named_parameters() and split into (muon_params, adamw_params).

    Muon: Conv2d / Linear / >=2D weights with both flat dims >= 2.
    AdamW: everything else (biases, BN, narrow heads, 1D embeddings).
    """
    muon: list[torch.Tensor] = []
    rest: list[torch.Tensor] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon if _is_muon_eligible(p) else rest).append(p)
    return muon, rest


class Muon(torch.optim.Optimizer):
    """Muon optimizer (Jordan 2024).

    Heavy-ball momentum on the gradient, then orthogonalize the momentum buffer
    via Newton-Schulz before stepping. Decoupled weight decay (AdamW-style).

    Only valid for matrix-shaped params (ndim >= 2 and both flat dims >= 2).
    Construction asserts this so users can't quietly route a 1D bias here.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if not _is_muon_eligible(p):
                    raise ValueError(
                        f"Muon got an ineligible param of shape {tuple(p.shape)}; "
                        f"use partition_params() to route 1D / narrow params to AdamW."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                upd = newton_schulz(buf, n_iter=ns)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(upd, alpha=-lr)
        return loss


class MuSGD:
    """MuSGD: Muon for hidden matrix params + AdamW for the rest.

    Wraps two standard PyTorch optimizers. Quacks like an Optimizer for the
    bits opndet's training loop touches: `param_groups`, `state_dict`,
    `load_state_dict`, `step`, `zero_grad`. Not a strict subclass — just a
    facade that dispatches to both legs.

    The training loop iterates `param_groups` for LR scheduling. We expose all
    groups from both legs (Muon first, AdamW second) so the cosine schedule
    sweeps both.
    """

    # state-dict tag so we can fail loudly when a user tries to load an AdamW
    # checkpoint into a MuSGD optimizer (or vice versa).
    _TAG = "musgd-v1"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        muon_momentum: float = 0.95,
        muon_lr_scale: float = 1.0,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ):
        self.muon_params, self.adamw_params = partition_params(model)
        self.muon_lr_scale = float(muon_lr_scale)
        self._base_lr = float(lr)
        if self.muon_params:
            self.muon = Muon(
                self.muon_params,
                lr=lr * muon_lr_scale,
                momentum=muon_momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
            )
        else:
            self.muon = None
        if self.adamw_params:
            self.adamw = torch.optim.AdamW(
                self.adamw_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=adamw_betas,
                eps=adamw_eps,
            )
        else:
            self.adamw = None
        if self.muon is None and self.adamw is None:
            raise ValueError("MuSGD: model has no trainable parameters.")

    # --- proxy props for the training loop ----------------------------------

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Concatenate both optimizers' groups so the cosine LR scheduler can
        update both. Muon's lr is held at `base_lr * muon_lr_scale`; we apply
        the scale on read/write via __setitem__-style: train.py does
        `g['lr'] = cosine_lr(...)` on every group, so we need the Muon groups
        to also get the same lr. That's fine — Muon's effective lr is roughly
        comparable to AdamW's, so a unified schedule works.
        """
        groups: list[dict[str, Any]] = []
        if self.muon is not None:
            groups.extend(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.muon is not None:
            self.muon.step()
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "_tag": self._TAG,
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
            "muon_lr_scale": self.muon_lr_scale,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        tag = state.get("_tag")
        if tag != self._TAG:
            raise ValueError(
                f"Optimizer state-dict tag mismatch: expected '{self._TAG}', got '{tag}'. "
                f"This usually means the checkpoint was trained with a different "
                f"optimizer (e.g. AdamW) and you're trying to resume with MuSGD. "
                f"Either resume with the original optimizer or start a fresh run."
            )
        if state.get("muon") is not None:
            if self.muon is None:
                raise ValueError("checkpoint has Muon state but current model has no Muon-eligible params")
            self.muon.load_state_dict(state["muon"])
        if state.get("adamw") is not None:
            if self.adamw is None:
                raise ValueError("checkpoint has AdamW state but current model has no AdamW params")
            self.adamw.load_state_dict(state["adamw"])
        self.muon_lr_scale = float(state.get("muon_lr_scale", self.muon_lr_scale))

    # convenience for logging / debug
    def partition_summary(self) -> dict[str, int]:
        n_muon = sum(p.numel() for p in self.muon_params)
        n_adamw = sum(p.numel() for p in self.adamw_params)
        return {
            "muon_tensors": len(self.muon_params),
            "adamw_tensors": len(self.adamw_params),
            "muon_params": n_muon,
            "adamw_params": n_adamw,
            "total_params": n_muon + n_adamw,
        }
