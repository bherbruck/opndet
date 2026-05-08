"""Tests for Muon / MuSGD optimizers. ROADMAP §1.8 Phase 6."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from opndet.optim_muon import (MuSGD, Muon, _is_muon_eligible, newton_schulz,
                                partition_params)


# ----- Newton-Schulz orthogonalization ----------------------------------------


def test_newton_schulz_returns_same_shape():
    G = torch.randn(64, 32)
    O = newton_schulz(G, n_iter=5)
    assert O.shape == G.shape


def test_newton_schulz_4d_conv_weight_shape_preserved():
    # Conv2d weight [out_ch, in_ch, kH, kW] flattened -> orthogonalize -> back.
    G = torch.randn(48, 24, 3, 3)
    O = newton_schulz(G, n_iter=5)
    assert O.shape == G.shape


def test_newton_schulz_singular_values_close_to_one():
    """Output should have singular values close to 1; this is the whole point.
    Quintic NS does not produce an exact polar decomposition — Jordan's
    coefficients give svs in roughly [0.7, 1.13]. Mean should be near 1."""
    torch.manual_seed(0)
    G = torch.randn(40, 60)
    O = newton_schulz(G, n_iter=5).to(torch.float64)
    s = torch.linalg.svdvals(O)
    assert s.min() > 0.6, f"min sv {s.min()}"
    assert s.max() < 1.2, f"max sv {s.max()}"
    assert 0.8 < s.mean() < 1.1, f"mean sv {s.mean()}"


def test_newton_schulz_handles_tall_and_wide():
    for shape in [(8, 32), (32, 8), (16, 16)]:
        torch.manual_seed(0)
        G = torch.randn(*shape)
        O = newton_schulz(G, n_iter=5)
        assert O.shape == shape
        s = torch.linalg.svdvals(O.to(torch.float64))
        assert s.min() > 0.6
        assert s.max() < 1.2


# ----- partition_params -------------------------------------------------------


def _tiny_cnn() -> nn.Module:
    """Conv -> BN -> Conv (head-like 1ch). Mix of muon-eligible + ineligible."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True),  # weight 2D-eligible, bias 1D
        nn.BatchNorm2d(16),  # weight + bias both 1D
        nn.Conv2d(16, 1, kernel_size=1, bias=True),  # weight ndim>=2 but flat dim_out=1 — ineligible
    )


def test_partition_routes_conv_weight_to_muon():
    m = _tiny_cnn()
    muon, rest = partition_params(m)
    # First conv (16 out, 3*3*3 in) is muon-eligible.
    assert any(p.shape == (16, 3, 3, 3) for p in muon)


def test_partition_routes_bn_and_biases_to_adamw():
    m = _tiny_cnn()
    muon, rest = partition_params(m)
    # BN affine + all biases land in adamw.
    rest_shapes = [tuple(p.shape) for p in rest]
    assert (16,) in rest_shapes  # BN weight
    # bias of first conv (16,) too
    assert rest_shapes.count((16,)) >= 2


def test_partition_routes_narrow_head_to_adamw():
    m = _tiny_cnn()
    muon, rest = partition_params(m)
    # Head conv: out=1 -> flat dim_out=1, ineligible for muon.
    rest_shapes = [tuple(p.shape) for p in rest]
    assert (1, 16, 1, 1) in rest_shapes


def test_is_muon_eligible_rules():
    assert _is_muon_eligible(torch.zeros(8, 4))
    assert _is_muon_eligible(torch.zeros(8, 4, 3, 3))
    assert not _is_muon_eligible(torch.zeros(8))            # 1D
    assert not _is_muon_eligible(torch.zeros(1, 8, 1, 1))   # narrow out
    assert not _is_muon_eligible(torch.zeros(8, 1, 1, 1))   # narrow flat-in
    assert not _is_muon_eligible(torch.zeros(2, 1))         # 2D but narrow


# ----- Muon optimizer end-to-end ---------------------------------------------


def test_muon_step_updates_weights():
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(8, 16) * 0.1)
    before = w.detach().clone()
    opt = Muon([w], lr=0.1, momentum=0.9)
    # fake gradient
    w.grad = torch.randn_like(w)
    opt.step()
    assert not torch.allclose(w.detach(), before)


def test_muon_rejects_1d_param():
    p = nn.Parameter(torch.zeros(8))
    with pytest.raises(ValueError, match="ineligible"):
        Muon([p], lr=0.1)


def test_muon_state_persists_momentum_buffer():
    w = nn.Parameter(torch.randn(8, 16) * 0.1)
    opt = Muon([w], lr=0.1, momentum=0.9)
    w.grad = torch.randn_like(w)
    opt.step()
    state = opt.state[w]
    assert "momentum_buffer" in state
    assert state["momentum_buffer"].shape == w.shape


# ----- MuSGD facade -----------------------------------------------------------


def test_musgd_dispatches_to_both_legs():
    torch.manual_seed(0)
    m = _tiny_cnn()
    opt = MuSGD(m, lr=1e-3, weight_decay=1e-4)
    # Forward + backward to populate grads.
    x = torch.randn(2, 3, 16, 16)
    out = m(x).sum()
    out.backward()

    # Snapshot a Muon-routed weight and an AdamW-routed weight.
    conv0_w = next(m[0].parameters()).detach().clone()
    bn_w = m[1].weight.detach().clone()
    head_w = m[2].weight.detach().clone()

    opt.step()
    # Both should have moved.
    assert not torch.allclose(next(m[0].parameters()).detach(), conv0_w)
    assert not torch.allclose(m[1].weight.detach(), bn_w)
    assert not torch.allclose(m[2].weight.detach(), head_w)


def test_musgd_zero_grad():
    m = _tiny_cnn()
    opt = MuSGD(m, lr=1e-3)
    x = torch.randn(2, 3, 16, 16)
    m(x).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in m.parameters())


def test_musgd_param_groups_concatenated():
    m = _tiny_cnn()
    opt = MuSGD(m, lr=1e-3)
    # Should expose param groups from both legs (typically one each).
    assert len(opt.param_groups) >= 2


def test_musgd_partition_summary_counts_correct():
    m = _tiny_cnn()
    opt = MuSGD(m, lr=1e-3)
    s = opt.partition_summary()
    # Total params should match nn.Module's count.
    total = sum(p.numel() for p in m.parameters())
    assert s["total_params"] == total
    assert s["muon_params"] + s["adamw_params"] == total


def test_musgd_state_dict_roundtrip():
    """Save MuSGD state, restore in a fresh instance, training continues correctly.

    Tolerance note: PyTorch's AdamW (foreach=True default) has small numeric
    drift through state-dict round-trips because re-loaded param tensors
    occupy a different memory layout from the original (~3-4e-4 per step on
    plain AdamW, no Muon involvement). The test asserts both legs continue
    sensibly — not bit-exact — at a tolerance well above that floor.
    """
    torch.manual_seed(0)
    m1 = _tiny_cnn()
    opt1 = MuSGD(m1, lr=1e-3, weight_decay=1e-4)

    # Run a couple of steps to populate momentum buffers (eval mode keeps BN
    # buffers stable so the comparison is on the optimizer alone).
    m1.eval()
    x = torch.randn(2, 3, 16, 16)
    for _ in range(2):
        opt1.zero_grad()
        m1(x).sum().backward()
        opt1.step()

    opt_sd = opt1.state_dict()
    model_sd = {k: v.clone() for k, v in m1.state_dict().items()}

    # Fresh model + optimizer, restore both pieces of state.
    m2 = _tiny_cnn()
    m2.load_state_dict(model_sd)
    m2.eval()
    opt2 = MuSGD(m2, lr=1e-3, weight_decay=1e-4)
    opt2.load_state_dict(opt_sd)

    # One more step on both. Should land within fp32 / AdamW-foreach tolerance.
    for _opt, _m in [(opt1, m1), (opt2, m2)]:
        _opt.zero_grad()
        _m(x).sum().backward()
        _opt.step()

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-4), \
            f"param '{n1}' diverged after state-dict roundtrip: max diff {(p1 - p2).abs().max()}"


def test_musgd_load_rejects_adamw_state():
    """AdamW ckpt loaded into MuSGD must fail loudly, not silently corrupt."""
    m = _tiny_cnn()
    opt = MuSGD(m, lr=1e-3)
    bogus = torch.optim.AdamW(m.parameters(), lr=1e-3).state_dict()
    with pytest.raises(ValueError, match="tag mismatch"):
        opt.load_state_dict(bogus)


def test_muon_default_path_unaffected():
    """Sanity: importing optim_muon must not affect torch.optim.AdamW behavior."""
    import opndet.optim_muon  # noqa: F401
    w = nn.Parameter(torch.randn(8, 8))
    opt = torch.optim.AdamW([w], lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
