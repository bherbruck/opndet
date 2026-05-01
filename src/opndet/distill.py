from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F


def load_teacher(ckpt_path: str | Path, device: torch.device | str) -> tuple[torch.nn.Module, str]:
    """Build teacher from its saved config, load weights, set eval/no-grad."""
    from opndet.calibrate import apply_temperature
    from opndet.presets import resolve as _resolve_preset
    from opndet.yaml_build import build_model_from_yaml

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(sd, dict) or "config" not in sd:
        raise ValueError(
            f"teacher ckpt {ckpt_path} missing 'config' field — re-save it from a current opndet train run "
            "so the architecture is self-describing."
        )
    teacher_preset = sd["config"]["model_config"]
    teacher = build_model_from_yaml(_resolve_preset(teacher_preset)).to(device).eval()
    teacher.load_state_dict(sd["model"])

    T = float(sd.get("temperature", 1.0))
    if T != 1.0:
        apply_temperature(teacher, T)

    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, teacher_preset


def distillation_loss(
    student_raw: torch.Tensor,        # [B, 5+, H, W] pre-sigmoid
    teacher_out: torch.Tensor,        # [B, 5, H, W] post-sigmoid + peak-suppressed
    conf_gate: float = 0.5,
    hm_weight: float = 1.0,
    reg_weight: float = 0.5,
    full_distill: bool = False,
    neg_gate: float = 0.0,
    kd_temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Soft-label KD on the deployed output.

    Heatmap (obj) supervision modes:
      - full_distill=False (default), neg_gate=0: only positive KD where teacher_obj > conf_gate.
        Background cells get NO KD signal — VFL/focal on the GT side handles "be 0 here".
      - full_distill=False, neg_gate>0: asymmetric. Positive supervision where teacher_obj > conf_gate
        AND negative supervision where teacher_obj < neg_gate. The "uncertain middle" is unsupervised.
      - full_distill=True: supervise EVERY cell, matching teacher's full distribution. Strongest
        anti-false-positive signal. Use when student over-predicts at low confidence.

    Box regression (cxy, wh) is always gated to teacher-confident cells — matching boxes at
    empty-background cells is meaningless.

    kd_temperature softens teacher's targets via sigmoid(teacher_logit / T_kd). Default 1.0 is no-op.
    Increase (e.g. 2.0) to make teacher targets less confident — useful when teacher is over-sharp
    and student has trouble matching the bimodal extremes.
    """
    t_obj = teacher_out[:, 0:1]
    t_cxy = teacher_out[:, 1:3]
    t_wh = teacher_out[:, 3:5]

    if kd_temperature != 1.0:
        # Soften teacher's obj targets. Use logit space so the operation is well-defined for 0/1.
        eps = 1e-6
        t_obj_clamp = t_obj.clamp(eps, 1.0 - eps)
        t_logit = torch.log(t_obj_clamp / (1.0 - t_obj_clamp))
        t_obj = torch.sigmoid(t_logit / kd_temperature)

    s_obj = torch.sigmoid(student_raw[:, 0:1])
    s_cxy = torch.sigmoid(student_raw[:, 1:3])
    s_wh = torch.sigmoid(student_raw[:, 3:5])

    eps = 1e-6
    s_obj_c = s_obj.clamp(eps, 1.0 - eps)
    bce_per_cell = -(t_obj * s_obj_c.log() + (1.0 - t_obj) * (1.0 - s_obj_c).log())

    # Box regression: always gated to teacher-confident cells.
    pos_gate = (t_obj > conf_gate).float()
    n_pos = pos_gate.sum().clamp(min=1.0)
    l_xy = F.l1_loss(s_cxy, t_cxy, reduction="none").sum(dim=1, keepdim=True)
    l_wh = F.l1_loss(s_wh, t_wh, reduction="none").sum(dim=1, keepdim=True)
    l_reg = ((l_xy + l_wh) * pos_gate).sum() / n_pos

    # Heatmap KD: full | gated (positive only) | asymmetric (positive + negative).
    if full_distill:
        l_hm = bce_per_cell.mean()
    elif neg_gate > 0:
        neg_mask = (t_obj < neg_gate).float()
        gate = pos_gate + neg_mask
        n_gate = gate.sum().clamp(min=1.0)
        l_hm = (bce_per_cell * gate).sum() / n_gate
    else:
        l_hm = (bce_per_cell * pos_gate).sum() / n_pos

    return {
        "l_kd_hm": l_hm,
        "l_kd_reg": l_reg,
        "l_kd": hm_weight * l_hm + reg_weight * l_reg,
    }
