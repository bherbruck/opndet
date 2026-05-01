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
) -> dict[str, torch.Tensor]:
    """Soft-label KD on the deployed output. At cells where the teacher's peak-suppressed obj > gate,
    push the student's sigmoid(obj_logit) and (cxy, wh) to match the teacher's. Where the teacher
    didn't fire, no signal. The student's GT-driven loss handles the rest."""
    t_obj = teacher_out[:, 0:1]
    t_cxy = teacher_out[:, 1:3]
    t_wh = teacher_out[:, 3:5]

    s_obj = torch.sigmoid(student_raw[:, 0:1])
    s_cxy = torch.sigmoid(student_raw[:, 1:3])
    s_wh = torch.sigmoid(student_raw[:, 3:5])

    gate = (t_obj > conf_gate).float()
    n_gate = gate.sum().clamp(min=1.0)

    eps = 1e-6
    s_obj_c = s_obj.clamp(eps, 1.0 - eps)
    bce = -(t_obj * s_obj_c.log() + (1.0 - t_obj) * (1.0 - s_obj_c).log())
    l_hm = (bce * gate).sum() / n_gate

    l_xy = F.l1_loss(s_cxy, t_cxy, reduction="none").sum(dim=1, keepdim=True)
    l_wh = F.l1_loss(s_wh, t_wh, reduction="none").sum(dim=1, keepdim=True)
    l_reg = ((l_xy + l_wh) * gate).sum() / n_gate

    return {
        "l_kd_hm": l_hm,
        "l_kd_reg": l_reg,
        "l_kd": hm_weight * l_hm + reg_weight * l_reg,
    }
