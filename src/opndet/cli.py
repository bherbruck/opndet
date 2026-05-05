from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from opndet.train import train
    train(args.config, run_name=args.run_name, runs_dir=args.runs_dir, resume=args.resume,
          teacher=args.teacher, self_distill=args.self_distill)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from opndet.config import ModelConfig
    from opndet.export import build_pt_model, export_onnx, verify_onnx
    from opndet.presets import resolve

    if args.model:
        from opndet.export import _InputNormalizer
        from opndet.yaml_build import build_model_from_yaml
        m = build_model_from_yaml(resolve(args.model)).eval()
        if args.ckpt:
            import torch
            sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            m.load_state_dict(sd["model"] if "model" in sd else sd)
            T = float(sd.get("temperature", 1.0)) if isinstance(sd, dict) else 1.0
            if T != 1.0:
                from opndet.calibrate import apply_temperature
                apply_temperature(m, T)
                print(f"baking calibration temperature T={T:.4f} into the graph")
        import torch
        c, h, w = m.input_shape
        if args.bake_input_norm:
            if c != 3:
                print(f"FAIL: --bake-input-norm only supports 3-ch models; this model has in_ch={c} "
                      "(temporal-prior variants take a 4th channel that is not photometric)",
                      file=sys.stderr)
                return 2
            m = _InputNormalizer(m).eval()
            dummy = torch.rand(1, c, h, w) * 255.0
        else:
            dummy = torch.randn(1, c, h, w)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(m, dummy, str(out_path),
                          input_names=["image"], output_names=["output"],
                          opset_version=args.opset, do_constant_folding=True,
                          dynamic_axes=None, dynamo=False)
        print(f"exported: {out_path}{' (with input norm baked in: expects raw 0-255)' if args.bake_input_norm else ''}")
        return 0

    cfg = ModelConfig()
    m = build_pt_model(args.ckpt, cfg)
    path = export_onnx(m, args.out, cfg=cfg, opset=args.opset)
    print(f"exported: {path}")
    info = verify_onnx(path, m)
    print(f"ops: {info['ops']}")
    print(f"max_abs_diff: {info['max_abs_diff']:.2e}  cosine_sim: {info['cosine_sim']:.6f}  parity: {info['atol_pass']}")
    return 0 if info["atol_pass"] else 1


def _cmd_predict(args: argparse.Namespace) -> int:
    from opndet.predict import predict_image, predict_video
    from opndet.presets import resolve

    if args.video:
        stats = predict_video(
            video_path_or_url=args.video,
            model_config=resolve(args.model),
            ckpt=args.ckpt,
            threshold=args.threshold,
            device=args.device,
            save_path=args.save or "predict_out.mp4",
            stride=args.stride,
            max_frames=args.max_frames,
        )
        print(json.dumps(stats, indent=2))
        return 0

    if not args.image:
        print("FAIL: must pass --image PATH or --video URL_OR_PATH", file=sys.stderr)
        return 2

    results = predict_image(
        image_path=args.image,
        model_config=resolve(args.model),
        ckpt=args.ckpt,
        threshold=args.threshold,
        device=args.device,
        save_path=args.save,
        stride=args.stride,
    )
    print(json.dumps(results, indent=2))
    if args.save:
        print(f"vis saved: {args.save}", file=sys.stderr)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from opndet.calibrate import calibrate_ckpt
    out = calibrate_ckpt(args.ckpt, args.config, split=args.split, save=not args.dry_run)
    print(f"T={out['temperature']:.4f}  ECE {out['ece_before']:.4f} -> {out['ece_after']:.4f}  n={out['n_samples']}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from opndet.eval import run_eval
    out = run_eval(
        ckpt_path=args.ckpt,
        config_path=args.config,
        split=args.split,
        out_dir=args.out,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
        batch_size=args.batch_size,
        stability=args.stability,
        n_perturbations=args.n_perturbations,
        auto_threshold=args.auto_threshold,
    )
    s = out["report"]["summary"]
    cs = out["report"]["counts"]
    print(f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}  mAP@.5={s['map50']:.3f} mAP@.5:.95={s['map_50_95']:.3f}")
    print(f"count exact={cs['exact_count_frac']:.1%}  abs_err mean={cs['abs_err_mean']:.2f} p95={cs['abs_err_p95']:.0f}")
    print(f"out: {out['out_dir']}")
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    from opndet.quantize import parity_check, quantize_onnx
    info = quantize_onnx(args.onnx, args.out, args.calib, n_calib=args.n_calib, quant_format=args.format)
    print(f"int8: {info['int8_path']}")
    print(f"  fp32 size:  {info['fp32_bytes']/1024:.1f} KB")
    print(f"  int8 size:  {info['int8_bytes']/1024:.1f} KB")
    print(f"  ratio:      {info['compression']:.2f}x")
    print(f"  calibrated on {info['n_calibration_images']} images")
    if args.verify:
        diff = parity_check(args.onnx, args.out, args.calib, n_check=16)
        print(f"parity (n={diff['n']}): obj_mae={diff['obj_mean_abs_diff']:.2e}  all_mae={diff['all_mean_abs_diff']:.2e}")
    return 0


def _cmd_init_config(args: argparse.Namespace) -> int:
    from opndet.presets import bundled_train_template
    src = Path(bundled_train_template())
    content = src.read_text()
    if args.out == "-":
        sys.stdout.write(content)
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    print(f"wrote: {out}", file=sys.stderr)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from opndet.export import ALLOWED_OPS
    from opndet.presets import list_presets, resolve
    from opndet.yaml_build import build_model_from_yaml

    if args.model is None:
        print("bundled presets:")
        for p in list_presets():
            print(f"  {p}")
        return 0

    path = resolve(args.model)
    m = build_model_from_yaml(path)
    n = sum(p.numel() for p in m.parameters())
    print(f"resolved:     {path}")
    print(f"input shape:  ({m.input_shape[0]}, {m.input_shape[1]}, {m.input_shape[2]})")
    print(f"params:       {n:,}  ({n/1e6:.2f}M)")
    print(f"layers:       {len(m.layers)}")
    named = sorted(m.aliases.keys())
    print(f"named layers: {named[:20]}{' ...' if len(named) > 20 else ''}  ({len(named)} total)")
    print(f"output specs: {[s['name'] for s in m._out_specs]}")
    print(f"opset 13 allowed ops: {len(ALLOWED_OPS)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="opndet", description="Tiny single-class detector. OpenVINO 2022 opset compatible.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Train a model")
    pt.add_argument("--config", required=True, help="Path to training YAML config")
    pt.add_argument("--run-name", default=None, help="Override config 'name' (the run subdir)")
    pt.add_argument("--runs-dir", default=None, help="Override config 'runs_dir' (parent of run dirs)")
    pt.add_argument("--resume", default=None, help="Resume from ckpt .pt or run dir (uses last.pt)")
    distill_group = pt.add_mutually_exclusive_group()
    distill_group.add_argument("--teacher", default=None,
                               help="Path to a trained teacher .pt; enables knowledge distillation. "
                                    "Architecture is read from the teacher's saved config.")
    distill_group.add_argument("--self-distill", action="store_true",
                               help="Use the model's own EMA shadow as the teacher. Requires ema_decay > 0.")
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("export", help="Export to ONNX (opset 13)")
    pe.add_argument("--ckpt", default=None, help="Trained checkpoint .pt")
    pe.add_argument("--out", default="opndet.onnx", help="Output ONNX path")
    pe.add_argument("--opset", type=int, default=13)
    pe.add_argument("--model", default=None, help="Preset name (bbox-n|s|m) or path to YAML")
    pe.add_argument("--bake-input-norm", action="store_true",
                    help="Prepend ImageNet mean/std normalization to the graph. "
                         "Use for embedded deployment (depthai, OpenVINO) that "
                         "passes raw uint8 0-255 RGB frames without preprocessing.")
    pe.set_defaults(func=_cmd_export)

    pp = sub.add_parser("predict", help="Run inference on an image or video")
    pp.add_argument("--image", default=None, help="Path to a single image")
    pp.add_argument("--video", default=None,
                    help="Path or URL to a video. URLs use yt-dlp (must be installed).")
    pp.add_argument("--model", required=True, help="Preset (bbox-n|s|m|p|f) or path to YAML")
    pp.add_argument("--ckpt", default=None)
    pp.add_argument("--threshold", type=float, default=0.3)
    pp.add_argument("--stride", type=int, default=4)
    pp.add_argument("--device", default="cpu")
    pp.add_argument("--save", default=None,
                    help="Save annotated output. For --image: PNG/JPG. For --video: MP4 (default predict_out.mp4)")
    pp.add_argument("--max-frames", type=int, default=None, help="Cap video frame count (debug)")
    pp.set_defaults(func=_cmd_predict)

    pi = sub.add_parser("info", help="Show model info; pass preset name or YAML path")
    pi.add_argument("model", nargs="?", help="Preset (bbox-n|s|m) or path. Omit to list presets.")
    pi.set_defaults(func=_cmd_info)

    pinit = sub.add_parser("init-config", help="Write the bundled training config template to stdout (or --out path)")
    pinit.add_argument("--out", default="-", help="Path or - for stdout")
    pinit.set_defaults(func=_cmd_init_config)

    pcal = sub.add_parser("calibrate", help="Fit Platt-style temperature on val split; bake it into the ckpt")
    pcal.add_argument("--ckpt", required=True, help="Trained checkpoint .pt")
    pcal.add_argument("--config", default=None, help="Training YAML (optional — falls back to the ckpt's saved config)")
    pcal.add_argument("--split", default="val", choices=["train", "val", "test"], help="Which split to fit on")
    pcal.add_argument("--dry-run", action="store_true", help="Compute T but don't write back to ckpt")
    pcal.set_defaults(func=_cmd_calibrate)

    pev = sub.add_parser("eval", help="Run full validation suite (Hungarian-matched, calibration, count, size strata) and write report")
    pev.add_argument("--ckpt", required=True, help="Trained checkpoint .pt")
    pev.add_argument("--config", default=None, help="Training YAML (optional — falls back to the ckpt's saved config)")
    pev.add_argument("--split", default="val", choices=["train", "val", "test"], help="Which split to evaluate")
    pev.add_argument("--out", default=None, help="Output dir (default: <ckpt_dir>/eval_<split>)")
    pev.add_argument("--score-thresh", type=float, default=None, help="Confidence threshold for fixed-threshold metrics (default: cfg.eval_threshold)")
    pev.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for TP/FN matching")
    pev.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    pev.add_argument("--stability", action="store_true",
                     help="Run perturbation-stability metric (proxy for flapping). ~Nx slower (default N=8).")
    pev.add_argument("--n-perturbations", type=int, default=8,
                     help="Number of small perturbations per image when --stability is set")
    pev.add_argument("--auto-threshold", action="store_true",
                     help="After the first pass, snap score_thresh to the F1-optimal value from the PR sweep "
                          "and recompute fixed-threshold metrics. Honest reporting when the chosen threshold is off the knee.")
    pev.set_defaults(func=_cmd_eval)

    pq = sub.add_parser("quantize", help="Static int8 PTQ on a trained ONNX")
    pq.add_argument("--onnx", required=True, help="Input fp32 ONNX")
    pq.add_argument("--out", required=True, help="Output int8 ONNX")
    pq.add_argument("--calib", required=True, help="Image directory for calibration")
    pq.add_argument("--n-calib", type=int, default=100, help="Number of calibration images")
    pq.add_argument("--format", default="qdq", choices=["qdq", "qoperator"], help="Quantization format")
    pq.add_argument("--verify", action="store_true", help="After quant, run parity check vs fp32")
    pq.set_defaults(func=_cmd_quantize)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
