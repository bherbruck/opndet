from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    from opndet.train import train
    train(args.config, run_name=args.run_name, runs_dir=args.runs_dir, resume=args.resume)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from opndet.config import ModelConfig
    from opndet.export import build_pt_model, export_onnx, verify_onnx
    from opndet.presets import resolve

    if args.model:
        from opndet.yaml_build import build_model_from_yaml
        m = build_model_from_yaml(resolve(args.model)).eval()
        if args.ckpt:
            import torch
            sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
            m.load_state_dict(sd["model"] if "model" in sd else sd)
        import torch
        c, h, w = m.input_shape
        dummy = torch.randn(1, c, h, w)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(m, dummy, str(out_path),
                          input_names=["image"], output_names=["output"],
                          opset_version=args.opset, do_constant_folding=True,
                          dynamic_axes=None, dynamo=False)
        print(f"exported: {out_path}")
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
    from opndet.predict import predict_image
    from opndet.presets import resolve

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
    print(f"input shape:  (3, {m.input_shape[1]}, {m.input_shape[2]})")
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
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("export", help="Export to ONNX (opset 13)")
    pe.add_argument("--ckpt", default=None, help="Trained checkpoint .pt")
    pe.add_argument("--out", default="opndet.onnx", help="Output ONNX path")
    pe.add_argument("--opset", type=int, default=13)
    pe.add_argument("--model", default=None, help="Preset name (bbox-n|s|m) or path to YAML")
    pe.set_defaults(func=_cmd_export)

    pp = sub.add_parser("predict", help="Run inference on a single image")
    pp.add_argument("--image", required=True)
    pp.add_argument("--model", required=True, help="Preset name (bbox-n|s|m) or path to YAML")
    pp.add_argument("--ckpt", default=None)
    pp.add_argument("--threshold", type=float, default=0.3)
    pp.add_argument("--stride", type=int, default=4)
    pp.add_argument("--device", default="cpu")
    pp.add_argument("--save", default=None, help="Save visualization to this path")
    pp.set_defaults(func=_cmd_predict)

    pi = sub.add_parser("info", help="Show model info; pass preset name or YAML path")
    pi.add_argument("model", nargs="?", help="Preset (bbox-n|s|m) or path. Omit to list presets.")
    pi.set_defaults(func=_cmd_info)

    pinit = sub.add_parser("init-config", help="Write the bundled training config template to stdout (or --out path)")
    pinit.add_argument("--out", default="-", help="Path or - for stdout")
    pinit.set_defaults(func=_cmd_init_config)

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
