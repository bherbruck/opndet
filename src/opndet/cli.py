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
        from opndet.export import _DiagnosticWrapper, _InputNormalizer
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
        diag_names: list[str] = []
        if args.diagnostic:
            with torch.no_grad():
                cache = m._run(torch.zeros(1, c, h, w))
            for name, idx in sorted(m.aliases.items(), key=lambda kv: kv[1]):
                if cache[idx].ndim == 4:
                    diag_names.append(name)
            m = _DiagnosticWrapper(m, diag_names).eval()
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
        output_names = ["output"] + diag_names
        torch.onnx.export(m, dummy, str(out_path),
                          input_names=["image"], output_names=output_names,
                          opset_version=args.opset, do_constant_folding=True,
                          dynamic_axes=None, dynamo=False)
        # Tier-aware op allowlist check (Phase 2). The underlying YamlModel
        # carries .tier; for _DiagnosticWrapper / _InputNormalizer wrappers we
        # resolve through the inner model.
        from opndet.export import allowed_ops_for_tier, check_resize_attrs
        import onnx as _onnx
        inner = m
        while hasattr(inner, "model") and not hasattr(inner, "tier"):
            inner = inner.model
        tier = getattr(inner, "tier", "edge")
        om = _onnx.load(str(out_path))
        ops = {n.op_type for n in om.graph.node}
        forbidden = ops - allowed_ops_for_tier(tier)
        if forbidden:
            print(f"FAIL: {out_path} (tier={tier}) has forbidden ops: {sorted(forbidden)}",
                  file=sys.stderr)
            return 3
        check_resize_attrs(om, tier=tier)
        suffix_parts = [f"tier={tier}"]
        if args.bake_input_norm:
            suffix_parts.append("with input norm baked in: expects raw 0-255")
        if args.diagnostic:
            suffix_parts.append(f"diagnostic: +{len(diag_names)} layer outputs")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        print(f"exported: {out_path}{suffix}")
        if args.diagnostic:
            print(f"  diagnostic outputs: {diag_names}")
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


def _cmd_analyze(args: argparse.Namespace) -> int:
    from opndet.analyze import run as analyze_run
    from opndet.presets import resolve
    from pathlib import Path
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda not available; falling back to cpu", file=sys.stderr)
        device = "cpu"
    out = args.out
    if out is None:
        ckpt_p = Path(args.ckpt)
        out = str(ckpt_p.parent / f"analyze_{ckpt_p.stem}")
    if not args.image and not args.video:
        print("FAIL: pass --image PATH (repeatable) or --video PATH", file=sys.stderr)
        return 2
    analyze_run(
        ckpt=args.ckpt,
        model_yaml=resolve(args.model),
        out_dir=Path(out),
        device=device,
        threshold=args.threshold,
        video=args.video,
        n_frames=args.frames,
        images=args.image,
    )
    return 0


def _cmd_mine_negatives(args: argparse.Namespace) -> int:
    from opndet.mine_negatives import mine
    out = args.out
    if out is None:
        ckpt_p = Path(args.ckpt)
        out = str(ckpt_p.parent / f"hard_negatives_{ckpt_p.stem}")
    res = mine(
        ckpt=args.ckpt,
        config=args.config,
        out_dir=out,
        split=args.split,
        max_samples=args.max_samples,
        top_k_per_sample=args.top_k_per_sample,
        patch_size=args.patch_size,
        score_thresh=args.score_thresh,
        clusters=args.clusters,
        device=args.device,
    )
    print(f"\nsummary: {res['n_patches']} patches, {res['n_samples_with_ghosts']} samples with ghosts")
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


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from opndet.dashboard import serve
    root = args.root or args.run
    if root is None:
        print("FAIL: pass --root <runs_parent_or_single_run_dir>", file=sys.stderr)
        return 2
    serve(root_dir=root, host=args.host, port=args.port)
    return 0


def _cmd_sam_obb(args: argparse.Namespace) -> int:
    from opndet.sam_preprocess import run as sam_run
    stats = sam_run(
        coco_json=args.coco,
        images_dir=args.images,
        out_dir=args.out,
        sam_model=args.sam_model,
        device=args.device,
        max_images=args.max_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_rejected=args.save_rejected,
    )
    print(f"processed={stats.n_images_processed} skipped={stats.n_images_skipped} "
          f"obj={stats.n_objects_processed} obb={stats.n_obb_extracted} "
          f"round={stats.n_aabb_fallback} drop={stats.n_invalid_dropped} "
          f"errors={len(stats.errors)} ({stats.duration_seconds}s)")
    print(f"  drop breakdown: no_corners={stats.n_drop_no_corners} "
          f"geometry={stats.n_drop_geometry} area={stats.n_drop_area} "
          f"centroid={stats.n_drop_centroid}")
    print(f"manifest: {Path(args.out) / 'manifest.json'}")
    rejected_dir = Path(args.out) / "_rejected"
    if rejected_dir.exists():
        n_saved = len(list(rejected_dir.glob("*.png")))
        if n_saved > 0:
            print(f"rejected previews: {n_saved} in {rejected_dir}")
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
    pe.add_argument("--diagnostic", action="store_true",
                    help="Expose every named 4D layer activation as an additional ONNX output. "
                         "Used by the webui's explain mode (per-layer activation slider). "
                         "Production graph (no flag) is unchanged.")
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

    pa = sub.add_parser("analyze", help="Postmortem on a saved checkpoint: per-layer activation "
                                          "slider + Grad-CAM per detection, rendered as standalone HTML.")
    pa.add_argument("--ckpt", required=True, help="Trained checkpoint .pt")
    pa.add_argument("--model", required=True, help="Preset name (bbox-x|s|m|...) or YAML path")
    pa.add_argument("--image", action="append", default=None,
                    help="Path to a single image (repeatable: --image a.jpg --image b.jpg)")
    pa.add_argument("--video", default=None, help="Path to a video file; samples N evenly-spaced frames")
    pa.add_argument("--frames", type=int, default=6, help="Number of frames to sample (video mode)")
    pa.add_argument("--out", default=None,
                    help="Output dir (default: <ckpt-dir>/analyze_<ckpt-stem>)")
    pa.add_argument("--device", default="cuda",
                    help="cuda or cpu; falls back to cpu if cuda not available")
    pa.add_argument("--threshold", type=float, default=0.5,
                    help="Detection score threshold for which detections get Grad-CAM")
    pa.set_defaults(func=_cmd_analyze)

    pd = sub.add_parser("dashboard", help="Run-metrics web viewer (DuckDB-backed). "
                                            "Pass a single run dir or a runs parent — "
                                            "auto-discovers every metrics.duckdb under it.")
    pd.add_argument("--root", default=None,
                    help="Runs parent OR a single run dir. Auto-scans for metrics.duckdb. "
                         "Tolerant of missing/empty paths — picks up new runs as they appear.")
    pd.add_argument("--run", default=None,
                    help="(legacy alias for --root; either flag works)")
    pd.add_argument("--host", default="127.0.0.1", help="Bind host")
    pd.add_argument("--port", type=int, default=5000, help="Bind port")
    pd.set_defaults(func=_cmd_dashboard)

    pmn = sub.add_parser("mine-negatives",
                         help="Mine hard-negative patches from a trained ckpt's ghosts (Grad-CAM-driven). "
                              "Writes a patch pool + manifest.json suitable for augment's hard_negative_pool.")
    pmn.add_argument("--ckpt", required=True, help="Trained checkpoint .pt")
    pmn.add_argument("--config", default=None, help="Training YAML (optional — falls back to ckpt's saved config)")
    pmn.add_argument("--split", default="val", choices=["train", "val", "test"])
    pmn.add_argument("--max-samples", type=int, default=500, help="Cap on number of images to scan")
    pmn.add_argument("--top-k-per-sample", type=int, default=4,
                     help="Keep up to K highest-score ghosts per image")
    pmn.add_argument("--patch-size", type=int, default=32, help="Side length of mined patches (px)")
    pmn.add_argument("--score-thresh", type=float, default=None,
                     help="Confidence threshold for the ghost test (default: cfg.eval_threshold or 0.3)")
    pmn.add_argument("--clusters", type=int, default=8, help="K-means cluster count (sklearn). Set to 1 to skip.")
    pmn.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    pmn.add_argument("--out", default=None,
                     help="Output dir (default: <ckpt-dir>/hard_negatives_<ckpt-stem>)")
    pmn.set_defaults(func=_cmd_mine_negatives)

    pso = sub.add_parser("sam-obb",
                         help="Preprocess: SAM2 + COCO AABBs → YOLOv8-OBB *.txt files. "
                              "One-shot per dataset, idempotent. "
                              "Output OBB coords may extend outside [0,1] for objects "
                              "truncated at frame edges — intentional.")
    pso.add_argument("--coco", required=True, help="Path to COCO _annotations.coco.json")
    pso.add_argument("--images", required=True, help="Directory of images referenced by the COCO file")
    pso.add_argument("--out", required=True, help="Output dir for per-image *.txt + manifest.json")
    pso.add_argument("--sam-model", default="sam2_b",
                     help="SAM2 size: sam2_t|s|b|l or sam2.1_t|s|b|l, or raw HF id (default: sam2_b)")
    pso.add_argument("--device", default="cuda", help="cuda or cpu")
    pso.add_argument("--max-images", type=int, default=None, help="Cap number of images (debug)")
    pso.add_argument("--batch-size", type=int, default=8,
                     help="Images per SAM2 image-encoder batch. Bump on big GPUs (16-32 on A100/H100); "
                          "drop to 4 if OOM on T4. Higher = better GPU utilization (default: 8)")
    pso.add_argument("--num-workers", type=int, default=8,
                     help="ThreadPoolExecutor workers for parallel disk reads during preload (default: 8)")
    pso.add_argument("--save-rejected", type=int, default=16,
                     help="Per-rule cap on rejected-OBB diagnostic previews saved to "
                          "<out>/_rejected/ (default: 16; 0 to disable). Each preview is a "
                          "3-panel image: AABB prompt | SAM mask | candidate OBB.")
    pso.set_defaults(func=_cmd_sam_obb)

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
