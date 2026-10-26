#!/usr/bin/env python3
"""Disposable GPU smoke/timing on the first real Tr batch; no V/Final access.

Loads the frozen H1 and original data/normalizer bindings. The benchmark never
writes model checkpoints or computes AP. CUDA-event and synchronized wall times
include the actual head validation/loss operations, but exclude data loading,
checkpointing, gradient diagnostics, and epoch metrics. Parameters are not speed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WARMUP, REPEATS = 3, 10
AUXILIARY_WEIGHT = .05


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in ("backbone", "reference_plan", "source", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.suffix.lower() != ".json":
        parser.error("--output must name a fresh JSON file")
    return args


def require_idle_gpu():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=15)
    others = [line.strip() for line in result.stdout.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    if others:
        raise RuntimeError("Refusing GPU benchmark alongside compute PIDs: " + ", ".join(others))


def measure(function, torch, device, batch_size):
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    event_ms, wall_ms = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(REPEATS):
        torch.cuda.synchronize(device)
        wall_start = time.perf_counter()
        start.record()
        function()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        event_ms.append(start.elapsed_time(end))
    if not all(math.isfinite(v) and v > 0 for v in event_ms + wall_ms):
        raise RuntimeError("Timing contains a nonfinite or nonpositive duration")
    peak_allocated = torch.cuda.max_memory_allocated(device)
    return dict(warmup=WARMUP, repetitions=REPEATS, cuda_event_ms=event_ms, wall_ms=wall_ms,
                mean_cuda_event_ms=statistics.mean(event_ms), median_cuda_event_ms=statistics.median(event_ms),
                mean_wall_ms=statistics.mean(wall_ms), median_wall_ms=statistics.median(wall_ms),
                samples_per_second=batch_size * 1000 / statistics.mean(wall_ms),
                baseline_allocated_bytes=baseline_allocated, baseline_reserved_bytes=baseline_reserved,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                incremental_peak_allocated_bytes=peak_allocated - baseline_allocated)


def execute(args, report, save):
    sys.path.insert(0, str(args.source))
    import torch
    from torch.utils.data._utils.collate import default_collate
    import fwt_v2
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import DiffusionConfig
    from fwt_v2.engine import audited_manifest, load_det, model_state_digest, seed_everything, to_device
    from fwt_v2.literature_refiner import DecoderFeatureRefiner, masked_refinement_loss
    from fwt_v2.logit_refiner import ConditionalLogitRefiner
    from fwt_v2.manifest import manifest_digest, samples_for_role

    if Path(fwt_v2.__file__).resolve().parent != args.source / "fwt_v2":
        raise RuntimeError("Imported fwt_v2 is not the requested source")
    require_idle_gpu()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Benchmark requires exactly one visible CUDA GPU")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda:0")
    seed_everything(20261026)
    plan = read_json(args.reference_plan)
    metadata = plan["bound_data_metadata_paths"]
    manifest_path, normalizer_path = Path(metadata["manifest"]), Path(metadata["normalizer"])
    if args.output in {args.backbone, args.reference_plan, manifest_path, normalizer_path}:
        raise ValueError("Output collides with an input")
    watched = {str(args.reference_plan): sha(args.reference_plan), str(args.backbone): sha(args.backbone)}
    expected_backbone = plan["input_files_sha256"][plan["input_paths"]["backbone"]]
    if watched[str(args.backbone)] != expected_backbone:
        raise ValueError("H1 checkpoint bytes differ from the reference plan")
    for key in ("manifest", "normalizer", "weather", "arrays_index"):
        path = Path(metadata[key])
        digest = sha(path)
        if digest != plan["input_files_sha256"][str(path)]:
            raise ValueError("Reference metadata bytes changed: " + key)
        watched[str(path)] = digest
    source_files = list((args.source / "fwt_v2").glob("*.py")) + [Path(__file__).resolve()]
    watched.update({str(path): sha(path) for path in source_files})
    report.update(status="preparing_real_Tr", source=str(args.source),
                  environment=dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(device), cpu_threads=2, precision="float32",
                    cuda_matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
                    deterministic_algorithms=False), reference_plan_sha256=watched[str(args.reference_plan)])
    save()
    os.chdir(manifest_path.parent)  # Retain the original relative weather locator in DataConfig.
    manifest = audited_manifest(manifest_path)
    if manifest_digest(manifest) != plan["manifest_sha256"] or sha(normalizer_path) != plan["normalizer_sha256"]:
        raise ValueError("Original manifest or normalizer semantic identity changed")
    train_rows = samples_for_role(manifest, "residual_train")
    if [row["sample_id"] for row in train_rows] != plan["training_sample_ids"]:
        raise ValueError("Tr identity/order differs from reference")
    if args.batch_size > len(train_rows):
        raise ValueError("Requested batch exceeds available Tr samples")
    selected_rows = train_rows[:args.batch_size]
    index = {(r["sample_id"], r["field"]): r for r in read_json(metadata["arrays_index"])["files"]}
    selected_files = []
    for row in selected_rows:
        for field in ("input_path", "label_path"):
            bound = index[(row["sample_id"], field)]
            path, digest = Path(row[field]), sha(row[field])
            if bound["role"] != "residual_train" or bound["sha256"] != digest:
                raise ValueError("Selected Tr array byte identity failed")
            watched[str(path.resolve())] = digest
            selected_files.append(dict(sample_id=row["sample_id"], field=field, path=str(path.resolve()), sha256=digest))
    data_config = DataConfig(**plan["data_config"])
    dataset = RCDADataset(manifest, "residual_train", NormalizationStats.load(normalizer_path), data_config)
    # No sampler, no label-based selection, no access to any other role's arrays.
    cpu_batch = default_collate([dataset[index] for index in range(args.batch_size)])
    if list(cpu_batch["sample_id"]) != [row["sample_id"] for row in selected_rows]:
        raise RuntimeError("Loaded batch identity differs from fixed Tr prefix")
    if tuple(cpu_batch["x_norm"].shape) != (args.batch_size, 22, 256, 256):
        raise ValueError("Expected actual [B,22,256,256] inputs")
    report["data"] = dict(role="residual_train", selection="first_manifest_order_batch_no_label_search",
        sample_ids=list(cpu_batch["sample_id"]), array_files=selected_files,
        input_shape=list(cpu_batch["x_norm"].shape), validation_arrays_read=0, final_arrays_read=0,
        preprocessing="original_RCDADataset_and_Tb_normalizer", weather_source_verified=True)
    save()
    require_idle_gpu()  # Weather verification may take minutes; recheck before allocation.
    backbone, bundle = load_det(args.backbone, device)
    for key in ("model_config", "data_config"):
        if canonical(bundle[key]) != canonical(plan[key]):
            raise ValueError("Loaded H1 configuration differs from reference: " + key)
    if not bundle.get("selection_frozen") or bundle["family"] != "det":
        raise ValueError("A completed selected H1 is required")
    backbone.requires_grad_(False).eval()
    backbone_before = model_state_digest(backbone)
    batch = to_device(cpu_batch, device)
    with torch.no_grad():
        original = backbone(batch["x_norm"], batch["physics"])
        cached = backbone(batch["x_norm"], batch["physics"], return_decoder_features=True)
    if not torch.equal(original["logits_det"], cached["logits_det"]):
        raise RuntimeError("H1 optional decoder exposure failed exact logit parity")
    for key in ("relational_feature", "modality_mask", "direction_weights"):
        if original[key] is not None and not torch.equal(original[key], cached[key]):
            raise RuntimeError("H1 condition changed: " + key)
    for key in ("fused_pyramid", "gate_weights"):
        if not all(torch.equal(a, b) for a, b in zip(original[key], cached[key])):
            raise RuntimeError("H1 condition changed: " + key)
    del original
    report["parity"] = dict(default_vs_decoder_feature_logits_exact=True,
        backbone_before_state_sha256=backbone_before, decoder_feature_shape=list(cached["decoder_feature"].shape))
    report["smoke"], report["timings"], report["parameter_counts"] = {}, {}, {}
    config = DiffusionConfig(**dict(plan["diffusion_config"], prediction_type="clean_probability", output_initialization="zero"))

    def create_head(kind):
        seed_everything(20261026)  # Identical initial head bytes across loss variants.
        head = ConditionalLogitRefiner(config, loss="bce") if kind == "unet" else DecoderFeatureRefiner()
        return head.to(device)

    def frozen_gate():
        if any(p.grad is not None for p in backbone.parameters()) or model_state_digest(backbone) != backbone_before:
            raise RuntimeError("Frozen H1 state/gradients changed")

    for kind in ("unet", "decoder"):
        for loss_name, weight in (("bce", 0.), ("bce_tversky", AUXILIARY_WEIGHT)):
            arm = kind + "_" + loss_name
            head = create_head(kind).train()
            before_head = model_state_digest(head)
            report["parameter_counts"][kind] = sum(p.numel() for p in head.parameters())
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
            logits = head(cached, batch["physics"])
            if not torch.equal(logits, cached["logits_det"]):
                raise RuntimeError("Zero head failed exact initial H1 parity: " + arm)
            values = masked_refinement_loss(logits, batch["target"], batch["valid_mask"], aux_weight=weight)
            if not all(bool(torch.isfinite(value)) for value in values.values()):
                raise RuntimeError("Nonfinite smoke loss: " + arm)
            values["total"].backward()
            gradients = [p.grad for p in head.parameters()]
            if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
                raise RuntimeError("Missing/nonfinite smoke gradients: " + arm)
            grad_norm = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError("Nonpositive/nonfinite smoke gradient norm")
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
                raise RuntimeError("Nonfinite parameters after disposable update")
            frozen_gate()
            report["smoke"][arm] = dict(initial_h1_logits_exact=True, losses={k: float(v.detach()) for k, v in values.items()},
                gradient_norm=grad_norm, gradients_finite=True, frozen_h1_unchanged=True,
                initial_head_state_sha256=before_head, configuration=head.configuration(),
                loss=dict(aux_weight=weight, alpha_fp=.3, beta_fn=.7, reduction="batch_pooled_valid_pixels"))
            del optimizer, head, logits, values, gradients
            torch.cuda.empty_cache()
    report["status"] = "timing"
    save()

    for kind in ("unet", "decoder"):
        head = create_head(kind).eval()

        def head_forward():
            with torch.no_grad():
                head(cached, batch["physics"])

        def complete_forward():
            with torch.no_grad():
                outputs = backbone(batch["x_norm"], batch["physics"], return_decoder_features=kind == "decoder")
                head(outputs, batch["physics"])

        report["timings"][kind + "_head_forward"] = measure(head_forward, torch, device, args.batch_size)
        report["timings"][kind + "_h1_head_forward"] = measure(complete_forward, torch, device, args.batch_size)
        del head
        torch.cuda.empty_cache()
        for loss_name, weight in (("bce", 0.), ("bce_tversky", AUXILIARY_WEIGHT)):
            for full_h1 in (False, True):
                head = create_head(kind).train()
                optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)

                def update():
                    optimizer.zero_grad(set_to_none=True)
                    outputs = cached
                    if full_h1:
                        with torch.no_grad():
                            outputs = backbone(batch["x_norm"], batch["physics"], return_decoder_features=kind == "decoder")
                    logits = head(outputs, batch["physics"])
                    values = masked_refinement_loss(logits, batch["target"], batch["valid_mask"], aux_weight=weight)
                    values["total"].backward()
                    optimizer.step()

                key = kind + "_" + loss_name + ("_h1_head_update" if full_h1 else "_head_update")
                report["timings"][key] = measure(update, torch, device, args.batch_size)
                if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
                    raise RuntimeError("Timed disposable updates produced nonfinite weights")
                frozen_gate()
                del optimizer, head
                torch.cuda.empty_cache()
        save()
    frozen_gate()
    for path, digest in watched.items():
        if sha(path) != digest:
            raise RuntimeError("Benchmark mutated input or source bytes: " + path)
    report["speed_ratios_unet_over_decoder"] = {}
    for suffix in ("head_forward", "h1_head_forward", "bce_head_update", "bce_h1_head_update",
                   "bce_tversky_head_update", "bce_tversky_h1_head_update"):
        old, new = report["timings"]["unet_" + suffix], report["timings"]["decoder_" + suffix]
        report["speed_ratios_unet_over_decoder"][suffix] = old["mean_wall_ms"] / new["mean_wall_ms"]
    report.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
        frozen_h1_after_state_sha256=model_state_digest(backbone), frozen_h1_unchanged=True,
        input_and_source_sha256=watched, all_input_source_bytes_unchanged=True,
        interpretation="Repeated first Tr batch only; no epoch throughput or quality claim. Parameter ratio is not speedup.",
        timing_scope="Forward includes actual validation checks; update includes masked loss/backward/AdamW. Excludes loading, checkpointing, gradient diagnostics, and metrics.",
        memory_scope="Absolute PyTorch peaks include H1, real batch and cached frozen features; incremental peaks exclude those live baseline allocations.")
    save()


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema="fwt_literature_refiner_real_batch_benchmark_v1", status="starting",
                  created_utc=datetime.now(timezone.utc).isoformat(), batch_size=args.batch_size,
                  warmup=WARMUP, repetitions=REPEATS, checkpoint_writes=0, formal_training=False)
    # Exclusive creation is the ownership claim; later replacement updates only our file.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)

    def save():
        temp = args.output.with_name(args.output.name + ".tmp-" + str(os.getpid()))
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temp, args.output)
        print(json.dumps(dict(event="benchmark_status", status=report["status"], output=str(args.output))), flush=True)

    try:
        execute(args, report, save)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
