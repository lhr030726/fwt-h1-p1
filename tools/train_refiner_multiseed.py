#!/usr/bin/env python3
"""Three fresh BCE refiner seeds with frozen H1; default is a read-only plan.

Only original development Tr/V arrays can be opened here. Independent evaluation
is a separate, later frozen-prediction stage. Seed zero is historical only.
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EPOCHS, BATCH_SIZE, VALIDATION_BATCH_SIZE = 10, 16, 8
SEEDS = (1, 2, 3)
SCHEMA = "fwt_logit_refiner_multiseed_state_v1"
RECIPE = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, validation_batch_size=VALIDATION_BATCH_SIZE,
              learning_rate=1e-4, weight_decay=.01, optimizer="AdamW", seeds=list(SEEDS), loss="bce",
              cpu_threads=2, precision="float32", sample_cache_gib=16,
              checkpoint_selection="maximum_full_V_pooled_AP_earliest_tie",
              training_role="residual_train", validation_role="validation",
              train_samples=811, validation_samples=568, sampling="none",
              initialization="seed_then_construct_load_fixed_h1_then_construct_zero_output_refiner",
              batch_order="independent_cpu_torch_generator_seeded_once_per_run_then_ten_randperm",
              historical_seed_zero_in_confirmation_summary=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_copy(source, destination):
    """Copy checkpoint bytes without changing their hash; publish atomically."""
    handle, temporary = tempfile.mkstemp(dir=Path(destination).parent, suffix=".tmp")
    os.close(handle)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def selected_snapshot_path(arm, last_state):
    expected_name = f"selected-epoch-{last_state['selected_epoch'] + 1:03d}.pth"
    if last_state.get("selected_checkpoint_file") != expected_name:
        raise ValueError("Last checkpoint selected-file identity differs")
    path = arm / expected_name
    if sha(path) != last_state["selected_checkpoint_sha256"]:
        raise ValueError("Resume selected checkpoint does not match last state")
    return path


def source_files(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted((root / "fwt_v2").glob("*.py"))} | {"train.py": sha(root / "train.py")}


def seeded_order(seed, train_ids):
    """Generate a frozen Tr-only batch plan without consuming global RNG."""
    import torch
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only predeclared confirmation seeds 1, 2, 3 are permitted")
    if len(train_ids) != 811 or len(set(train_ids)) != 811:
        raise ValueError("Require the original 811 unique Tr identities")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    orders = []
    for _ in range(EPOCHS):
        ordered = [train_ids[index] for index in torch.randperm(len(train_ids), generator=generator).tolist()]
        orders.append([ordered[start:start + BATCH_SIZE] for start in range(0, len(ordered), BATCH_SIZE)])
    validate_order(orders, train_ids)
    return orders


def validate_order(orders, train_ids):
    if len(orders) != EPOCHS:
        raise ValueError("Expected ten complete training permutations")
    for batches in orders:
        if len(batches) != 51 or [len(batch) for batch in batches] != [16] * 50 + [11]:
            raise ValueError("Training batch shape differs")
        if sorted(sid for batch in batches for sid in batch) != sorted(train_ids):
            raise ValueError("Training identities duplicated, missing, or foreign")
    return True


def verify_selected_array_index(index, samples, hashes, data_root):
    """Match actual Tr/V bytes to C's published index; never open Tb arrays."""
    if index.get("schema_version") != "fwt_portable_array_index_v1":
        raise ValueError("Expected C's published array byte index")
    expected = {(row["sample_id"], field): row for row in samples for field in ("input_path", "label_path")}
    found, all_keys = set(), set()
    for item in index.get("files", []):
        key = (item["sample_id"], item["field"])
        if key in all_keys:
            raise ValueError("Duplicate published array identity")
        all_keys.add(key)
        if key not in expected:
            continue
        row = expected[key]
        path = Path(row[item["field"]]).resolve()
        index_path = item.get("path", "")
        if (not index_path.startswith("data/") or item["role"] != row["role"]
                or (data_root / index_path[len("data/"):]).resolve() != path
                or hashes[str(path)] != item["sha256"] or path.stat().st_size != item["size"]):
            raise ValueError("Tr/V array bytes differ from original C index")
        found.add(key)
    if found != set(expected):
        raise ValueError("Published array index does not cover all Tr/V inputs and labels")
    return len(found)


def assert_gpu_idle():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, text=True, capture_output=True, timeout=20)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if lines:
        if not all(line.isdigit() for line in lines):
            raise RuntimeError("Cannot interpret nvidia-smi process query: " + repr(lines))
        raise RuntimeError("GPU is occupied; multiseed run not started. PIDs=" + ",".join(lines))


def rss_bytes():
    status = Path("/proc/self/status")
    if not status.exists():
        return dict(rss_bytes=None, peak_rss_bytes=None)
    rows = dict(line.split(":", 1) for line in status.read_text().splitlines() if ":" in line)
    return {key: int(rows[field].split()[0]) * 1024 for key, field in
            (("rss_bytes", "VmRSS"), ("peak_rss_bytes", "VmHWM"))}


def resource_snapshot():
    """Small observational snapshot; missing telemetry never changes training."""
    result = rss_bytes()
    for field, path in (("cgroup_memory_bytes", "/sys/fs/cgroup/memory.current"),
                        ("cgroup_memory_limit_bytes", "/sys/fs/cgroup/memory.max")):
        try:
            value = Path(path).read_text().strip()
            result[field] = int(value) if value.isdigit() else value
        except OSError:
            result[field] = None
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True, timeout=5)
        rows = [[int(value.strip()) for value in row.split(",")] for row in query.stdout.strip().splitlines()]
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError("Expected a single GPU telemetry row")
        result.update(gpu_utilization_percent=rows[0][0], gpu_memory_used_mib=rows[0][1],
                      gpu_memory_total_mib=rows[0][2])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["gpu_telemetry_error"] = type(exc).__name__
    return result


def write_status(args, phase, *, seed=None, epoch=0, updates=0, **details):
    from fwt_v2.artifacts import atomic_json
    value = dict(schema="fwt_refiner_multiseed_status_v1", phase=phase, seed=seed,
                 epoch=epoch, updates=updates, started_at=args._run_started_at,
                 updated_at=time.time(), resources=resource_snapshot(), **details)
    atomic_json(args.output / "status.json", value)
    with (args.output / "resource-history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("manifest", "normalizer", "backbone", "h1-validation", "h1-receipt",
                 "c-diff-configuration", "c-diff-diagnostics", "c-diff-receipt", "reference-source", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--resume-run", type=Path, help="Read each seed's last.pth; write to a fresh output directory")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def make_plan(args):
    configuration = read_json(args.c_diff_configuration)
    receipt = read_json(args.c_diff_receipt)
    h1_receipt, validation = read_json(args.h1_receipt), read_json(args.h1_validation)
    if receipt.get("status") != "completed" or h1_receipt.get("status") != "completed":
        raise ValueError("Both original C stages must be completed")
    if (configuration.get("seed") != 0 or configuration.get("batch_size") != 16
            or configuration.get("learning_rate") != 1e-4
            or configuration.get("requested_epochs") != 60
            or configuration.get("loss") != "clean_probability_bce"
            or configuration.get("diffusion_config", {}).get("output_initialization") != "zero"
            or configuration.get("diffusion_config", {}).get("prediction_type") != "clean_probability"):
        raise ValueError("Reference recipe is not the intended completed C configuration")
    for path, key in ((args.c_diff_configuration, "configuration.json"), (args.c_diff_diagnostics, "diagnostics.json")):
        if sha(path) != receipt["files"][key]:
            raise ValueError("C receipt hash mismatch: " + key)
    backbone_sha = sha(args.backbone)
    if (backbone_sha != configuration["backbone_sha256"]
            or backbone_sha != h1_receipt["files"]["best.pth"]
            or backbone_sha != validation["checkpoint_sha256"]
            or sha(args.h1_validation) != h1_receipt["files"]["validation_selected.json"]):
        raise ValueError("H1 reference checkpoint/report bindings disagree")
    if validation.get("independent_evaluation") or not validation.get("selection_frozen"):
        raise ValueError("Require the original frozen development-validation H1 report")
    reference = source_files(args.reference_source)
    current = source_files(ROOT)
    if any(current.get(name) != digest for name, digest in reference.items()):
        raise ValueError("Multiseed runner changed an existing original C module")
    if set(current) - set(reference) != {"fwt_v2/logit_refiner.py"}:
        raise ValueError("Only the new logit_refiner module may extend original C modules")
    reference_digest = hashlib.sha256("".join(Path(name).name + reference[name] for name in sorted(reference) if name.startswith("fwt_v2/")).encode()).hexdigest()
    if reference_digest != configuration["source_sha256"] or reference_digest != validation["source_sha256"]:
        raise ValueError("Reference source differs from the exact completed C source")
    current["tools/train_refiner_multiseed.py"] = sha(__file__)
    paths = {key: str(value) for key, value in vars(args).items()
             if isinstance(value, Path) and key not in {"output", "resume_run", "reference_source"}}
    files = {str(path): sha(path) for path in map(Path, paths.values())}
    bundle_path = args.manifest.parent / "bundle.json"
    data_bundle = read_json(bundle_path)
    if (data_bundle.get("schema_version") != "fwt_portable_data_bundle_v1"
            or data_bundle.get("data_scope") != "development_only"
            or data_bundle.get("heldout_arrays_included") is not False):
        raise ValueError("Require C's original development-only data bundle")
    bound_paths = {}
    for key in ("manifest", "normalizer", "weather", "config", "arrays_index"):
        item = data_bundle["files"][key]
        path = (args.manifest.parent / item["path"]).resolve()
        if args.manifest.parent not in path.parents:
            raise ValueError("Bound metadata path escaped the data root")
        digest = sha(path)
        if digest != item["sha256"] or digest != receipt["binding"]["files"][key]:
            raise ValueError("Original C data metadata changed: " + key)
        bound_paths[key] = str(path)
        files[str(path)] = digest
    if Path(bound_paths["manifest"]) != args.manifest or sha(args.normalizer) != files[bound_paths["normalizer"]]:
        raise ValueError("Explicit manifest/normalizer differs from the original bundle")
    if files[str(args.manifest)] != configuration["manifest_file_sha256"]:
        raise ValueError("Manifest bytes differ from original C configuration")
    files[str(bundle_path)] = sha(bundle_path)
    orders = {str(seed): seeded_order(seed, configuration["training_sample_ids"]) for seed in SEEDS}
    order_hashes = {seed: canonical_sha(value) for seed, value in orders.items()}
    if len(set(order_hashes.values())) != len(SEEDS):
        raise RuntimeError("Distinct seeds produced identical complete batch plans")
    if len(configuration["validation_sample_ids"]) != 568 or validation["validation_sample_ids"] != configuration["validation_sample_ids"]:
        raise ValueError("Expected the unchanged 568-sample V")
    expected_ap = validation["validation"]["pooled_ap"]
    if not isinstance(expected_ap, (int, float)) or not math.isfinite(expected_ap):
        raise ValueError("Reference H1 AP must be finite")
    plan = dict(schema="fwt_logit_refiner_multiseed_plan_v1", recipe=RECIPE,
                seeds=list(SEEDS), expected_h1_validation_ap=expected_ap,
                h1_validation_ap_tolerance=1e-6, reference_source=str(args.reference_source),
                reference_source_files=reference, source_files=current,
                input_files_sha256=files, input_paths=paths,
                bound_data_metadata_paths=bound_paths,
                training_sample_ids=configuration["training_sample_ids"],
                validation_sample_ids=configuration["validation_sample_ids"],
                batch_orders=orders, batch_order_sha256=order_hashes,
                diffusion_config=configuration["diffusion_config"],
                data_config=configuration["data_config"], model_config=configuration["model_config"],
                manifest_sha256=configuration["manifest_sha256"], normalizer_sha256=configuration["normalizer_sha256"],
                limits=["Independent multiseed schema, not formal fwt_checkpoint_v2.",
                        "Fresh seeds 1/2/3 vary refiner initialization and Tr order; historical seed zero is not part of confirmation summaries.",
                        "The same selected H1 is frozen in all runs; this measures conditional refiner stability, not full-model seed variability.",
                        "V alone selects earliest maximum AP and development operating points; this entry never reads held-out arrays.",
                        "Independent data evaluation must follow checkpoint and threshold freezing in a separate entry."],
                dry_run=not args.execute, output=str(args.output),
                resume_run=str(args.resume_run) if args.resume_run else None)
    if args.resume_run:
        previous = read_json(args.resume_run / "plan.json")
        ignored = {"dry_run", "output", "resume_run"}
        if ({key: value for key, value in previous.items() if key not in ignored}
                != {key: value for key, value in plan.items() if key not in ignored}):
            raise ValueError("Resume plan differs from the fixed source/data/seeds/recipe")
    return plan


def tensor_cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tensor_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_cpu_tree(item) for item in value)
    return deepcopy(value)


def initialize_head(seed, backbone_path, device, config, expected_backbone_sha):
    """Reproducible initialization convention; loading H1 consumes constructor RNG."""
    from fwt_v2.engine import load_det, model_state_digest, seed_everything
    from fwt_v2.logit_refiner import ConditionalLogitRefiner
    from fwt_v2.training_state import capture_rng
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Unexpected confirmation seed")
    seed_everything(seed)
    constructed_backbone, _ = load_det(backbone_path, device)
    if model_state_digest(constructed_backbone) != expected_backbone_sha:
        raise RuntimeError("Fresh seed loaded a different fixed H1")
    del constructed_backbone
    head = ConditionalLogitRefiner(config, loss="bce").to(device)
    return head, model_state_digest(head), capture_rng(device)


def restore_multiseed_state(path, model, optimizer, expected_binding, device):
    """Strict full-epoch restore for the identical seed/binding; never a warm start."""
    import torch
    from fwt_v2.training_state import restore_rng, validate_rng, optimizer_state_integrity
    from fwt_v2.engine import model_state_digest
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != SCHEMA or state.get("binding") != expected_binding:
        raise ValueError("Multiseed resume schema/binding differs")
    completed = state["completed_epochs"]
    if not isinstance(completed, int) or not 1 <= completed <= EPOCHS:
        raise ValueError("Invalid saved completed epochs")
    if state["updates"] != completed * 51 or len(state["history"]) != completed:
        raise ValueError("Multiseed resume progress/history mismatch")
    if ([row["epoch"] for row in state["history"]] != list(range(completed))
            or state["diagnostics"]["completed_epochs"] != completed
            or state["diagnostics"]["epochs"] != [row["training_checks"] for row in state["history"]]):
        raise ValueError("Multiseed resume history/diagnostics differ")
    if state["optimizer_integrity"] != optimizer_state_integrity(state["optimizer_state_dict"]):
        raise ValueError("Saved optimizer integrity mismatch")
    if state["optimizer_state_dict"]["param_groups"] != optimizer.state_dict()["param_groups"]:
        raise ValueError("Saved optimizer settings differ")
    current = model.state_dict()
    saved = state["model_state_dict"]
    if set(saved) != set(current) or any(not isinstance(saved[key], torch.Tensor)
            or saved[key].shape != current[key].shape or saved[key].dtype != current[key].dtype
            or not torch.isfinite(saved[key]).all() for key in current):
        raise ValueError("Saved model tensor structure/values differ")
    parameter_ids = {key for group in state["optimizer_state_dict"]["param_groups"] for key in group["params"]}
    parameter_by_id = {key: parameter
                       for saved_group, live_group in zip(state["optimizer_state_dict"]["param_groups"], optimizer.param_groups)
                       for key, parameter in zip(saved_group["params"], live_group["params"])}
    moments = state["optimizer_state_dict"]["state"]
    if not moments or not set(moments) <= parameter_ids:
        raise ValueError("Saved optimizer has no state or unknown parameters")
    for parameter_id, moment in moments.items():
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(moment):
            raise ValueError("Incomplete AdamW state")
        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in moment.values()):
            raise ValueError("Nonfinite AdamW state")
        parameter = parameter_by_id[parameter_id]
        if any(moment[key].shape != parameter.shape or moment[key].dtype != parameter.dtype
               for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if key in moment):
            raise ValueError("AdamW moment tensor structure differs")
        if moment["step"].numel() != 1:
            raise ValueError("AdamW step must be scalar")
        step = float(moment["step"])
        if step != int(step) or not 1 <= step <= state["updates"] or (moment["exp_avg_sq"] < 0).any():
            raise ValueError("Invalid AdamW step or second moment")
    validate_rng(state["rng_state"], device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if model_state_digest(model) != state["model_state_sha256"]:
        raise ValueError("Saved model integrity mismatch")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    restore_rng(state["rng_state"], device)
    return state


def execute(args, plan):
    import torch
    from torch.utils.data import DataLoader
    from fwt_v2.artifacts import atomic_checkpoint, atomic_json, output_path
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import DiffusionConfig
    from fwt_v2.engine import (audited_manifest, load_det, model_state_digest, seed_everything,
                               validation_records, prediction_diagnostics, to_device)
    from fwt_v2.gradient_diagnostics import gradients_finite, gradient_l2_norm
    from fwt_v2.logit_refiner import ConditionalLogitRefiner
    from fwt_v2.manifest import manifest_digest, samples_for_role
    from fwt_v2.protocol import probability_metrics, select_raw_threshold
    from fwt_v2.reporting import operating_point_report
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
    from fwt_v2.training_state import capture_rng, restore_rng, optimizer_state_integrity, sample_file_identity

    output_path(args.output / "plan.json")
    if args.output.exists():
        raise FileExistsError("Output must be a new directory, including for resume")
    if args.resume_run and not (args.resume_run / "plan.json").is_file():
        raise ValueError("Resume directory has no saved multiseed plan")
    assert_gpu_idle()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Multiseed runner requires exactly one visible CUDA GPU")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    args.output.mkdir(parents=True, exist_ok=False)
    args._output_created = True
    args._run_started_at = time.time()
    atomic_json(args.output / "plan.json", plan)
    write_status(args, "verify_data", step="verify_manifest_and_hash_arrays")
    print(json.dumps(dict(event="preparation", phase="verify_data", step="verify_manifest_and_hash_arrays")), flush=True)
    device = torch.device("cuda:0")
    # Keep the original relative weather path in DataConfig/normalizer identity.
    os.chdir(args.manifest.parent)
    manifest = audited_manifest(args.manifest)
    if manifest_digest(manifest) != plan["manifest_sha256"] or sha(args.normalizer) != plan["normalizer_sha256"]:
        raise ValueError("Original manifest/normalizer binding changed")
    train_rows, val_rows = samples_for_role(manifest, "residual_train"), samples_for_role(manifest, "validation")
    for rows, key in ((train_rows, "training_sample_ids"), (val_rows, "validation_sample_ids")):
        if [row["sample_id"] for row in rows] != plan[key]:
            raise ValueError("Manifest sample identities differ from original C")
    samples = train_rows + val_rows
    sample_identity = sample_file_identity(samples)
    array_hashes = {str(Path(row[key]).resolve()): sha(row[key]) for row in samples for key in ("input_path", "label_path")}
    indexed_arrays = verify_selected_array_index(read_json(plan["bound_data_metadata_paths"]["arrays_index"]),
                                                samples, array_hashes, args.manifest.parent)
    normalizer = NormalizationStats.load(args.normalizer)
    data_config = DataConfig(**plan["data_config"])
    write_status(args, "verify_data", step="validate_original_hourly_wind_once")
    print(json.dumps(dict(event="preparation", phase="verify_data", step="validate_original_hourly_wind_once")), flush=True)
    dataset = RCDADataset(manifest, "residual_train", normalizer, data_config)
    validation_dataset = copy(dataset)
    validation_dataset.role, validation_dataset.samples = "validation", val_rows
    cache = SharedTensorCache(16 * 1024**3)
    training = CachedDataset(dataset, cache)
    validation = CachedDataset(validation_dataset, cache)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
    id_to_index = {row["sample_id"]: index for index, row in enumerate(train_rows)}
    batch_indices = {key: [[[id_to_index[sid] for sid in batch] for batch in epoch] for epoch in orders]
                     for key, orders in plan["batch_orders"].items()}
    # The H1 baseline and data cache are prepared only once. Each fresh seed below
    # repeats seed -> construct/load fixed H1 -> construct head, independently.
    seed_everything(0)
    backbone, bundle = load_det(args.backbone, device)
    settings = dict(plan["diffusion_config"], output_initialization="zero", prediction_type="clean_probability")
    config = DiffusionConfig(**settings)
    if (not bundle.get("selection_frozen") or bundle["family"] != "det"
            or canonical_sha(bundle["model_config"]) != canonical_sha(plan["model_config"])
            or canonical_sha(bundle["data_config"]) != canonical_sha(plan["data_config"])
            or bundle["manifest_sha256"] != plan["manifest_sha256"] or bundle["normalizer_sha256"] != plan["normalizer_sha256"]):
        raise ValueError("Loaded H1 metadata differs from the completed C reference")
    backbone.requires_grad_(False).eval()
    frozen_sha = model_state_digest(backbone)
    sampling = dict(seed=0)  # engine's deterministic path ignores diffusion options
    write_status(args, "h1_baseline", step="reproduce_complete_original_V")
    print(json.dumps(dict(event="preparation", phase="h1_baseline", step="reproduce_complete_original_V")), flush=True)
    baseline_records = validation_records(backbone, None, val_loader, device, sampling)
    baseline_metrics = probability_metrics(baseline_records)
    baseline_predictions = prediction_diagnostics(baseline_records)
    if abs(baseline_metrics["pooled_ap"] - plan["expected_h1_validation_ap"]) > 1e-6:
        raise RuntimeError("Full-V H1 baseline failed the fixed AP reproduction gate")
    val_ids = [r["sample_id"] for r in baseline_records]
    if val_ids != plan["validation_sample_ids"]:
        raise RuntimeError("Validation order differs from original C")
    domain_hash = hashlib.sha256()
    for record in baseline_records:
        domain_hash.update(record["sample_id"].encode())
        for key in ("target", "valid_mask", "domain_mask"):
            domain_hash.update(record[key].tobytes())
    domain_sha = domain_hash.hexdigest()
    atomic_json(args.output / "baseline.json", dict(schema="fwt_refiner_multiseed_baseline_v1", validation=baseline_metrics,
        prediction_checks=baseline_predictions, expected_ap=plan["expected_h1_validation_ap"], tolerance=1e-6,
        sample_ids=val_ids, target_and_masks_sha256=domain_sha, backbone_state_sha256=frozen_sha))
    print(json.dumps(dict(event="baseline_passed", ap=baseline_metrics["pooled_ap"],
                         expected_ap=plan["expected_h1_validation_ap"], tolerance=1e-6)), flush=True)
    del baseline_records
    weather = Path(data_config.external_wind_manifest).resolve() if data_config.external_wind_variant else None
    fixed_files = dict(plan["input_files_sha256"])
    if weather:
        fixed_files[str(weather)] = sha(weather)
    common_binding = dict(recipe=RECIPE, source_files=plan["source_files"], input_files_sha256=fixed_files,
        reference_source_files=plan["reference_source_files"], array_files_sha256=array_hashes,
        original_C_array_index_match=True, selected_array_files_verified=indexed_arrays,
        array_stat_identity=sample_identity, batch_order_sha256=plan["batch_order_sha256"],
        frozen_backbone_state_sha256=frozen_sha,
        target_and_masks_sha256=domain_sha, training_sample_ids=plan["training_sample_ids"],
        validation_sample_ids=plan["validation_sample_ids"],
        runtime=dict(python=sys.version, torch=str(torch.__version__), device=torch.cuda.get_device_name(),
                     cuda_matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
                     deterministic_algorithms=False, cpu_threads=2, interop_threads=2))
    atomic_json(args.output / "input-binding.json", common_binding)

    def verify_unchanged():
        current_sources = source_files(ROOT) | {"tools/train_refiner_multiseed.py": sha(__file__)}
        if current_sources != plan["source_files"] or source_files(args.reference_source) != plan["reference_source_files"]:
            raise RuntimeError("Source changed during multiseed run")
        if any(sha(path) != digest for path, digest in fixed_files.items()):
            raise RuntimeError("Fixed input/checkpoint/evidence changed during multiseed run")
        if sample_file_identity(samples) != sample_identity:
            raise RuntimeError("Training/validation array identity changed during multiseed run")
        if model_state_digest(backbone) != frozen_sha or any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
            raise RuntimeError("Frozen H1 changed or acquired gradients")

    seed_results, initial_hashes = {}, {}
    for seed in SEEDS:
        seed_key = str(seed)
        write_status(args, "training", seed=seed, step="initialize_or_restore_seed")
        arm = args.output / f"seed-{seed:03d}"
        arm.mkdir(exist_ok=False)
        # Preserve the historical initialization convention without disturbing
        # the shared frozen H1 or sharing an initialization across fresh seeds.
        head, initial_sha, initial_rng = initialize_head(seed, args.backbone, device, config, frozen_sha)
        if initial_sha in initial_hashes.values():
            raise RuntimeError("Distinct seeds have identical initial head states")
        initial_hashes[seed_key] = initial_sha
        atomic_checkpoint(arm / "initial-head.pth", dict(schema="fwt_refiner_multiseed_initialization_v1",
            seed=seed, model_state_dict=tensor_cpu_tree(head.state_dict()),
            model_state_sha256=initial_sha, rng_state=initial_rng, configuration=head.configuration()))
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
        binding = dict(common_binding, seed=seed, loss="bce", head_configuration=head.configuration(),
                       initial_head_state_sha256=initial_sha, batch_order_sha256=plan["batch_order_sha256"][seed_key],
                       optimizer_defaults=deepcopy(optimizer.defaults))
        # JSON canonicalization makes tuple/list and primitive encodings explicit.
        binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
        atomic_json(arm / "configuration.json", binding)
        history, epoch_diagnostics, best_bundle, start_epoch = [], [], None, 0
        restore_rng(initial_rng, device)
        if args.resume_run:
            old_arm = args.resume_run / f"seed-{seed:03d}"
            if (old_arm / "last.pth").exists():
                restored = restore_multiseed_state(old_arm / "last.pth", head, optimizer, binding, device)
                start_epoch, history = restored["completed_epochs"], restored["history"]
                epoch_diagnostics = restored["diagnostics"]["epochs"]
                best_path = selected_snapshot_path(old_arm, restored)
                best_bundle = torch.load(best_path, map_location="cpu", weights_only=True)
                if best_bundle.get("binding") != binding or best_bundle.get("schema") != SCHEMA:
                    raise ValueError("Resume selected state belongs to a different seed")
                # Preserve bytes: reserializing changes zip headers and the SHA
                # referenced by the restored last checkpoint.
                atomic_copy(best_path, arm / best_path.name)
                atomic_copy(best_path, arm / "best.pth")
                atomic_copy(old_arm / "last.pth", arm / "last.pth")
        best_ap = best_bundle["validation_ap"] if best_bundle else -1.0
        for epoch in range(start_epoch, EPOCHS):
            write_status(args, "training", seed=seed, epoch=epoch + 1, updates=epoch * 51,
                         completed_epochs=epoch, step="training_batches")
            verify_unchanged()
            torch.cuda.synchronize()
            epoch_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            head.train(); backbone.eval()
            train_loader = DataLoader(training, batch_sampler=batch_indices[seed_key][epoch], num_workers=0,
                                      generator=torch.Generator().manual_seed(0))
            losses, norms, observed_batches = [], [], []
            for batch_index, cpu_batch in enumerate(train_loader):
                if list(cpu_batch["sample_id"]) != plan["batch_orders"][seed_key][epoch][batch_index]:
                    raise RuntimeError("Actual training batch differs from the frozen seed plan")
                batch = to_device(cpu_batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    outputs = backbone(batch["x_norm"], batch["physics"])
                loss = head.training_loss(batch["target"], outputs, batch["physics"], batch["valid_mask"])
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite or nonscalar training loss")
                loss.backward()
                grads = [p.grad for p in head.parameters() if p.grad is not None]
                if not grads or not gradients_finite(grads):
                    raise RuntimeError("Missing or nonfinite gradients")
                norm = gradient_l2_norm(grads)
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Nonfinite or zero aggregate gradient")
                optimizer.step()
                losses.append(float(loss.detach())); norms.append(norm)
                observed_batches.append(list(cpu_batch["sample_id"]))
                if (batch_index + 1) % 10 == 0:
                    write_status(args, "training", seed=seed, epoch=epoch + 1,
                                 updates=epoch * 51 + batch_index + 1, completed_epochs=epoch,
                                 step="training_batches", batch=batch_index + 1, batches=51)
            if observed_batches != plan["batch_orders"][seed_key][epoch] or len(losses) != 51:
                raise RuntimeError("Incomplete seeded epoch")
            train_seconds = time.perf_counter() - epoch_started
            write_status(args, "training", seed=seed, epoch=epoch + 1, updates=(epoch + 1) * 51,
                         completed_epochs=epoch, step="full_validation")
            records = validation_records(backbone, head, val_loader, device, sampling)
            predictions = prediction_diagnostics(records)
            digest = hashlib.sha256()
            for record in records:
                digest.update(record["sample_id"].encode())
                for key in ("target", "valid_mask", "domain_mask"):
                    digest.update(record[key].tobytes())
            if [r["sample_id"] for r in records] != val_ids or digest.hexdigest() != domain_sha:
                raise RuntimeError("Full V identities/labels/domain changed")
            metrics = probability_metrics(records)
            ap = metrics["pooled_ap"]
            if ap is None or not math.isfinite(ap):
                raise RuntimeError("Undefined/nonfinite pooled validation AP")
            threshold = select_raw_threshold(records)
            operating = operating_point_report(records, threshold, sampling=None)
            del records
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - epoch_started
            checks = dict(epoch=epoch, updates=51, cumulative_updates=(epoch + 1) * 51,
                loss_finite=True, gradient_finite=True, all_updates_have_nonzero_gradients=True,
                gradient_l2_min=min(norms), gradient_l2_max=max(norms), gradient_l2_mean=sum(norms)/51,
                validation_predictions=predictions, batch_order_sha256=canonical_sha(observed_batches),
                train_seconds=train_seconds, train_validation_seconds=elapsed,
                timing_scope="training plus validation/metrics; excludes epoch integrity checks and checkpoint writes",
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), **resource_snapshot())
            verify_unchanged()
            epoch_diagnostics.append(checks)
            history.append(dict(epoch=epoch, training_loss=sum(losses)/len(losses),
                validation=metrics, validation_operating_points=operating, training_checks=checks))
            diagnostics = dict(schema="fwt_logit_refiner_multiseed_diagnostics_v1", seed=seed, status="completed" if epoch + 1 == EPOCHS else "running",
                completed_epochs=epoch + 1, requested_epochs=EPOCHS, total_updates=(epoch + 1)*51,
                frozen_backbone=dict(state_sha256=frozen_sha, unchanged=True, gradients_absent=True),
                initial_head_state_sha256=initial_sha, sample_cache=cache.stats(), epochs=epoch_diagnostics)
            state = dict(schema=SCHEMA, binding=binding, completed_epochs=epoch + 1, updates=(epoch + 1)*51,
                model_state_dict=tensor_cpu_tree(head.state_dict()), model_state_sha256=model_state_digest(head),
                optimizer_state_dict=tensor_cpu_tree(optimizer.state_dict()), rng_state=capture_rng(device),
                history=deepcopy(history), diagnostics=deepcopy(diagnostics), epoch=epoch,
                validation_ap=ap, raw_threshold=threshold,
                state_scope="Full optimizer/RNG snapshot at completed_epochs; final selection is authorized only by the separate completed receipt")
            state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
            if ap > best_ap:
                best_ap = ap
                best_bundle = deepcopy(state)
                best_bundle["checkpoint_kind"] = "selected_best_full_training_state"
                version = arm / f"selected-epoch-{epoch + 1:03d}.pth"
                if version.exists():
                    raise FileExistsError("An immutable selected epoch already exists")
                atomic_checkpoint(version, best_bundle)
                atomic_copy(version, arm / "best.pth")
            selected_name = f"selected-epoch-{best_bundle['epoch'] + 1:03d}.pth"
            state.update(checkpoint_kind="last_full_training_state", best_validation_ap=best_ap,
                         selected_epoch=best_bundle["epoch"], selected_checkpoint_file=selected_name,
                         selected_checkpoint_sha256=sha(arm / selected_name))
            atomic_checkpoint(arm / "last.pth", state)
            atomic_json(arm / "history.json", history)
            atomic_json(arm / "diagnostics.json", diagnostics)
            write_status(args, "training", seed=seed, epoch=epoch + 1, updates=(epoch + 1) * 51,
                         completed_epochs=epoch + 1, step="epoch_checkpoint_saved", validation_ap=ap, best_ap=best_ap)
            print(json.dumps(dict(event="epoch_completed", seed=seed, loss_name="bce", epoch=epoch + 1,
                loss=history[-1]["training_loss"], ap=ap, best_ap=best_ap, seconds=elapsed)), flush=True)
        # Also writes complete reports when a resumed seed already had ten epochs.
        if len(history) != EPOCHS or best_bundle is None:
            raise RuntimeError("Seed did not complete its ten-epoch budget")
        verify_unchanged()
        selected_index = max(range(EPOCHS), key=lambda i: history[i]["validation"]["pooled_ap"])
        if selected_index != best_bundle["epoch"]:
            raise RuntimeError("Saved best violates earliest maximum AP selection")
        if not (arm / "history.json").exists():
            atomic_json(arm / "history.json", history)
            atomic_json(arm / "diagnostics.json", restored["diagnostics"])
        selected_report = dict(schema="fwt_logit_refiner_multiseed_selected_v1", seed=seed, independent_evaluation=False,
            selection_frozen=True, epoch=selected_index, completed_epochs=EPOCHS,
            checkpoint_sha256=sha(arm / "best.pth"), validation=history[selected_index]["validation"],
            operating_points=history[selected_index]["validation_operating_points"], binding=binding)
        atomic_json(arm / "validation_selected.json", selected_report)
        receipt = dict(schema="fwt_logit_refiner_seed_completed_v1", status="completed", seed=seed, loss="bce",
            completed_epochs=EPOCHS, total_updates=EPOCHS * 51, best_epoch=selected_index,
            best_validation_ap=best_ap, last_validation_ap=history[-1]["validation"]["pooled_ap"],
            binding_sha256=canonical_sha(binding), initial_head_state_sha256=initial_sha,
            selected_checkpoint_file=f"selected-epoch-{selected_index + 1:03d}.pth",
            files={name: sha(arm / name) for name in ("initial-head.pth", "best.pth", "last.pth", "history.json", "diagnostics.json", "configuration.json", "validation_selected.json", f"selected-epoch-{selected_index + 1:03d}.pth")})
        atomic_json(arm / "receipt.json", receipt)
        seed_results[seed_key] = receipt
        del optimizer, head
        torch.cuda.empty_cache()
    verify_unchanged()
    atomic_json(args.output / "receipt.json", dict(schema="fwt_logit_refiner_multiseed_completed_v1", status="completed",
        seeds=seed_results, initial_head_state_sha256=initial_hashes, batch_order_sha256=plan["batch_order_sha256"],
        baseline_ap=baseline_metrics["pooled_ap"], input_binding_sha256=sha(args.output / "input-binding.json"),
        completed_at=time.time(), limitations=plan["limits"]))
    write_status(args, "completed", epoch=EPOCHS, updates=EPOCHS * 51 * len(SEEDS),
                 completed_seeds=list(SEEDS), updates_scope="sum_of_three_seed_runs")


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    # Never construct datasets, initialize CUDA, or create output on the default path.
    if not args.execute:
        concise = {key: value for key, value in plan.items() if key not in {"batch_orders", "training_sample_ids", "validation_sample_ids"}}
        print(json.dumps(concise, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    try:
        execute(args, plan)
    except Exception as exc:
        if getattr(args, "_output_created", False):
            from fwt_v2.artifacts import atomic_json
            # Do not overwrite a completed receipt, including on accidental reruns.
            if not (args.output / "receipt.json").exists() and not (args.output / "failure.json").exists():
                atomic_json(args.output / "failure.json", dict(schema="fwt_refiner_multiseed_failure_v1",
                    status="failed", error_type=type(exc).__name__, message=str(exc), recorded_at=time.time()))
                write_status(args, "failed", error_type=type(exc).__name__, message=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
