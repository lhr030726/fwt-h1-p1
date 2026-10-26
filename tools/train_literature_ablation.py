#!/usr/bin/env python3
"""Fixed-budget Tr/V-only paired loss/head ablation; dry-run unless --execute.

Pilot: seed 4, old/compact heads crossed with BCE/BCE+.05*Tversky.
Confirm: seeds 5/6/7, the explicitly frozen candidate versus old BCE.
Epoch transactions preserve all paired heads together, including optimizer/RNG,
without accumulating checkpoint versions or changing any historical artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import copy, deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import train_refiner_multiseed as old

SCHEMA = "fwt_literature_ablation_state_v1"
TRANSACTION_SCHEMA = "fwt_literature_ablation_transaction_v1"
EXPECTED_AP = 0.23928948793196242
ARMS = ("old_bce", "old_tversky", "compact_bce", "compact_tversky")
CANDIDATES = ARMS[1:]
EPOCHS, BATCH_SIZE, VALIDATION_BATCH_SIZE = 10, 16, 8
BASE_RECIPE = dict(epochs=10, batch_size=16, validation_batch_size=8,
    learning_rate=1e-4, weight_decay=.01, optimizer="AdamW", precision="float32",
    cpu_threads=2, sample_cache_gib=16, augmentation="none", alpha_fp=.3, beta_fn=.7,
    tversky_weight=.05, tversky_reduction="batch_pooled_valid_pixels",
    bce_reduction="mean_all_valid_pixels", no_positive_batch="zero_auxiliary_keep_bce",
    checkpoint_selection="maximum_full_V_pooled_AP_earliest_tie",
    initialization="per_arm_seed_then_construct_load_H1_then_construct_zero_output_head",
    batch_order="one_seeded_CPU_generator_ten_complete_Tr_permutations_shared_by_arms",
    backbone="fixed_original_H1_no_grad_eval", training_role="residual_train",
    validation_role="validation", train_samples=811, validation_samples=568,
    epoch_metrics="one_pooled_AP_sort_raw_logit_event_equal_NLL_Brier",
    selected_threshold_rules=["V_max_F1_tie_higher", "V_recall_at_least_0.5_0.7_0.8_highest_threshold"])

read_json, sha, canonical_sha = old.read_json, old.sha, old.canonical_sha
tensor_cpu_tree = old.tensor_cpu_tree


def stage_design(stage, candidate=None):
    if stage == "pilot" and candidate is None:
        return (4,), ARMS
    if stage == "confirm" and candidate in CANDIDATES:
        return (5, 6, 7), ("old_bce", candidate)
    raise ValueError("pilot has no candidate; confirm requires one of the three nonbaseline candidates")


def seeded_order(seed, train_ids):
    import torch
    if type(seed) is not int or seed not in (4, 5, 6, 7):
        raise ValueError("Only predeclared seeds 4/5/6/7 are allowed")
    if len(train_ids) != 811 or len(set(train_ids)) != 811:
        raise ValueError("Require 811 unique original Tr identities")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    orders = []
    for _ in range(EPOCHS):
        ordered = [train_ids[i] for i in torch.randperm(811, generator=generator).tolist()]
        orders.append([ordered[i:i + BATCH_SIZE] for i in range(0, 811, BATCH_SIZE)])
    old.validate_order(orders, train_ids)
    return orders


def source_files():
    return old.source_files(ROOT) | {
        "tools/train_refiner_multiseed.py": sha(ROOT / "tools/train_refiner_multiseed.py"),
        "tools/train_literature_ablation.py": sha(__file__)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference-plan", "backbone", "manifest", "normalizer", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--selection-decision", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    return args


def make_plan(args):
    seeds, arms = stage_design(args.stage, args.candidate)
    if (args.stage == "confirm") != (args.selection_decision is not None):
        raise ValueError("Confirm requires --selection-decision; pilot must not supply it")
    reference = read_json(args.reference_plan)
    receipt_path = args.reference_plan.parent / "receipt.json"
    receipt = read_json(receipt_path)
    if (reference.get("schema") != "fwt_logit_refiner_multiseed_plan_v1"
            or receipt.get("schema") != "fwt_logit_refiner_multiseed_completed_v1"
            or receipt.get("status") != "completed" or set(receipt.get("seeds", {})) != {"1", "2", "3"}
            or any(r.get("status") != "completed" or r.get("completed_epochs") != 10
                   or r.get("total_updates") != 510 for r in receipt["seeds"].values())):
        raise ValueError("Require the completed three-seed BCE reference plan and receipt")
    if abs(reference["expected_h1_validation_ap"] - EXPECTED_AP) > 1e-12:
        raise ValueError("Reference H1 baseline is not the original fixed result")
    current = source_files()
    allowed_changed = {"fwt_v2/models.py", "fwt_v2/rcda.py"}
    for name, digest in reference["source_files"].items():
        if name not in allowed_changed and current.get(name) != digest:
            raise ValueError("An unchanged reference source differs: " + name)
    if set(current) - set(reference["source_files"]) != {
            "fwt_v2/literature_refiner.py", "tools/train_literature_ablation.py"}:
        raise ValueError("Unexpected source additions outside the declared ablation")
    files = dict(reference["input_files_sha256"])
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError("Reference input bytes changed: " + path)
    for key in ("backbone", "manifest", "normalizer"):
        path = getattr(args, key)
        if str(path) != reference["input_paths"][key] or sha(path) != files[str(path)]:
            raise ValueError("Explicit argument differs from original reference: " + key)
    raw_manifest = read_json(args.manifest)
    if {r["role"] for r in raw_manifest["samples"]} != {
            "backbone_train", "residual_train", "validation"}:
        raise ValueError("Only development Tb/Tr/V manifest is permitted; held-out roles forbidden")
    for role, key, count in (("residual_train", "training_sample_ids", 811),
                             ("validation", "validation_sample_ids", 568)):
        actual = [r["sample_id"] for r in raw_manifest["samples"] if r["role"] == role]
        if actual != reference[key] or len(actual) != count or len(set(actual)) != count:
            raise ValueError("Original Tr/V sample order or identities changed")
    bundle_path = args.manifest.parent / "bundle.json"
    bundle = read_json(bundle_path)
    if bundle.get("data_scope") != "development_only" or bundle.get("heldout_arrays_included") is not False:
        raise ValueError("Require the original development-only bundle")
    files[str(args.reference_plan)] = sha(args.reference_plan)
    files[str(receipt_path)] = sha(receipt_path)
    input_binding_path = args.reference_plan.parent / "input-binding.json"
    if sha(input_binding_path) != receipt["input_binding_sha256"]:
        raise ValueError("Completed reference input binding changed")
    files[str(input_binding_path)] = sha(input_binding_path)
    if args.stage == "confirm":
        files.update(verify_selection_decision(args.selection_decision, args.candidate,
            current, files, str(args.reference_plan)))
    orders = {str(seed): seeded_order(seed, reference["training_sample_ids"]) for seed in seeds}
    plan = dict(schema="fwt_literature_ablation_plan_v1", stage=args.stage,
        candidate=args.candidate, seeds=list(seeds), arms=list(arms), recipe=BASE_RECIPE,
        source_files=current, reference_source_files=reference["source_files"],
        allowed_reference_source_changes=sorted(allowed_changed), input_files_sha256=files,
        reference_plan=str(args.reference_plan), reference_plan_sha256=sha(args.reference_plan),
        selection_decision=str(args.selection_decision) if args.selection_decision else None,
        bound_data_metadata_paths=reference["bound_data_metadata_paths"],
        batch_orders=orders, batch_order_sha256={s: canonical_sha(o) for s, o in orders.items()},
        expected_h1_validation_ap=EXPECTED_AP, h1_validation_ap_tolerance=1e-6,
        output=str(args.output), resume_run=str(args.resume_run) if args.resume_run else None,
        dry_run=not args.execute,
        limits=["Tr/V development only; no final or calibration arrays are opened.",
                "All heads are newly initialized; H1 is fixed. Not an original-head continuation.",
                "Confirm candidate must be selected using pilot V only before this plan is frozen.",
                "Previously evaluated final set is already exposed and is not fresh independent confirmation.",
                "Tversky is an overlap auxiliary, not an AP objective."])
    for key in ("training_sample_ids", "validation_sample_ids", "diffusion_config", "data_config",
                "model_config", "manifest_sha256", "normalizer_sha256"):
        plan[key] = reference[key]
    if args.resume_run:
        previous = read_json(args.resume_run / "plan.json")
        ignore = {"output", "resume_run", "dry_run"}
        if canonical_sha({k: v for k, v in plan.items() if k not in ignore}) != canonical_sha(
                {k: v for k, v in previous.items() if k not in ignore}):
            raise ValueError("Resume plan differs in source/data/stage/candidate/seeds/recipe")
    return plan


def verify_selection_decision(path, candidate, sources, reference_files, reference_plan):
    """Require a completed, same-contract pilot and a concrete frozen decision."""
    decision = read_json(path)
    if (decision.get("schema") != "fwt_literature_selection_decision_v1"
            or decision.get("candidate") != candidate or decision.get("selection_frozen") is not True):
        raise ValueError("Confirm selection decision schema/candidate/freeze differs")
    files = {str(path): sha(path)}
    for key in ("pilot_plan", "pilot_receipt"):
        bound = Path(decision[key]).resolve()
        if sha(bound) != decision[key + "_sha256"]:
            raise ValueError("Confirm pilot evidence digest differs: " + key)
        files[str(bound)] = sha(bound)
    pilot = read_json(decision["pilot_plan"])
    receipt = read_json(decision["pilot_receipt"])
    if (pilot.get("schema") != "fwt_literature_ablation_plan_v1" or pilot.get("stage") != "pilot"
            or pilot.get("seeds") != [4] or pilot.get("arms") != list(ARMS)
            or pilot.get("source_files") != sources or pilot.get("input_files_sha256") != reference_files
            or pilot.get("reference_plan") != reference_plan
            or receipt.get("schema") != "fwt_literature_ablation_completed_v1"
            or receipt.get("status") != "completed" or receipt.get("stage") != "pilot"
            or receipt.get("plan_sha256") != decision["pilot_plan_sha256"]
            or set(receipt.get("seeds", {}).get("4", {})) != set(ARMS)
            or any(r.get("status") != "completed" or r.get("total_updates") != 510
                   or r.get("completed_epochs") != 10 for r in receipt["seeds"]["4"].values())):
        raise ValueError("Confirm requires a completed pilot with identical source/reference binding")
    if Path(decision["pilot_plan"]).resolve().parent != Path(decision["pilot_receipt"]).resolve().parent:
        raise ValueError("Pilot plan and receipt must belong to the same run")
    for name, digest in decision.get("evidence_files_sha256", {}).items():
        if sha(name) != digest:
            raise ValueError("Selection decision evidence bytes changed")
        files[name] = digest
    return files


def initialize_head(seed, arm, backbone_path, device, config, expected_h1_sha):
    from fwt_v2.engine import load_det, model_state_digest, seed_everything
    from fwt_v2.logit_refiner import ConditionalLogitRefiner
    from fwt_v2.literature_refiner import DecoderFeatureRefiner
    from fwt_v2.training_state import capture_rng
    if type(seed) is not int or seed not in (4, 5, 6, 7) or arm not in ARMS:
        raise ValueError("Unregistered seed or arm")
    seed_everything(seed)
    temporary, _ = load_det(backbone_path, device)
    if model_state_digest(temporary) != expected_h1_sha:
        raise ValueError("Initialization loaded a different fixed H1")
    del temporary
    head = (ConditionalLogitRefiner(config, loss="bce") if arm.startswith("old_")
            else DecoderFeatureRefiner()).to(device)
    return head, model_state_digest(head), capture_rng(device)


def validation_records(backbone, head, loader, device, compact=False):
    import torch
    from fwt_v2.engine import records_from_batch, to_device
    backbone.eval()
    if head is not None:
        head.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = (backbone(batch["x_norm"], batch["physics"], return_decoder_features=True)
                       if compact else backbone(batch["x_norm"], batch["physics"]))
            logits = outputs["logits_det"] if head is None else head(outputs, batch["physics"])
            records.extend(records_from_batch(batch, logits.sigmoid(), logits))
    return records


def records_identity(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["sample_id"].encode())
        for key in ("target", "valid_mask", "domain_mask"):
            digest.update(record[key].tobytes())
    return digest.hexdigest()


def epoch_metrics(records):
    """One exact pooled AP sort; proper scores use raw logits, event equal."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    from fwt_v2.protocol import _validated
    records = _validated(records)
    probabilities, labels, groups = [], [], defaultdict(lambda: [0., 0., 0])
    for row in records:
        valid = np.asarray(row["valid_mask"], bool)
        p = np.asarray(row["probability"])[valid]
        y = np.asarray(row["target"])[valid]
        raw = np.asarray(row["calibration_logit"], dtype=np.float64)
        if raw.shape != valid.shape or not np.isfinite(raw).all():
            raise ValueError("Raw validation logits must be finite and match masks")
        z = raw[valid]
        group = groups[row["event_id"]]
        group[0] += float(np.sum(np.logaddexp(0, z) - y * z, dtype=np.float64))
        group[1] += float(np.sum((p.astype(np.float64) - y) ** 2, dtype=np.float64))
        group[2] += len(p)
        probabilities.append(p); labels.append(y)
    p, y = np.concatenate(probabilities), np.concatenate(labels)
    if not len(p) or not (y > .5).any():
        raise ValueError("Full V must contain observed positives")
    observed = [v for v in groups.values() if v[2]]
    return dict(pooled_ap=float(average_precision_score(y, p)),
        nll=float(np.mean([v[0] / v[2] for v in observed])),
        brier=float(np.mean([v[1] / v[2] for v in observed])),
        n_valid_pixel_days=len(p), n_positive_pixel_days=int(y.sum()), n_events=len(groups),
        n_events_without_supervision=len(groups) - len(observed),
        proper_score_aggregation="event_equal_valid_pixel_days_raw_logits")


def selected_operating_points(records):
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    from fwt_v2.reporting import _confusion
    from fwt_v2.protocol import _validated
    rows = _validated(records)
    p = np.concatenate([r["probability"][r["valid_mask"].astype(bool)] for r in rows])
    y = np.concatenate([r["target"][r["valid_mask"].astype(bool)] for r in rows]) > .5
    if not y.any():
        raise ValueError("Cannot choose recall threshold without positives")
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-30)
    f1_index = int(np.flatnonzero(f1 == np.max(f1))[-1])
    result = dict(independent_evaluation=False, selected_on="V_only",
        tie_rule="p>=threshold; largest threshold on F1 tie or satisfying recall target",
        fixed_0_5=_confusion(p, y, .5),
        validation_max_f1=_confusion(p, y, float(thresholds[f1_index])))
    for target in (.5, .7, .8):
        index = int(np.flatnonzero(recall[:-1] >= target)[-1])
        result[f"validation_recall_{int(target * 100)}"] = _confusion(p, y, float(thresholds[index]))
    return result


def selected_positive_event_ap(records):
    """Extra event sorts are reserved for the frozen selected checkpoint."""
    import numpy as np
    from sklearn.metrics import average_precision_score
    from fwt_v2.protocol import _validated
    grouped = defaultdict(lambda: [[], []])
    for row in _validated(records):
        valid = np.asarray(row["valid_mask"], bool)
        grouped[row["event_id"]][0].append(np.asarray(row["probability"])[valid])
        grouped[row["event_id"]][1].append(np.asarray(row["target"])[valid])
    scores = {}
    for event, (probabilities, labels) in grouped.items():
        p, y = np.concatenate(probabilities), np.concatenate(labels)
        if (y > .5).any():
            scores[event] = float(average_precision_score(y, p))
    return dict(positive_event_ap=float(np.mean(list(scores.values()))) if scores else None,
                n_positive_events=len(scores), positive_event_ap_by_event=scores)


@contextmanager
def cuda_interval(events, key):
    """Record stream elapsed time; read once after the epoch synchronization."""
    import torch
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        events[key].append((start, end))


def cuda_elapsed_seconds(events):
    # Caller has already synchronized once; no per-batch timing reads.
    return {key: sum(start.elapsed_time(end) for start, end in pairs) / 1000
            for key, pairs in events.items()}


def serialize_state(value):
    import torch
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def deserialize_state(value):
    import torch
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


def atomic_bytes(path, value):
    from fwt_v2.artifacts import output_path
    path = output_path(path)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_transaction(seed, states, best_bytes, rng_state):
    """Build a complete paired epoch commit, then mirror best/last bytes later."""
    import torch
    completed = {s["completed_epochs"] for s in states.values()}
    if len(completed) != 1 or set(states) != set(best_bytes):
        raise ValueError("Paired states must commit the same completed epoch")
    arms = {}
    for name, state in states.items():
        best = best_bytes[name]
        state = deepcopy(state)
        state["selected_checkpoint_sha256"] = hashlib.sha256(best).hexdigest()
        last = serialize_state(state)
        # uint8 tensors, rather than Python bytes, keep torch 2.0 weights_only
        # loading safe and portable (bytes pickle via unsupported _codecs.encode).
        arms[name] = dict(last_data=torch.frombuffer(bytearray(last), dtype=torch.uint8),
                          last_sha256=hashlib.sha256(last).hexdigest(),
                          best_data=torch.frombuffer(bytearray(best), dtype=torch.uint8),
                          best_sha256=hashlib.sha256(best).hexdigest())
    return dict(schema=TRANSACTION_SCHEMA, seed=seed, completed_epochs=completed.pop(),
                arms=arms, rng_state=rng_state)


def transaction_checkpoint_bytes(entry, kind):
    import torch
    value = entry[kind + "_data"]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
        raise ValueError("Transaction payload must be a CPU uint8 tensor")
    value = value.numpy().tobytes()
    if hashlib.sha256(value).hexdigest() != entry[kind + "_sha256"]:
        raise ValueError("Transaction checkpoint bytes failed integrity")
    return value


def decode_transaction(transaction, seed, expected_arms):
    if (transaction.get("schema") != TRANSACTION_SCHEMA or transaction.get("seed") != seed
            or set(transaction.get("arms", {})) != set(expected_arms)
            or type(transaction.get("completed_epochs")) is not int
            or not 1 <= transaction["completed_epochs"] <= EPOCHS):
        raise ValueError("Transaction schema/seed/arms/progress differs")
    result = {}
    for arm, entry in transaction["arms"].items():
        last, best = (deserialize_state(transaction_checkpoint_bytes(entry, kind)) for kind in ("last", "best"))
        if (last.get("schema") != SCHEMA or best.get("schema") != SCHEMA
                or last["completed_epochs"] != transaction["completed_epochs"]
                or len(last.get("history", [])) != last["completed_epochs"]
                or last.get("epoch") != last["completed_epochs"] - 1
                or last["binding"].get("seed") != seed or last["binding"].get("arm") != arm
                or last["selected_checkpoint_sha256"] != entry["best_sha256"]
                or last["binding"] != best["binding"]
                or last["selected_epoch"] != best["epoch"]
                or best["completed_epochs"] != best["epoch"] + 1
                or best["updates"] != best["completed_epochs"] * 51
                or best["history"] != last["history"][:best["completed_epochs"]]
                or best["validation_ap"] != last["history"][best["epoch"]]["validation"]["pooled_ap"]
                or best["epoch"] != max(range(len(last["history"])),
                    key=lambda i: last["history"][i]["validation"]["pooled_ap"])):
            raise ValueError("Transaction best/last selection or progress mismatch")
        result[arm] = (last, best)
    return result


def restore_state(state, model, optimizer, binding, device):
    from fwt_v2.engine import model_state_digest
    from fwt_v2.training_state import restore_training_parameters, validate_rng, restore_rng
    if state.get("schema") != SCHEMA or state.get("binding") != binding:
        raise ValueError("Checkpoint schema/binding differs")
    completed = state.get("completed_epochs")
    if (type(completed) is not int or not 1 <= completed <= EPOCHS
            or state.get("updates") != completed * 51 or len(state.get("history", [])) != completed
            or [r["epoch"] for r in state["history"]] != list(range(completed))):
        raise ValueError("Checkpoint progress/history mismatch")
    validate_rng(state["rng_state"], device)
    # Existing helper rigorously validates every tensor, moment, setting and hash.
    checked = dict(state, selected=dict(state_dict=state["model_state_dict"]))
    restore_training_parameters(checked, model, optimizer)
    if model_state_digest(model) != state["model_state_sha256"]:
        raise ValueError("Checkpoint model state digest mismatch")
    restore_rng(state["rng_state"], device)


def publish_transaction(seed_dir, transaction):
    from fwt_v2.artifacts import atomic_checkpoint
    # Authority publishes first. An interrupted mirror write is recoverable from
    # the transaction; the previous full epoch stays intact until this replace.
    atomic_checkpoint(seed_dir / "transaction.pth", transaction)
    for name, entry in transaction["arms"].items():
        atomic_bytes(seed_dir / name / "best.pth", transaction_checkpoint_bytes(entry, "best"))
        atomic_bytes(seed_dir / name / "last.pth", transaction_checkpoint_bytes(entry, "last"))


def restore_seed_transaction(previous, seed_dir, seed, heads, optimizers, bindings, device):
    """Restore a paired full epoch, including a seed that is already complete."""
    import torch
    from fwt_v2.training_state import restore_rng, validate_rng
    transaction = torch.load(previous, map_location="cpu", weights_only=True)
    decoded = decode_transaction(transaction, seed, heads)
    validate_rng(transaction["rng_state"], device)
    histories, best_bytes, best_aps, selected_epochs = {}, {}, {}, {}
    for arm, (last, best) in decoded.items():
        # Validate the best optimizer too: best.pth is a full recoverable state.
        restore_state(best, heads[arm], optimizers[arm], bindings[arm], device)
        restore_state(last, heads[arm], optimizers[arm], bindings[arm], device)
        histories[arm], best_aps[arm], selected_epochs[arm] = last["history"], best["validation_ap"], best["epoch"]
        best_bytes[arm] = transaction_checkpoint_bytes(transaction["arms"][arm], "best")
    restore_rng(transaction["rng_state"], device)
    publish_transaction(seed_dir, transaction)
    return transaction["completed_epochs"], histories, best_bytes, best_aps, selected_epochs


def write_status(args, phase, **details):
    from fwt_v2.artifacts import atomic_json
    value = dict(schema="fwt_literature_ablation_status_v1", phase=phase, stage=args.stage,
        updated_at=time.time(), started_at=args._run_started_at,
        resources=old.resource_snapshot(), **details)
    atomic_json(args.output / "status.json", value)
    with (args.output / "resource-history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def execute(args, plan):
    import torch
    from torch.utils.data import DataLoader
    from fwt_v2.artifacts import atomic_json, atomic_checkpoint, output_path
    from fwt_v2.data import DataConfig, NormalizationStats, RCDADataset
    from fwt_v2.diffusion import DiffusionConfig
    from fwt_v2.engine import audited_manifest, load_det, model_state_digest, seed_everything, to_device
    from fwt_v2.gradient_diagnostics import gradients_finite, gradient_l2_norm
    from fwt_v2.literature_refiner import masked_refinement_loss
    from fwt_v2.manifest import manifest_digest, samples_for_role
    from fwt_v2.sample_cache import SharedTensorCache, CachedDataset
    from fwt_v2.training_state import capture_rng, restore_rng, sample_file_identity, optimizer_state_integrity

    output_path(args.output / "plan.json")
    if args.output.exists():
        raise FileExistsError("Output must be a new directory, also for resume")
    old.assert_gpu_idle()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA GPU required")
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    args.output.mkdir(parents=True, exist_ok=False)
    args._output_created, args._run_started_at = True, time.time()
    atomic_json(args.output / "plan.json", plan)
    write_status(args, "verify_data", step="original_Tr_V_array_hashes")
    device = torch.device("cuda:0")
    os.chdir(args.manifest.parent)
    manifest = audited_manifest(args.manifest)
    if manifest_digest(manifest) != plan["manifest_sha256"] or sha(args.normalizer) != plan["normalizer_sha256"]:
        raise ValueError("Manifest/normalizer content identity changed")
    train_rows = samples_for_role(manifest, "residual_train")
    val_rows = samples_for_role(manifest, "validation")
    for rows, key in ((train_rows, "training_sample_ids"), (val_rows, "validation_sample_ids")):
        if [r["sample_id"] for r in rows] != plan[key]:
            raise ValueError("Tr/V original identities changed")
    samples = train_rows + val_rows
    sample_identity = sample_file_identity(samples)
    array_hashes = {str(Path(row[k]).resolve()): sha(row[k]) for row in samples for k in ("input_path", "label_path")}
    old.verify_selected_array_index(read_json(plan["bound_data_metadata_paths"]["arrays_index"]),
                                   samples, array_hashes, args.manifest.parent)
    write_status(args, "verify_data", step="validate_hourly_wind_once")
    dataset = RCDADataset(manifest, "residual_train", NormalizationStats.load(args.normalizer), DataConfig(**plan["data_config"]))
    val_dataset = copy(dataset)
    val_dataset.role, val_dataset.samples = "validation", val_rows
    cache = SharedTensorCache(16 * 1024**3)
    training, validation = CachedDataset(dataset, cache), CachedDataset(val_dataset, cache)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
    id_index = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    seed_everything(0)
    backbone, bundle = load_det(args.backbone, device)
    if (not bundle.get("selection_frozen") or bundle["family"] != "det"
            or any(canonical_sha(bundle[k]) != canonical_sha(plan[k]) for k in
                   ("data_config", "model_config", "manifest_sha256", "normalizer_sha256"))):
        raise ValueError("Fixed H1 metadata differs from reference")
    backbone.requires_grad_(False).eval()
    h1_sha = model_state_digest(backbone)
    write_status(args, "h1_baseline", step="full_original_V")
    baseline = validation_records(backbone, None, val_loader, device)
    baseline_metrics, domain_sha = epoch_metrics(baseline), records_identity(baseline)
    if ([r["sample_id"] for r in baseline] != plan["validation_sample_ids"]
            or abs(baseline_metrics["pooled_ap"] - EXPECTED_AP) > 1e-6):
        raise RuntimeError("Original H1 full-V AP reproduction failed")
    del baseline
    atomic_json(args.output / "baseline.json", dict(validation=baseline_metrics,
        backbone_state_sha256=h1_sha, target_and_masks_sha256=domain_sha))
    common = dict(recipe=plan["recipe"], stage=plan["stage"], candidate=plan["candidate"],
        source_files=plan["source_files"], input_files_sha256=plan["input_files_sha256"],
        array_files_sha256=array_hashes, array_stat_identity=sample_identity,
        training_sample_ids=plan["training_sample_ids"], validation_sample_ids=plan["validation_sample_ids"],
        target_and_masks_sha256=domain_sha, frozen_backbone_state_sha256=h1_sha,
        runtime=dict(python=sys.version, torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                     cpu_threads=2, interop_threads=2, matmul_tf32=False, cudnn_tf32=True))
    atomic_json(args.output / "input-binding.json", common)

    def verify_unchanged():
        if source_files() != plan["source_files"]:
            raise RuntimeError("Ablation source changed during execution")
        if any(sha(p) != h for p, h in plan["input_files_sha256"].items()):
            raise RuntimeError("Reference inputs changed during execution")
        if sample_file_identity(samples) != sample_identity:
            raise RuntimeError("Tr/V array identity changed")
        if model_state_digest(backbone) != h1_sha or any(p.requires_grad or p.grad is not None for p in backbone.parameters()):
            raise RuntimeError("Frozen H1 changed or acquired gradients")

    config = DiffusionConfig(**dict(plan["diffusion_config"], output_initialization="zero", prediction_type="clean_probability"))
    results, seed_initials = {}, {}
    for seed in plan["seeds"]:
        seed_dir = args.output / f"seed-{seed:03d}"
        seed_dir.mkdir()
        heads, optimizers, bindings, initial_hashes = {}, {}, {}, {}
        histories, best_bytes, best_aps, selected_epochs = {}, {}, {}, {}
        for arm in plan["arms"]:
            arm_dir = seed_dir / arm
            arm_dir.mkdir()
            head, initial_sha, initial_rng = initialize_head(seed, arm, args.backbone, device, config, h1_sha)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=.01)
            binding = dict(common, seed=seed, arm=arm, head_configuration=head.configuration(),
                head_parameters=sum(p.numel() for p in head.parameters()),
                trainable_head_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                auxiliary_weight=.05 if arm.endswith("tversky") else 0.,
                initial_head_state_sha256=initial_sha, batch_order_sha256=plan["batch_order_sha256"][str(seed)],
                optimizer_defaults=deepcopy(optimizer.defaults))
            binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
            heads[arm], optimizers[arm], bindings[arm] = head, optimizer, binding
            initial_hashes[arm], histories[arm], best_aps[arm] = initial_sha, [], -1.
            atomic_json(arm_dir / "configuration.json", binding)
            atomic_checkpoint(arm_dir / "initial-head.pth", dict(schema="fwt_literature_initialization_v1",
                seed=seed, arm=arm, model_state_dict=tensor_cpu_tree(head.state_dict()),
                model_state_sha256=initial_sha, rng_state=initial_rng))
        for architecture in ("old", "compact"):
            paired = [h for a, h in initial_hashes.items() if a.startswith(architecture + "_")]
            if len(set(paired)) > 1:
                raise RuntimeError("Loss-paired heads did not share their architecture initialization")
        if any(initial_hashes == earlier for earlier in seed_initials.values()):
            raise RuntimeError("Distinct seeds produced identical head initializations")
        seed_initials[str(seed)] = initial_hashes
        restore_rng(initial_rng, device)
        start_epoch = 0
        if args.resume_run:
            previous = args.resume_run / seed_dir.name / "transaction.pth"
            if previous.exists():
                start_epoch, histories, best_bytes, best_aps, selected_epochs = restore_seed_transaction(
                    previous, seed_dir, seed, heads, optimizers, bindings, device)
        for epoch in range(start_epoch, EPOCHS):
            verify_unchanged()
            write_status(args, "training", seed=seed, epoch=epoch + 1, completed_epochs=epoch, batch=0, batches=51)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for head in heads.values():
                head.train()
            batches = [[id_index[sid] for sid in batch] for batch in plan["batch_orders"][str(seed)][epoch]]
            loader = DataLoader(training, batch_sampler=batches, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            losses, norms = {a: [] for a in heads}, {a: [] for a in heads}
            gpu_intervals = defaultdict(list)
            observed = []
            for index, cpu_batch in enumerate(loader):
                if list(cpu_batch["sample_id"]) != plan["batch_orders"][str(seed)][epoch][index]:
                    raise RuntimeError("Actual batch differs from the frozen shared order")
                batch = to_device(cpu_batch, device)
                with torch.no_grad(), cuda_interval(gpu_intervals, "shared_H1"):
                    outputs = backbone(batch["x_norm"], batch["physics"], return_decoder_features=True)
                for arm, head in heads.items():
                    optimizer = optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    with cuda_interval(gpu_intervals, arm + "/forward_and_loss"):
                        logits = head(outputs, batch["physics"])
                        if epoch == 0 and index == 0 and not torch.equal(logits, outputs["logits_det"]):
                            raise RuntimeError("Fresh zero-output head did not reproduce H1 logits")
                        parts = masked_refinement_loss(logits, batch["target"], batch["valid_mask"],
                            aux_weight=bindings[arm]["auxiliary_weight"], alpha_fp=.3, beta_fn=.7)
                    loss = parts["total"]
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite or nonscalar loss")
                    with cuda_interval(gpu_intervals, arm + "/backward"):
                        loss.backward()
                    gradients = [p.grad for p in head.parameters() if p.grad is not None]
                    if not gradients or not gradients_finite(gradients):
                        raise RuntimeError("Missing/nonfinite head gradients")
                    norm = gradient_l2_norm(gradients)
                    if not math.isfinite(norm) or norm <= 0:
                        raise RuntimeError("Invalid aggregate gradient")
                    with cuda_interval(gpu_intervals, arm + "/optimizer"):
                        optimizer.step()
                    losses[arm].append({k: float(v.detach()) for k, v in parts.items()})
                    norms[arm].append(norm)
                observed.append(list(cpu_batch["sample_id"]))
                if (index + 1) % 10 == 0:
                    write_status(args, "training", seed=seed, epoch=epoch + 1, batch=index + 1,
                                 batches=51, completed_epochs=epoch, updates_per_arm=epoch * 51 + index + 1)
            if observed != plan["batch_orders"][str(seed)][epoch]:
                raise RuntimeError("Incomplete shared training epoch")
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            gpu_seconds = cuda_elapsed_seconds(gpu_intervals)
            train_peak_allocated = torch.cuda.max_memory_allocated()
            train_peak_reserved = torch.cuda.max_memory_reserved()
            states = {}
            for arm, head in heads.items():
                write_status(args, "validation", seed=seed, epoch=epoch + 1, arm=arm, completed_epochs=epoch)
                records = validation_records(backbone, head, val_loader, device, arm.startswith("compact_"))
                if ([r["sample_id"] for r in records] != plan["validation_sample_ids"]
                        or records_identity(records) != domain_sha):
                    raise RuntimeError("Full-V identities/targets/masks changed")
                metrics = epoch_metrics(records)
                del records
                ap = metrics["pooled_ap"]
                if not math.isfinite(ap):
                    raise RuntimeError("Nonfinite validation AP")
                checks = dict(updates=51, loss_finite=True, gradients_finite=True,
                    gradient_l2_min=min(norms[arm]), gradient_l2_max=max(norms[arm]),
                    shared_batch_order_sha256=canonical_sha(observed),
                    shared_training_seconds=train_seconds, shared_H1_cuda_stream_seconds=gpu_seconds["shared_H1"],
                    head_cuda_stream_seconds={phase: gpu_seconds[arm + "/" + phase]
                        for phase in ("forward_and_loss", "backward", "optimizer")},
                    timing_scope="CUDA event stream elapsed, includes launch gaps; not GPU kernel busy time",
                    gpu_peak_allocated_bytes=train_peak_allocated, gpu_peak_reserved_bytes=train_peak_reserved)
                history = histories[arm]
                history.append(dict(epoch=epoch, training_loss=sum(v["total"] for v in losses[arm]) / 51,
                    loss_components={k: sum(v[k] for v in losses[arm]) / 51 for k in ("bce", "tversky")},
                    validation=metrics, training_checks=checks))
                state = dict(schema=SCHEMA, checkpoint_kind="last_full_training_state", binding=bindings[arm],
                    completed_epochs=epoch + 1, updates=(epoch + 1) * 51, epoch=epoch,
                    model_state_dict=tensor_cpu_tree(head.state_dict()), model_state_sha256=model_state_digest(head),
                    optimizer_state_dict=tensor_cpu_tree(optimizers[arm].state_dict()),
                    rng_state=capture_rng(device), history=deepcopy(history), validation_ap=ap)
                state["optimizer_integrity"] = optimizer_state_integrity(state["optimizer_state_dict"])
                if ap > best_aps[arm]:
                    best_aps[arm], selected_epochs[arm] = ap, epoch
                    best = dict(state, checkpoint_kind="selected_best_full_training_state")
                    best_bytes[arm] = serialize_state(best)
                state.update(best_validation_ap=best_aps[arm], selected_epoch=selected_epochs[arm])
                states[arm] = state
            verify_unchanged()
            transaction = build_transaction(seed, states, best_bytes, capture_rng(device))
            publish_transaction(seed_dir, transaction)
            for arm in heads:
                atomic_json(seed_dir / arm / "history.json", histories[arm])
            write_status(args, "training", seed=seed, epoch=epoch + 1, completed_epochs=epoch + 1,
                         step="paired_epoch_committed", best_ap=best_aps, updates_per_arm=(epoch + 1) * 51,
                         shared_epoch_seconds=time.perf_counter() - started)
            print(json.dumps(dict(event="paired_epoch_completed", seed=seed, epoch=epoch + 1,
                                  best_ap=best_aps, seconds=time.perf_counter() - started)), flush=True)
        seed_receipts = {}
        for arm, head in heads.items():
            if len(histories[arm]) != EPOCHS:
                raise RuntimeError("Arm did not finish ten epochs")
            best = deserialize_state(best_bytes[arm])
            head.load_state_dict(best["model_state_dict"], strict=True)
            if model_state_digest(head) != best["model_state_sha256"]:
                raise RuntimeError("Selected model digest mismatch")
            records = validation_records(backbone, head, val_loader, device, arm.startswith("compact_"))
            if records_identity(records) != domain_sha:
                raise RuntimeError("Selected V target/mask identity changed")
            metrics = epoch_metrics(records)
            metrics.update(selected_positive_event_ap(records))
            if abs(metrics["pooled_ap"] - best["validation_ap"]) > 1e-7:
                raise RuntimeError("Selected checkpoint AP did not reproduce")
            operating = selected_operating_points(records)
            del records
            arm_dir = seed_dir / arm
            atomic_json(arm_dir / "history.json", histories[arm])
            atomic_json(arm_dir / "validation_selected.json", dict(schema="fwt_literature_selected_v1",
                seed=seed, arm=arm, completed_epochs=EPOCHS, epoch=best["epoch"], selection_frozen=True,
                independent_evaluation=False, checkpoint_sha256=sha(arm_dir / "best.pth"),
                model_state_sha256=best["model_state_sha256"], validation=metrics, operating_points=operating))
            receipt = dict(schema="fwt_literature_arm_completed_v1", status="completed", seed=seed, arm=arm,
                completed_epochs=EPOCHS, total_updates=510, best_epoch=best["epoch"],
                best_validation_ap=metrics["pooled_ap"], last_validation_ap=histories[arm][-1]["validation"]["pooled_ap"],
                binding_sha256=canonical_sha(bindings[arm]), initial_head_state_sha256=initial_hashes[arm],
                head_parameters=bindings[arm]["head_parameters"],
                files={name: sha(arm_dir / name) for name in ("best.pth", "last.pth", "initial-head.pth",
                    "configuration.json", "history.json", "validation_selected.json")})
            atomic_json(arm_dir / "receipt.json", receipt)
            seed_receipts[arm] = receipt
        verify_unchanged()
        results[str(seed)] = seed_receipts
        del heads, optimizers
        torch.cuda.empty_cache()
    atomic_json(args.output / "receipt.json", dict(schema="fwt_literature_ablation_completed_v1", status="completed",
        stage=args.stage, candidate=args.candidate, seeds=results, initial_head_state_sha256=seed_initials,
        batch_order_sha256=plan["batch_order_sha256"], baseline_ap=baseline_metrics["pooled_ap"],
        input_binding_sha256=sha(args.output / "input-binding.json"), plan_sha256=sha(args.output / "plan.json"),
        completed_at=time.time(), limitations=plan["limits"]))
    write_status(args, "completed", completed_seeds=plan["seeds"], completed_arms=plan["arms"],
                 updates_total=len(plan["seeds"]) * len(plan["arms"]) * 510)


def main(argv=None):
    args = parse_args(argv)
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps({k: v for k, v in plan.items() if k not in {
            "batch_orders", "training_sample_ids", "validation_sample_ids"}}, indent=2, allow_nan=False))
        return 0
    try:
        execute(args, plan)
    except Exception as exc:
        if getattr(args, "_output_created", False) and not (args.output / "receipt.json").exists():
            from fwt_v2.artifacts import atomic_json
            atomic_json(args.output / "failure.json", dict(status="failed", error_type=type(exc).__name__,
                message=str(exc), recorded_at=time.time()))
            write_status(args, "failed", error_type=type(exc).__name__, message=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
