"""Training-only event balancing and an optional bounded history correction.

The frozen v1 implementation is deliberately unchanged. These are candidate
interventions, not established fixes for the observed development failures.
Event support uses the complete residual-training supervision domain, including
samples without history. Validation outcomes never determine training weights.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .history_refiner import HistoryRefiner


class BoundedHistoryRefiner(HistoryRefiner):
    """Same parameters as v1, with an explicit smooth logit-difference limit."""

    schema = "fwt_frozen_parent_bounded_history_refiner_v2"

    def __init__(self, *, cap=2., **kwargs):
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ValueError("cap must be finite and positive")
        super().__init__(**kwargs)
        self.cap = float(cap)

    def configuration(self):
        return dict(super().configuration(), correction_cap=self.cap,
                    correction_transform="cap*tanh(raw_logit_difference/cap)",
                    output="detached_parent_plus_bounded_correction_where_history_valid")

    def forward(self, outputs, parent_logits, history_delta, history_valid):
        unbounded = super().forward(outputs, parent_logits, history_delta, history_valid)
        parent = parent_logits.detach()
        correction = self.cap * torch.tanh((unbounded - parent) / self.cap)
        return torch.where(history_valid, parent + correction, parent)


def _supervision(logits, target, mask):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[0] < 1
            or logits.shape[1] != 1 or not logits.is_floating_point()):
        raise ValueError("logits must be floating [B,1,H,W]")
    for name, value in (("target", target), ("mask", mask)):
        if not isinstance(value, torch.Tensor) or value.shape != logits.shape or value.device != logits.device:
            raise ValueError(f"{name} must match logits shape/device")
    if target.dtype != logits.dtype:
        raise ValueError("target must match logit dtype")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("mask must be binary")
    valid = mask.detach().bool()
    if not bool(valid.any()):
        raise ValueError("batch has no valid supervision")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    selected = target.detach()[valid]
    if not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError("valid targets must be finite and binary")
    return valid, torch.where(valid, target.detach(), torch.zeros_like(target))


def build_event_support(samples):
    """Count full Tr valid pixels, not minibatch event frequencies or history area.

    Caller supplies only the manifest's residual_train dataset. Masks are the
    unchanged supervised target masks; no new labels or input rules are formed.
    All-zero-support events are recorded but excluded from the event denominator.
    """
    events = defaultdict(lambda: dict(samples=0, valid_pixels=0, positive_pixels=0))
    identities, seen = [], set()
    for sample in samples:
        event, sid = sample.get("event_id"), sample.get("sample_id")
        if not isinstance(event, str) or not event or not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("event/sample IDs must be nonempty; sample IDs must be unique")
        if sample.get("role", "residual_train") != "residual_train":
            raise ValueError("event weights may only use residual_train")
        seen.add(sid)
        target, mask = torch.as_tensor(sample["target"]), torch.as_tensor(sample["valid_mask"])
        if target.shape != mask.shape or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("invalid training target mask")
        valid = mask.bool()
        labels = target[valid]
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("valid training targets must be binary")
        count, positives = int(valid.sum()), int(labels.sum())
        row = events[event]
        row["samples"] += 1
        row["valid_pixels"] += count
        row["positive_pixels"] += positives
        identities.append([sid, event, count, positives])
    if not identities or not any(v["valid_pixels"] for v in events.values()):
        raise ValueError("training support is empty")
    digest = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    return dict(schema="fwt_full_training_event_support_v2", role="residual_train",
                n_samples=len(identities), n_events=sum(v["valid_pixels"] > 0 for v in events.values()),
                valid_pixels=sum(v["valid_pixels"] for v in events.values()),
                events=dict(sorted(events.items())), sample_support_sha256=digest,
                history_filter_applied=False, weighting_uses_validation=False)


def mixed_event_bce(logits, target, mask, event_ids, support, event_mix=.5):
    """Mix original batch pixel BCE with unbiased event-risk estimation.

    For N uniformly shuffled samples, B current samples and E supervised events,
    event risk = N/(B*E) * sum_i(sum_valid_pixels BCE_i / full_Tr_pixels_event_i).
    Weighting batch estimates by B/N recovers the exact full-data event mean for
    fixed predictions. Do not also resample events or renormalize the weights
    within each batch. The short final batch uses its actual B.
    """
    if (isinstance(event_mix, bool) or not isinstance(event_mix, (int, float))
            or not math.isfinite(event_mix) or not 0 <= event_mix <= 1):
        raise ValueError("event_mix must be in [0,1]")
    valid, labels = _supervision(logits, target, mask)
    batch_size = logits.shape[0]
    if len(event_ids) != batch_size:
        raise ValueError("one event ID is required per sample")
    if (support.get("role") != "residual_train" or type(support.get("n_samples")) is not int
            or type(support.get("n_events")) is not int or support["n_samples"] < batch_size
            or support["n_events"] < 1):
        raise ValueError("invalid full-training event support")
    inverse_counts = []
    for event in event_ids:
        if event not in support["events"]:
            raise ValueError("event absent from training support: " + str(event))
        count = support["events"][event]["valid_pixels"]
        if type(count) is not int or count < 0:
            raise ValueError("invalid event pixel count")
        inverse_counts.append(1. / count if count else 0.)
    per_pixel = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    per_sample = torch.where(valid, per_pixel, torch.zeros_like(per_pixel)).flatten(1).sum(1)
    pixel_bce = per_sample.sum() / valid.sum()
    inverse = logits.new_tensor(inverse_counts)
    event_bce = (per_sample * inverse).sum() * (support["n_samples"] / (batch_size * support["n_events"]))
    return dict(total=(1 - event_mix) * pixel_bce + event_mix * event_bce,
                pixel_bce=pixel_bce, event_bce=event_bce)


def correction_diagnostics(records, parent_records):
    """Per-event prediction drift and proper-score changes on the exact same V.

    Reads already-collected predictions only; never changes a loss, threshold or
    checkpoint. Event support and zero-positive events remain explicitly visible.
    """
    import numpy as np
    identities = [r["sample_id"] for r in records]
    if identities != [r["sample_id"] for r in parent_records] or len(identities) != len(set(identities)):
        raise ValueError("diagnostic records differ in identity/order")
    accum = defaultdict(lambda: dict(absolute_delta=[], samples=0, valid_pixels=0,
                                     positive_pixels=0, nll=0., parent_nll=0., brier=0., parent_brier=0.))
    sample_rows = []
    for row, parent in zip(records, parent_records):
        if row["event_id"] != parent["event_id"]:
            raise ValueError("diagnostic event differs")
        raw_mask = np.asarray(row["valid_mask"])
        mask = raw_mask.astype(bool)
        target = np.asarray(row["target"])
        if (not np.isin(raw_mask, (0, 1)).all() or any(np.asarray(value).shape != mask.shape for value in
                (target, parent["target"], parent["valid_mask"], row["calibration_logit"], parent["calibration_logit"]))):
            raise ValueError("diagnostic shapes or binary mask differ")
        if not np.array_equal(mask, parent["valid_mask"]) or not np.array_equal(target[mask], np.asarray(parent["target"])[mask]):
            raise ValueError("diagnostic supervision differs")
        logits = np.asarray(row["calibration_logit"], dtype=np.float64)[mask]
        anchor = np.asarray(parent["calibration_logit"], dtype=np.float64)[mask]
        labels = target[mask].astype(np.float64)
        if not np.isin(labels, (0., 1.)).all():
            raise ValueError("valid diagnostic targets must be finite and binary")
        if not np.isfinite(logits).all() or not np.isfinite(anchor).all():
            raise ValueError("diagnostic logits must be finite")
        absolute = np.abs(logits - anchor)
        probability = np.exp(-np.logaddexp(0., -logits))
        parent_probability = np.exp(-np.logaddexp(0., -anchor))
        count = int(mask.sum())
        scores = dict(nll=float((np.logaddexp(0., logits) - labels * logits).sum()),
                      parent_nll=float((np.logaddexp(0., anchor) - labels * anchor).sum()),
                      brier=float(((probability - labels)**2).sum()),
                      parent_brier=float(((parent_probability - labels)**2).sum()))
        stats = accum[row["event_id"]]
        stats["samples"] += 1
        stats["valid_pixels"] += count
        stats["positive_pixels"] += int(labels.sum())
        stats["absolute_delta"].append(absolute)
        for key, value in scores.items():
            stats[key] += value
        sample_rows.append(dict(sample_id=row["sample_id"], event_id=row["event_id"],
                                valid_pixels=count, positive_pixels=int(labels.sum()),
                                correction_abs_mean=float(absolute.mean()) if count else None,
                                correction_abs_max=float(absolute.max()) if count else None,
                                **{k: v / count if count else None for k, v in scores.items()}))
    events = {}
    for event, stats in sorted(accum.items()):
        absolute = np.concatenate(stats.pop("absolute_delta"))
        count = stats["valid_pixels"]
        stats.update(correction_abs_mean=float(absolute.mean()) if count else None,
                     correction_abs_p95=float(np.quantile(absolute, .95)) if count else None,
                     correction_abs_max=float(absolute.max()) if count else None)
        for key in ("nll", "parent_nll", "brier", "parent_brier"):
            stats[key] = stats[key] / count if count else None
        for key in ("nll", "brier"):
            stats[key + "_change"] = stats[key] - stats["parent_" + key] if count else None
        events[event] = stats
    return dict(schema="fwt_history_correction_diagnostics_v2", scope="development_only",
                domain="unchanged_supervised_valid_pixels", events=events, samples=sample_rows,
                used_for_training_or_checkpoint_selection=False)
