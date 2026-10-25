"""Validation operating-point diagnostics; no fitting or checkpoint selection.

These reports supplement the historical fixed-0.5 metrics. A threshold selected
on the same validation predictions is an optimistic development diagnostic,
not an independent performance estimate or a calibration artifact.
"""
from collections import defaultdict
import math

import numpy as np

from .protocol import _validated


def _summary(probability, positive):
    count = len(probability)
    mean_probability = float(probability.mean())
    prevalence = float(positive.mean())
    return dict(n_valid_pixel_days=count, n_positive_pixel_days=int(positive.sum()),
                prevalence=prevalence, mean_probability=mean_probability,
                mean_probability_minus_prevalence=mean_probability - prevalence,
                positive_mean_probability=float(probability[positive].mean()) if positive.any() else None,
                negative_mean_probability=float(probability[~positive].mean()) if (~positive).any() else None)


def _confusion(probability, positive, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Operating threshold must be finite and within [0,1]')
    alarm = probability >= threshold
    tp = int((alarm & positive).sum())
    fp = int((alarm & ~positive).sum())
    fn = int((~alarm & positive).sum())
    tn = int((~alarm & ~positive).sum())
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                alarm_pixel_days=tp + fp, alarm_fraction=float(alarm.mean()))


def operating_point_report(records, candidate_threshold, sampling=None):
    """Describe fixed/candidate decisions without recomputing ranking metrics.

    The caller supplies the existing validation-F1 candidate threshold. This
    function never selects a threshold, checkpoint, sampler or temperature.
    Precision/recall are null when their respective denominator is zero;
    empty-union F1/IoU retain the existing protocol's value of one.
    """
    groups = defaultdict(lambda: [[], []])
    for record in _validated(records):
        mask = np.asarray(record['valid_mask'], dtype=bool)
        groups[record['event_id']][0].append(np.asarray(record['probability'], dtype=np.float64)[mask])
        groups[record['event_id']][1].append(np.asarray(record['target'])[mask] > .5)
    pooled_probability, pooled_positive, events = [], [], []
    empty = 0
    for event_id, (probabilities, positives) in groups.items():
        p, y = np.concatenate(probabilities), np.concatenate(positives)
        if not len(p):
            empty += 1
            events.append(dict(event_id=event_id, n_valid_pixel_days=0, n_positive_pixel_days=0,
                               excluded_from_event_equal=True))
            continue
        pooled_probability.append(p)
        pooled_positive.append(y)
        events.append(dict(event_id=event_id, excluded_from_event_equal=False, **_summary(p, y)))
    if not pooled_probability:
        raise ValueError('No valid supervision for operating-point diagnostics')
    p, y = np.concatenate(pooled_probability), np.concatenate(pooled_positive)
    observed = [event for event in events if not event['excluded_from_event_equal']]
    event_equal = {key: float(np.mean([event[key] for event in observed]))
                   for key in ('prevalence', 'mean_probability', 'mean_probability_minus_prevalence')}
    return dict(schema='fwt_validation_operating_points_v1',
                purpose='validation_selection_optimistic', independent_evaluation=False,
                decision_aggregation='pooled_valid_pixel_days',
                threshold_rule='pooled_validation_f1_max_tie_larger_threshold',
                checkpoint_selection_uses='pooled_validation_AP_only',
                sampling=dict(sampling) if sampling is not None else None,
                undefined_precision_or_recall=None,
                fixed_0_5=_confusion(p, y, .5),
                validation_f1_candidate=_confusion(p, y, candidate_threshold),
                pooled=_summary(p, y), event_equal=event_equal,
                n_events=len(groups), n_events_with_supervision=len(observed),
                n_events_without_supervision=empty, events=events)
