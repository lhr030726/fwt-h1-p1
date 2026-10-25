"""Independent probability calibration and whole-event expected FNR control.

CRC: Angelopoulos et al., arXiv:2208.02814. This is the RCDA branch's
proposal, not the WSTS content-three v3 protocol.
"""
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, precision_recall_curve


def _validated(records):
    records = list(records)
    if not records:
        raise ValueError('At least one prediction record is required')
    for item in records:
        p = np.asarray(item['probability'], dtype=np.float64)
        y = np.asarray(item['target'])
        valid = np.asarray(item['valid_mask'], dtype=bool)
        domain = np.asarray(item['domain_mask'], dtype=bool)
        if not item.get('event_id') or not (p.shape == y.shape == valid.shape == domain.shape):
            raise ValueError('Event identity and matching prediction/mask shapes are required')
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('Probability must be finite and within [0,1]')
        if not np.isfinite(y).all() or np.any((y != 0) & (y != 1)):
            raise ValueError('Targets must be binary and finite')
        if np.any(valid & ~domain) or np.any((y > 0) & ~(valid & domain)):
            raise ValueError('Supervision and positive targets must lie inside the prediction domain')
    return records


def calibrate(probability, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    return expit(logit(np.clip(probability, 1e-6, 1 - 1e-6)) / temperature)


def _event_pixels(records):
    grouped = defaultdict(lambda: [[], []])
    for item in _validated(records):
        mask = np.asarray(item['valid_mask'], dtype=bool)
        grouped[item['event_id']][0].append(np.asarray(item['probability'])[mask])
        grouped[item['event_id']][1].append(np.asarray(item['target'])[mask])
    return {event: (np.concatenate(p), np.concatenate(y)) for event, (p, y) in grouped.items()}


def _event_logits(records):
    grouped = defaultdict(lambda: [[], []])
    for item in _validated(records):
        mask = np.asarray(item['valid_mask'], dtype=bool)
        z = np.asarray(item.get('calibration_logit', logit(np.clip(item['probability'], 1e-6, 1-1e-6))))
        if z.shape != mask.shape or not np.isfinite(z).all():
            raise ValueError('Calibration logits must be finite and match probability shape')
        grouped[item['event_id']][0].append(z[mask])
        grouped[item['event_id']][1].append(np.asarray(item['target'])[mask])
    return {eid: (np.concatenate(z), np.concatenate(y)) for eid, (z, y) in grouped.items()}


def fit_temperature(records):
    groups = [(z, y) for z, y in _event_logits(records).values() if len(z)]
    if not groups:
        raise ValueError('Probability calibration has no valid supervision')

    def objective(log_temperature):
        losses = []
        for logits, y in groups:
            z = logits / np.exp(log_temperature)
            losses.append(np.mean(np.logaddexp(0, z) - y * z))
        return float(np.mean(losses))

    result = minimize_scalar(objective, bounds=(-6, 6), method='bounded')
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError('Temperature optimization failed')
    # T=1 is a valid candidate and prevents numerical optimization worsening NLL.
    value = float(result.x) if result.fun < objective(0) else 0.0
    return dict(temperature=float(np.exp(value)), n_events=len(groups),
                objective='event_equal_unweighted_pixel_day_nll',
                nll_before=objective(0), nll_after=objective(value))


def fit_risk(records, alpha):
    """Minimum feasible right-continuous set; explicit full-domain fallback.

    A whole event with no valid new burned pixels has loss zero. Events with
    an empty prediction domain are retained in this predeclared population.
    """
    if not 0 < alpha < 1:
        raise ValueError('alpha must be within (0,1)')
    records = _validated(records)
    if any(np.any(np.asarray(r['domain_mask'], bool) & ~np.asarray(r['valid_mask'], bool)) for r in records):
        raise ValueError('CRC requires observed labels across the declared candidate domain; unknown labels are not zero-new events')
    groups = _event_pixels(records)
    n = len(groups)
    scores, weights = [], []
    initial = 0.0
    for p, y in groups.values():
        positive = p[y > .5]
        if len(positive):
            initial += 1 / n
            scores.append(1 - positive)
            weights.append(np.full(len(positive), 1 / (n * len(positive))))
    candidates = np.array([0.0, 1.0])
    losses = np.array([initial, 0.0])
    if scores:
        score = np.concatenate(scores)
        weight = np.concatenate(weights)
        order = np.argsort(score, kind='stable')
        score, weight = score[order], weight[order]
        ends = np.r_[np.flatnonzero(np.diff(score)) , len(score) - 1]
        candidates = np.r_[0.0, score[ends], 1.0]
        losses = np.r_[initial, initial - np.cumsum(weight)[ends], 0.0]
        # At lambda=0, ties with exactly p=1 already belong to the set.
        losses[0] = initial - weight[score <= 0].sum()
    corrected = n / (n + 1) * np.clip(losses, 0, 1) + 1 / (n + 1)
    feasible = np.flatnonzero(corrected <= alpha + 1e-12)
    chosen = int(feasible[0]) if len(feasible) else len(candidates) - 1
    return dict(alpha=float(alpha), n_events=n, **{'lambda': float(candidates[chosen])},
                empirical_event_fnr=float(np.clip(losses[chosen], 0, 1)),
                corrected_empirical_bound=float(corrected[chosen]),
                finite_sample_condition_feasible=bool(len(feasible)),
                fallback_full_domain=not bool(len(feasible)),
                loss='whole_event_cumulative_new_pixel_fnr_zero_new_is_zero',
                population='all_predeclared_events_including_zero_new',
                guarantee='expected_risk_only_under_applicable_exchangeability_assumptions',
                tie_rule='1-probability <= lambda')


def risk_summary(records, risk_lambda):
    if not 0 <= risk_lambda <= 1:
        raise ValueError('Risk lambda must lie in [0,1]')
    grouped = defaultdict(lambda: dict(positive=0, missed=0, coverages=[], alarm_pixels=0,
                                       domain_pixels=0, supervised_pixels=0, incomplete=False))
    for item in _validated(records):
        p, y = np.asarray(item['probability']), np.asarray(item['target'])
        domain = np.asarray(item['domain_mask'], dtype=bool)
        valid = np.asarray(item['valid_mask'], dtype=bool)
        alarm = domain & ((1 - p) <= risk_lambda)
        positive = (y > .5) & valid
        group = grouped[item['event_id']]
        group['positive'] += int(positive.sum())
        group['missed'] += int((positive & ~alarm).sum())
        group['alarm_pixels'] += int(alarm.sum())
        group['domain_pixels'] += int(domain.sum())
        group['supervised_pixels'] += int(valid.sum())
        group['incomplete'] |= bool((domain & ~valid).any())
        if domain.any():
            group['coverages'].append(float(alarm.sum() / domain.sum()))
    losses, observed_losses, coverages, zeros, positive_losses = [], [], [], [], []
    empty_domain = 0
    for group in grouped.values():
        loss = group['missed'] / group['positive'] if group['positive'] else 0.0
        losses.append(loss)
        if group['supervised_pixels'] or not group['domain_pixels']:
            observed_losses.append(loss)
        coverage = float(np.mean(group['coverages'])) if group['coverages'] else None
        if coverage is None:
            empty_domain += 1
        else:
            coverages.append(coverage)
        if group['positive']:
            positive_losses.append(loss)
        elif not group['incomplete']:
            zeros.append((coverage, group['alarm_pixels'] > 0))
    zero_coverages = [c for c, _ in zeros if c is not None]
    incomplete = sum(g['incomplete'] for g in grouped.values())
    domains = sum(g['domain_pixels'] for g in grouped.values())
    supervised = sum(g['supervised_pixels'] for g in grouped.values())
    return dict(n_events=len(grouped), mean_event_fnr=float(np.mean(losses)) if not incomplete else None,
                observed_label_event_fnr=float(np.mean(observed_losses)) if observed_losses else None,
                full_event_risk_evaluable=not bool(incomplete),
                n_events_with_incomplete_labels=incomplete,
                supervision_fraction=supervised / domains if domains else None,
                positive_only_event_fnr=float(np.mean(positive_losses)) if positive_losses else None,
                event_alarm_coverage=float(np.mean(coverages)) if coverages else None,
                n_events_without_candidate_domain=empty_domain, n_zero_new_events=len(zeros),
                zero_new_event_alarm_coverage=float(np.mean(zero_coverages)) if zero_coverages else None,
                zero_new_event_false_alarm_rate=float(np.mean([a for _, a in zeros])) if zeros else None,
                alarm_pixel_days=sum(g['alarm_pixels'] for g in grouped.values()))


def probability_metrics(records, threshold=.5, bins=10):
    records = _validated(records)
    groups = _event_pixels(records)
    event_logits = _event_logits(records)
    p, y = [], []
    nll, brier = [], []
    positive_ap, empty = [], 0
    for event_id, (scores, labels) in groups.items():
        if not len(scores):
            empty += 1
            continue
        p.append(scores)
        y.append(labels)
        z = event_logits[event_id][0]
        nll.append(np.mean(np.logaddexp(0, z) - labels * z))
        brier.append(np.mean((scores - labels) ** 2))
        if labels.sum():
            positive_ap.append(average_precision_score(labels, scores))
    if not p:
        raise ValueError('No valid supervision for probability metrics')
    p, y = np.concatenate(p), np.concatenate(y)
    predicted = p >= threshold
    positive = y > .5
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    reliability, ece = [], 0.0
    bin_id = np.minimum((p * bins).astype(int), bins - 1)
    for index in range(bins):
        selected = bin_id == index
        if selected.any():
            confidence, frequency = float(p[selected].mean()), float(y[selected].mean())
            count = int(selected.sum())
            ece += count / len(p) * abs(confidence - frequency)
            reliability.append(dict(bin=index, count=count, probability=confidence, frequency=frequency))
    return dict(pooled_ap=float(average_precision_score(y, p)) if positive.any() else None,
                positive_event_ap=float(np.mean(positive_ap)) if positive_ap else None,
                n_events=len(groups), n_positive_events=len(positive_ap), n_events_without_supervision=empty,
                n_valid_pixel_days=len(p), nll=float(np.mean(nll)), brier=float(np.mean(brier)),
                ece_pooled=float(ece), reliability=reliability, threshold=float(threshold),
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0)


def select_raw_threshold(records):
    groups = _event_pixels(records)
    p = np.concatenate([p for p, _ in groups.values()])
    y = np.concatenate([y for _, y in groups.values()])
    if not len(p) or not y.sum():
        raise ValueError('Validation must contain valid positive targets for AP/threshold selection')
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])
