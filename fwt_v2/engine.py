"""Explicit training and frozen prediction; no job queue or automatic execution."""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset

from .artifacts import atomic_checkpoint, atomic_json, load_checkpoint, output_path, sha256
from .data import DataConfig, NormalizationStats, RCDADataset
from .diffusion import (ConditionalResidualDiffusion, DiffusionConfig, compute_probability,
                        DeterministicResidualHead)
from .manifest import load_manifest, manifest_digest
from .models import FWTConfig, FWTDet, RCDABaseline, validate_wind_prior_config
from .protocol import probability_metrics, select_raw_threshold


def audited_manifest(path):
    manifest = load_manifest(path)
    if {row['role'] for row in manifest['samples']} != {
            'backbone_train', 'residual_train', 'validation'}:
        raise ValueError('Portable training accepts exactly Tb, Tr and validation; no held-out arrays')
    audit = manifest.get('audit', {})
    required = dict(label_semantics='next_day_cumulative_burn', dates_verified=True,
                    roles_frozen=True)
    if any(audit.get(k) != v for k, v in required.items()) or not audit.get('exposure_review'):
        raise ValueError('Manifest needs audited label semantics, next-day dates, frozen roles and exposure_review')
    return manifest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_digest():
    files = sorted(Path(__file__).parent.glob('*.py'))
    return hashlib.sha256(''.join(p.name + sha256(p) for p in files).encode()).hexdigest()


def model_state_digest(model):
    """Hash parameters AND buffers without touching the model or random state."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}\n'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch_diagnostics(batch):
    valid = batch['valid_mask'].bool()
    return dict(sample_ids=list(batch['sample_id']),
                x_shape=list(batch['x_norm'].shape), target_shape=list(batch['target'].shape),
                domain_shape=list(batch['domain_mask'].shape),
                physics_shapes={key: list(value.shape) for key, value in batch['physics'].items()},
                input_finite=bool(torch.isfinite(batch['x_norm']).all()),
                target_finite=bool(torch.isfinite(batch['target']).all()),
                domain_pixels=int(batch['domain_mask'].sum()), valid_pixels=int(valid.sum()),
                new_pixels=int(((batch['target'] > .5) & valid).sum()))


def gradient_l2_norm(gradients):
    # Float64 reduction prevents squared finite FP32 values overflowing. No clipping,
    # parameter update, random draw or extra forward/backward pass occurs here.
    return math.hypot(*(float(torch.linalg.vector_norm(g.detach(), dtype=torch.float64))
                        for g in gradients))


def prediction_diagnostics(records):
    if not records:
        raise ValueError('Validation produced no predictions')
    probabilities = [record['probability'] for record in records]
    finite = all(np.isfinite(p).all() for p in probabilities)
    logits_finite = all(np.isfinite(record['calibration_logit']).all() for record in records)
    if not finite or not logits_finite:
        raise RuntimeError('Nonfinite validation probabilities or logits')
    valid_probabilities = np.concatenate([record['probability'][record['valid_mask'].astype(bool)].reshape(-1)
                                          for record in records])
    valid_count = len(valid_probabilities)
    # Diagnostics only: no clipping, selection change or sample filtering.
    saturation = None if not valid_count else float(np.mean(
        (valid_probabilities <= 1e-3) | (valid_probabilities >= 1 - 1e-3)))
    low_tail = None if not valid_count else float(np.mean(valid_probabilities < .1))
    high_tail = None if not valid_count else float(np.mean(valid_probabilities >= .9))
    return dict(sample_count=len(records), probability_finite=True, logits_finite=True,
                probability_min=float(min(p.min() for p in probabilities)),
                probability_max=float(max(p.max() for p in probabilities)),
                valid_pixels=sum(int(record['valid_mask'].sum()) for record in records),
                valid_probability_saturation_fraction=saturation,
                valid_probability_below_0_1_fraction=low_tail,
                valid_probability_at_least_0_9_fraction=high_tail,
                saturation_definition='p<=0.001 or p>=0.999; valid supervision domain only',
                new_pixels=sum(int(((record['target'] > .5) & record['valid_mask'].astype(bool)).sum())
                               for record in records))


def build_det(config, architecture, data_config=None):
    if data_config is not None:
        validate_wind_prior_config(data_config, config)
    if architecture == 'fwt':
        return FWTDet(config)
    if architecture in ('rcda12', 'rcda-derived'):
        return RCDABaseline(config, input_mode='raw12' if architecture == 'rcda12' else 'derived17')
    raise ValueError('Unknown deterministic architecture')


def to_device(batch, device):
    return dict(batch, x_norm=batch['x_norm'].to(device),
                target=batch['target'].to(device), valid_mask=batch['valid_mask'].to(device),
                physics={k: v.to(device) for k, v in batch['physics'].items()})


def deterministic_loss(logits, target, mask, loss='focal'):
    if not mask.any():
        raise ValueError('Training batch has no valid candidate supervision')
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    if loss == 'focal':
        p = logits.sigmoid()
        pt = torch.where(target > .5, p, 1 - p)
        alpha = torch.where(target > .5, .75, .25)
        bce = alpha * (1 - pt).square() * bce
    elif loss != 'bce':
        raise ValueError('Supported new-version losses: focal, bce')
    return bce[mask.bool()].mean()


def make_loader(manifest, role, normalizer, data_config, batch_size, shuffle=False, sample_cache=None):
    dataset = RCDADataset(manifest, role, normalizer, data_config)
    if sample_cache is not None:
        from .sample_cache import CachedDataset
        dataset = CachedDataset(dataset, sample_cache)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def load_det(path, device='cpu'):
    bundle = load_checkpoint(path)
    if bundle['family'] not in ('det', 'det_budget_control'):
        raise ValueError('A deterministic backbone checkpoint is required')
    model = build_det(FWTConfig(**bundle['model_config']), bundle.get('architecture', 'fwt'),
                      DataConfig(**bundle['data_config']))
    model.load_state_dict(bundle['state_dict'], strict=True)
    return model.to(device).eval(), bundle


def load_predictor(checkpoint, backbone_checkpoint=None, device='cpu'):
    bundle = load_checkpoint(checkpoint)
    if not bundle.get('selection_frozen'):
        raise ValueError('Only completed validation-selected checkpoints can be predicted')
    if bundle.get('source_sha256') != source_digest():
        raise ValueError('Frozen predictor source changed; restore the recorded implementation version')
    validate_wind_prior_config(DataConfig(**bundle['data_config']), FWTConfig(**bundle['model_config']))
    if bundle['family'] in ('det', 'det_budget_control'):
        backbone, _ = load_det(checkpoint, device)
        return backbone, None, bundle
    if backbone_checkpoint is None or sha256(backbone_checkpoint) != bundle['backbone_sha256']:
        raise ValueError('Exact frozen deterministic backbone is required')
    backbone, _ = load_det(backbone_checkpoint, device)
    cls = ConditionalResidualDiffusion if bundle['family'] == 'diff' else DeterministicResidualHead
    if bundle['family'] not in ('diff', 'residual_control'):
        raise ValueError('Unknown predictor family')
    head = cls(DiffusionConfig(**bundle['diffusion_config']))
    head.load_state_dict(bundle['state_dict'], strict=True)
    return backbone, head.to(device).eval(), bundle


def raw_probability(backbone, head, batch, sampling, generator=None, return_logits=False):
    outputs = backbone(batch['x_norm'], batch['physics'])
    if head is None:
        probability = outputs['logits_det'].sigmoid()
        return (probability, outputs['logits_det']) if return_logits else probability
    if isinstance(head, ConditionalResidualDiffusion):
        probabilities = []
        for index, sample_id in enumerate(batch['sample_id']):
            seed = int.from_bytes(hashlib.sha256(f"{sampling['seed']}:{sample_id}".encode()).digest()[:8], 'big') % (2**63 - 1)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            selected = dict(logits_det=outputs['logits_det'][index:index+1],
                            fused_pyramid=[f[index:index+1] for f in outputs['fused_pyramid']],
                            relational_feature=outputs['relational_feature'][index:index+1])
            physics = {k: v[index:index+1] for k, v in batch['physics'].items()}
            residual = head.sample(selected, physics, K=sampling['K'], steps=sampling['steps'],
                                   method=sampling['method'], generator=generator)
            probabilities.append(compute_probability(selected, residual))
        probability = torch.cat(probabilities)
    else:
        probability = head.probability(outputs, batch['physics'])
    return (probability, torch.logit(probability.clamp(1e-6, 1 - 1e-6))) if return_logits else probability


def records_from_batch(batch, probability, calibration_logits=None):
    for index, sid in enumerate(batch['sample_id']):
        record = dict(sample_id=sid, event_id=batch['event_id'][index],
                   probability=probability[index].detach().cpu().numpy(),
                   target=batch['target'][index].detach().cpu().numpy(),
                   valid_mask=batch['valid_mask'][index].detach().cpu().numpy(),
                      domain_mask=batch['domain_mask'][index].cpu().numpy())
        if calibration_logits is not None:
            record['calibration_logit'] = calibration_logits[index].detach().cpu().numpy()
        yield record


@torch.no_grad()
def validation_records(backbone, head, loader, device, sampling):
    backbone.eval()
    if head is not None:
        head.eval()
    records = []
    generator = torch.Generator(device='cpu').manual_seed(sampling['seed'])
    for batch in loader:
        batch = to_device(batch, device)
        probability, logits = raw_probability(backbone, head, batch, sampling, generator, return_logits=True)
        records.extend(records_from_batch(batch, probability, logits))
    return records


def train(args, config, family):
    from copy import deepcopy
    from .reporting import operating_point_report
    from .training_state import (training_binding, load_training_state, save_training_state,
                                 restore_rng, restore_training_parameters, sample_file_identity)

    started = time.perf_counter()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError('Positive explicit epochs, batch size and learning rate required')
    stop_after = getattr(args, 'stop_after_epoch', None)
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after < args.epochs):
        raise ValueError('stop-after-epoch must be positive and below the unchanged total epoch budget')
    resume_path = getattr(args, 'resume', None)
    data_config = DataConfig(**config['data'])
    if family in ('det', 'det_budget_control'):
        model_config = FWTConfig(**config.get('model', {}))
        validate_wind_prior_config(data_config, model_config)
    destination = output_path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Use a new empty training output directory')
    manifest = audited_manifest(args.manifest)
    digest = manifest_digest(manifest)
    normalizer = NormalizationStats.load(args.normalizer)
    seed_everything(args.seed)
    sampling = dict(K=args.samples, steps=args.sampling_steps, method=args.sampler, seed=args.seed,
                    seed_strategy='sha256_base_seed_sample_id_v1')
    head, backbone_bundle = None, None
    architecture = getattr(args, 'architecture', 'fwt')
    if family in ('det', 'det_budget_control'):
        backbone = build_det(model_config, architecture, data_config).to(args.device)
        model = backbone
        role = 'backbone_train'
    else:
        backbone, backbone_bundle = load_det(args.backbone, args.device)
        if backbone_bundle['family'] != 'det' or backbone_bundle.get('architecture', 'fwt') != 'fwt':
            raise ValueError('Residual models require the FWT backbone trained on Tb only')
        if backbone_bundle.get('source_sha256') != source_digest():
            raise ValueError('Backbone source version differs from the current implementation')
        if not backbone_bundle.get('selection_frozen'):
            raise ValueError('Freeze the completed backbone selection before residual training')
        for key, expected in [('manifest_sha256', digest), ('normalizer_sha256', sha256(args.normalizer))]:
            if backbone_bundle[key] != expected:
                raise ValueError('Backbone data/normalization version mismatch: ' + key)
        backbone.requires_grad_(False)
        model_config = FWTConfig(**backbone_bundle['model_config'])
        validate_wind_prior_config(data_config, model_config)
        if asdict(DataConfig(**backbone_bundle['data_config'])) != asdict(data_config):
            raise ValueError('Residual data configuration differs from the frozen backbone')
        settings = dict(config.get('diffusion', {}))
        settings.update(backbone_feature_channels=model_config.decoder_widths,
                        relational_channels=model_config.prop_channels)
        diffusion_config = DiffusionConfig(**settings)
        cls = ConditionalResidualDiffusion if family == 'diff' else DeterministicResidualHead
        head = cls(diffusion_config).to(args.device)
        model, role = head, 'residual_train'
    cache_gib = getattr(args, 'sample_cache_gib', 0.)
    if not math.isfinite(cache_gib) or not 0 <= cache_gib <= 64:
        raise ValueError('Sample cache must be a finite budget from 0 to 64 GiB')
    validation_batch_size = getattr(args, 'validation_batch_size', None) or args.batch_size
    if type(validation_batch_size) is not int or validation_batch_size < 1:
        raise ValueError('Validation batch size must be a positive integer')
    sample_cache = None
    if cache_gib:
        from .sample_cache import SharedTensorCache
        sample_cache = SharedTensorCache(int(cache_gib * 1024**3))
    train_loader = make_loader(manifest, role, normalizer, data_config, args.batch_size, True,
                               sample_cache=sample_cache)
    if family == 'det_budget_control':
        extra = make_loader(manifest, 'residual_train', normalizer, data_config, args.batch_size,
                            sample_cache=sample_cache).dataset
        combined = ConcatDataset([train_loader.dataset, extra])
        combined.samples = train_loader.dataset.samples + extra.samples
        train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = make_loader(manifest, 'validation', normalizer, data_config, validation_batch_size,
                             sample_cache=sample_cache)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    provenance = dict(schema='fwt_checkpoint_v2', family=family, architecture=architecture,
                      model_config=asdict(model_config), data_config=asdict(data_config),
                      manifest_sha256=digest, normalizer_sha256=sha256(args.normalizer),
                      training_sample_ids=[s['sample_id'] for s in train_loader.dataset.samples],
                      validation_sample_ids=[s['sample_id'] for s in val_loader.dataset.samples],
                      sampling=sampling if family == 'diff' else None,
                      selection_metric='pooled_validation_new_pixel_day_AP_max',
                      selection_tie_rule='earliest_epoch', seed=args.seed,
                      loss=args.loss if head is None else (('clean_probability_bce' if diffusion_config.prediction_type == 'clean_probability'
                                                           else diffusion_config.prediction_type + '_mse')
                                                          if family == 'diff' else 'residual_mse'),
                      optimizer='AdamW', learning_rate=args.learning_rate,
                      batch_size=args.batch_size, requested_epochs=args.epochs,
                      initialization='new_random_initialization', selection_frozen=False)
    provenance['source_sha256'] = source_digest()
    if head is not None:
        provenance.update(diffusion_config=asdict(diffusion_config), backbone_sha256=sha256(args.backbone))
        if family == 'diff':
            provenance['prediction_type'] = diffusion_config.prediction_type
            provenance['training_timestep_sampling'] = diffusion_config.training_timestep_sampling
    provenance['parameters'] = dict(optimized=sum(p.numel() for p in model.parameters()),
                                   trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                   backbone=sum(p.numel() for p in backbone.parameters()))
    provenance['device'] = str(args.device)
    provenance['cpu_threads'] = torch.get_num_threads()
    provenance['performance'] = dict(sample_cache_gib=cache_gib, num_workers=0,
                                     validation_batch_size=validation_batch_size)
    provenance['manifest_file_sha256'] = sha256(args.manifest)
    # Explicit defaults bind AdamW's actual recipe, not merely its class name.
    provenance['optimizer_defaults'] = deepcopy(optimizer.defaults)
    samples = train_loader.dataset.samples + val_loader.dataset.samples
    binding = training_binding(provenance, config, sampling, samples)
    restart = load_training_state(resume_path, binding) if resume_path else None
    start_epoch = restart['completed_epochs'] if restart else 0
    if stop_after is not None and stop_after <= start_epoch:
        raise ValueError('Controlled stop must be later than the restored completed epoch')
    if restart:
        restore_training_parameters(restart, model, optimizer)
        provenance['resume'] = dict(training_state_sha256=sha256(resume_path),
                                    completed_epochs=start_epoch, source=str(Path(resume_path).resolve()))
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'configuration.json', provenance)
    diagnostics = dict(schema='fwt_training_diagnostics_v1', status='running', family=family,
                       parameters=provenance['parameters'], device=provenance['device'],
                       cpu_threads=provenance['cpu_threads'], source_sha256=provenance['source_sha256'],
                       total_updates=0, completed_epochs=0, remaining_epochs=args.epochs,
                       first_batch=None, epochs=[], frozen_backbone=None,
                       wall_seconds_scope='current_process_segment')
    if family == 'diff':
        diagnostics['prediction_type'] = diffusion_config.prediction_type
        diagnostics['training_timestep_sampling'] = diffusion_config.training_timestep_sampling
        diagnostics.update(total_diffusion_timestep_exposures=0,
                           total_diffusion_denoiser_evaluations=0, total_diffusion_noise_draws=0)
    if head is not None:
        diagnostics['frozen_backbone'] = dict(
            state_sha256_before=model_state_digest(backbone), state_sha256_after=None,
            unchanged=None, parameters_require_grad=False, gradients_absent=None)
    atomic_json(destination / 'diagnostics.json', diagnostics)
    history, best, updates = [], -1.0, 0
    selected = None
    selected_validation = None
    if restart:
        history, best, updates = restart['history'], restart['best_validation_ap'], restart['updates']
        selected, selected_validation = restart['selected'], restart['selected_validation']
        diagnostics = restart['diagnostics']
        diagnostics.update(status='running', resume=provenance['resume'])
        # Best may remain in a previous segment; make this new directory a
        # self-contained run while keeping selection unfrozen until completion.
        atomic_checkpoint(destination / 'best.pth', selected)
        atomic_json(destination / 'history.json', history)
        atomic_json(destination / 'diagnostics.json', diagnostics)
        restore_rng(restart['rng_state'], args.device)
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        if head is not None:
            backbone.eval()
        losses, gradient_norms = [], []
        diffusion_training_steps = []
        for batch in train_loader:
            batch = to_device(batch, args.device)
            if diagnostics['first_batch'] is None:
                diagnostics['first_batch'] = batch_diagnostics(batch)
                atomic_json(destination / 'diagnostics.json', diagnostics)
            optimizer.zero_grad(set_to_none=True)
            if head is None:
                logits = backbone(batch['x_norm'], batch['physics'])['logits_det']
                loss = deterministic_loss(logits, batch['target'], batch['valid_mask'], args.loss)
            else:
                with torch.no_grad():
                    outputs = backbone(batch['x_norm'], batch['physics'])
                loss = head.training_loss(batch['target'], outputs, batch['physics'], mask=batch['valid_mask'])
                if family == 'diff':
                    diffusion_training_steps.append(dict(
                        update_index=updates + 1, epoch_batch_index=len(losses),
                        sample_ids=list(batch['sample_id']), **deepcopy(head.last_training_diagnostics)))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; stop and retain diagnostics')
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('Missing or nonfinite gradients')
            gradient_norms.append(gradient_l2_norm(gradients))
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        records = validation_records(backbone, head, val_loader, args.device, sampling)
        prediction_checks = prediction_diagnostics(records)
        metrics = probability_metrics(records)
        ap = metrics['pooled_ap']
        if ap is None:
            raise ValueError('Validation has no valid positive pixels; AP selection is undefined')
        candidate_threshold = select_raw_threshold(records)
        operating_points = operating_point_report(records, candidate_threshold, provenance['sampling'])
        epoch_checks = dict(epoch=epoch, updates=len(losses), cumulative_updates=updates,
                            loss_finite=bool(np.isfinite(losses).all()),
                            gradient_finite=bool(np.isfinite(gradient_norms).all()),
                            gradient_l2_min=min(gradient_norms), gradient_l2_max=max(gradient_norms),
                            gradient_l2_mean=float(np.mean(gradient_norms)),
                            nonzero_gradient_updates=sum(norm > 0 for norm in gradient_norms),
                            all_updates_have_nonzero_gradients=all(norm > 0 for norm in gradient_norms),
                            validation_predictions=prediction_checks)
        if family == 'diff':
            epoch_checks['diffusion_training_steps'] = diffusion_training_steps
            epoch_checks['diffusion_timestep_exposures'] = sum(
                len(group) for step in diffusion_training_steps for group in step['timestep_groups'])
            epoch_checks['diffusion_denoiser_evaluations'] = sum(
                step['denoiser_evaluations'] for step in diffusion_training_steps)
            epoch_checks['diffusion_noise_draws'] = sum(step['noise_draws'] for step in diffusion_training_steps)
            for key in ('diffusion_timestep_exposures', 'diffusion_denoiser_evaluations', 'diffusion_noise_draws'):
                diagnostics['total_' + key] += epoch_checks[key]
        history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)), validation=metrics,
                            validation_operating_points=operating_points, training_checks=epoch_checks))
        if ap > best:
            best = ap
            selected = dict(provenance, epoch=epoch, validation_ap=ap,
                            raw_threshold=candidate_threshold,
                            raw_threshold_purpose='validation_selection_optimistic',
                            raw_threshold_rule=operating_points['threshold_rule'], updates=updates,
                            state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            # Cache only the AP-selected epoch; the last epoch may not be selected.
            selected_validation = dict(epoch=epoch, validation=metrics, operating_points=operating_points)
            atomic_checkpoint(destination / 'best.pth', selected)
        epoch_checks['wall_seconds'] = time.perf_counter() - epoch_started
        diagnostics['epochs'].append(epoch_checks)
        diagnostics.update(total_updates=updates, completed_epochs=epoch + 1,
                           remaining_epochs=args.epochs - epoch - 1,
                           wall_seconds=time.perf_counter() - started)
        if sample_cache is not None:
            diagnostics['sample_cache'] = sample_cache.stats()
        # Every restart boundary must still describe this exact execution.
        if (provenance['source_sha256'] != source_digest() or
                provenance['normalizer_sha256'] != sha256(args.normalizer) or
                provenance['manifest_file_sha256'] != sha256(args.manifest) or
                binding['input_files'] != sample_file_identity(samples) or
                (head is not None and provenance['backbone_sha256'] != sha256(args.backbone))):
            raise RuntimeError('Training inputs/source changed; cannot save a resumable epoch or freeze selection')
        if head is not None:
            frozen = diagnostics['frozen_backbone']
            frozen.update(state_sha256_after=model_state_digest(backbone),
                          parameters_require_grad=any(p.requires_grad for p in backbone.parameters()),
                          gradients_absent=all(p.grad is None for p in backbone.parameters()))
            frozen['unchanged'] = frozen['state_sha256_before'] == frozen['state_sha256_after']
            if not frozen['unchanged'] or frozen['parameters_require_grad'] or not frozen['gradients_absent']:
                raise RuntimeError('Frozen backbone state or gradient isolation changed during residual training')
        if stop_after == epoch + 1:
            diagnostics.update(status='stopped_after_epoch', completed_epochs=epoch + 1,
                               remaining_epochs=args.epochs - epoch - 1, selection_frozen=False)
        atomic_json(destination / 'history.json', history)
        atomic_json(destination / 'diagnostics.json', diagnostics)
        save_training_state(destination / 'last.pth', binding=binding, model=model, optimizer=optimizer,
            device=args.device, completed_epochs=epoch + 1, history=history, selected=selected,
            selected_validation=selected_validation, diagnostics=diagnostics, updates=updates, best=best)
        print(f'epoch={epoch} family={family} train_loss={np.mean(losses):.6f} val_AP={ap:.6f}', flush=True)
        if stop_after == epoch + 1:
            return dict(training_state=str(destination / 'last.pth'), family=family,
                        status='stopped_after_epoch', completed_epochs=epoch + 1,
                        requested_epochs=args.epochs, selection_frozen=False)
    if provenance['source_sha256'] != source_digest():
        raise RuntimeError('Source changed during training; selection cannot be frozen')
    if head is not None:
        frozen = diagnostics['frozen_backbone']
        frozen['state_sha256_after'] = model_state_digest(backbone)
        frozen['unchanged'] = frozen['state_sha256_before'] == frozen['state_sha256_after']
        frozen['parameters_require_grad'] = any(p.requires_grad for p in backbone.parameters())
        frozen['gradients_absent'] = all(p.grad is None for p in backbone.parameters())
        atomic_json(destination / 'diagnostics.json', diagnostics)
        if not frozen['unchanged'] or frozen['parameters_require_grad'] or not frozen['gradients_absent']:
            raise RuntimeError('Frozen backbone state or gradient isolation changed during residual training')
    selected.update(selection_frozen=True, completed_epochs=args.epochs, total_updates=updates)
    if restart:
        selected['resume'] = provenance['resume']
    atomic_checkpoint(destination / 'best.pth', selected)
    atomic_json(destination / 'selection.json', {key: value for key, value in selected.items() if key != 'state_dict'})
    atomic_json(destination / 'validation_selected.json', dict(
        schema='fwt_selected_validation_report_v1', purpose='validation_selection_optimistic',
        independent_evaluation=False, selection_frozen=True,
        checkpoint_sha256=sha256(destination / 'best.pth'), source_sha256=provenance['source_sha256'],
        manifest_sha256=digest, normalizer_sha256=provenance['normalizer_sha256'],
        validation_sample_ids=provenance['validation_sample_ids'], sampling=provenance['sampling'],
        selection_metric=provenance['selection_metric'], selection_tie_rule=provenance['selection_tie_rule'],
        raw_threshold=selected['raw_threshold'], raw_threshold_rule=selected['raw_threshold_rule'],
        **selected_validation))
    diagnostics.update(status='completed', wall_seconds=time.perf_counter() - started,
                       completed_epochs=args.epochs, remaining_epochs=0, selection_frozen=True,
                       selected_epoch=selected['epoch'])
    atomic_json(destination / 'diagnostics.json', diagnostics)
    return dict(checkpoint=str(destination / 'best.pth'), family=family, validation_ap=best)
