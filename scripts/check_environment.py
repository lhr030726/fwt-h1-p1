"""Record the dedicated Ubuntu/Blackwell runtime; GPU checks require --gpu-test.

The optional check allocates only tiny synthetic tensors. It never imports the
FWT model, downloads data, reads checkpoints or performs backward/optimization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

REPOSITORY = Path(__file__).resolve().parents[1]
MARKER = '.fwt-managed-environment-v1'
MARKER_PREFIX = 'fwt-ubuntu22-python310-torch271-cu128-v1|'
EXPECTED = {'torch': '2.7.1', 'numpy': '1.22.3', 'scipy': '1.11.4',
            'scikit-learn': '1.2.2', 'pyproj': '3.7.1'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_managed_environment(root, prefix, base_prefix, version):
    root = Path(root).resolve()
    environment = root / '.venv-fwt'
    marker = environment / MARKER
    if (environment.is_symlink() or marker.is_symlink()
            or not marker.is_file()
            or marker.read_text(encoding='utf-8').rstrip('\r\n') != MARKER_PREFIX + root.as_posix()):
        raise RuntimeError('Use the dedicated environment created by scripts/setup_ubuntu22.sh')
    if (Path(prefix).resolve() != environment or prefix == base_prefix or tuple(version[:2]) != (3, 10)):
        raise RuntimeError('Activate this repository\'s Python 3.10 .venv-fwt first')


def check_versions(version):
    actual = {name: version(name) for name in EXPECTED}
    wrong = {name: value for name, value in actual.items()
             if (value.split('+')[0] if name == 'torch' else value) != EXPECTED[name]}
    if wrong:
        raise RuntimeError('Pinned runtime versions differ: ' + json.dumps(wrong, sort_keys=True))
    return actual


def gpu_probe(torch, device_index):
    if torch.__version__.split('+')[0] != '2.7.1' or torch.version.cuda != '12.8':
        raise RuntimeError('Blackwell setup requires torch 2.7.1 from the official cu128 index')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; verify platform GPU exposure and host driver, then the cu128 wheel')
    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError('Selected visible GPU index is unavailable')
    device = torch.device(f'cuda:{device_index}')
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability != (12, 0):
        raise RuntimeError('Expected RTX PRO 6000 Blackwell capability 12.0; actual selected GPU is '
                           + str(capability) + '. Check CUDA_VISIBLE_DEVICES/device-index.')
    result = dict(device=str(device), name=torch.cuda.get_device_name(device),
        capability=list(capability), cuda_runtime=torch.version.cuda,
        compiled_architectures=torch.cuda.get_arch_list(),
        visible_device_count=torch.cuda.device_count(),
        visible_device_selection=os.environ.get('CUDA_VISIBLE_DEVICES'),
        cudnn_version=torch.backends.cudnn.version())
    with torch.inference_mode():
        left = torch.ones((16, 16), device=device, dtype=torch.float32)
        product = left @ left
        image = torch.ones((1, 1, 8, 8), device=device, dtype=torch.float32)
        kernel = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
        convolution = torch.nn.functional.conv2d(image, kernel)
        torch.cuda.synchronize(device)
        matmul_sum = float(product.sum().item())
        convolution_sum = float(convolution.sum().item())
    if matmul_sum != 4096.0 or convolution_sum != 324.0:
        raise RuntimeError('Synthetic CUDA arithmetic returned unexpected values')
    result.update(status='passed', matmul_sum=matmul_sum, convolution_sum=convolution_sum,
        synthetic_only=True, backward_calls=0, optimizer_updates=0, model_loaded=False)
    return result


def collect(root, *, gpu_test=False, device_index=0, run=subprocess.run,
            version=importlib.metadata.version, distributions=importlib.metadata.distributions,
            torch_loader=None):
    """Separated dependencies allow CPU-only mocks without importing torch."""
    root = Path(root).resolve()
    result = dict(schema_version='fwt_environment_snapshot_v1', status='failed',
        created_at=datetime.now(timezone.utc).isoformat(), purpose='runtime_compatibility_only',
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        repository=str(root), gpu_check_requested=gpu_test, gpu_check_completed=False,
        training_executed=False, model_loaded=False, array_or_checkpoint_reads=0,
        migration='Historical checks used torch 2.0.0; this target is torch 2.7.1/cu128. Numerical equivalence is untested.')
    try:
        check_managed_environment(root, sys.prefix, sys.base_prefix, sys.version_info)
        result['versions'] = check_versions(version)
        result['requirements_sha256'] = {name: sha256(root / name)
            for name in ('requirements.txt', 'requirements-torch.txt')}
        # Record resolved dependency versions without URLs, tokens or the full environment.
        result['installed_distributions'] = sorted(
            [{'name': dist.metadata.get('Name', 'unknown'), 'version': dist.version}
             for dist in distributions()], key=lambda item: item['name'].casefold())
        checked = run([sys.executable, '-I', '-m', 'pip', '--isolated', 'check'],
                      capture_output=True, text=True, check=False, timeout=120)
        result['pip_check'] = dict(returncode=checked.returncode, stdout=checked.stdout, stderr=checked.stderr)
        if checked.returncode:
            raise RuntimeError('pip check reported incompatible dependencies')
        if gpu_test:
            driver = run(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=False, timeout=30)
            result['nvidia_smi'] = dict(returncode=driver.returncode, stdout=driver.stdout, stderr=driver.stderr)
            if driver.returncode:
                raise RuntimeError('nvidia-smi failed; ask the platform to expose its GPU/driver. No driver changes are performed.')
            if torch_loader is None:
                import torch
            else:
                torch = torch_loader()
            result['gpu'] = gpu_probe(torch, device_index)
            result['gpu_check_completed'] = True
        result['status'] = 'passed'
    except Exception as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc))
    return result


def write_snapshot(root, result):
    root = Path(root).resolve()
    output = root / 'runs' / 'environment'
    if root not in output.resolve().parents:
        raise RuntimeError('Environment snapshot directory escapes the repository')
    output.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8] + '.json'
    path = output / name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-test', action='store_true', help='Run small synthetic CUDA matmul/conv, no training')
    parser.add_argument('--device-index', type=int, default=0, help='Index among visible GPUs')
    args = parser.parse_args(argv)
    if args.device_index < 0:
        parser.error('device-index must be nonnegative')
    result = collect(REPOSITORY, gpu_test=args.gpu_test, device_index=args.device_index)
    path = write_snapshot(REPOSITORY, result)
    print(json.dumps(dict(status=result['status'], snapshot=str(path),
        gpu_check_completed=result['gpu_check_completed'], training_executed=False,
        error=result.get('error')), ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
