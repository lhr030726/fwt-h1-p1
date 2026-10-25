#!/usr/bin/env bash
# Explicit environment setup only: no driver changes, data download or training.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$repo"
gpu_index="${1:-0}"
if [[ "$#" -gt 1 || ! "$gpu_index" =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'Usage: bash scripts/setup_ubuntu22.sh [visible_GPU_index, default 0]' >&2
    exit 2
fi
if [[ ! -r /etc/os-release ]]; then
    printf '%s\n' 'This setup requires Ubuntu 22.04.' >&2
    exit 1
fi
. /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 22.04 ]]; then
    printf '%s\n' 'This setup is pinned to Ubuntu 22.04; it does not change the OS.' >&2
    exit 1
fi
system_python=/usr/bin/python3.10
if [[ ! -x "$system_python" ]]; then
    printf '%s\n' 'Install python3.10 and python3.10-venv with apt first (see README).' >&2
    exit 1
fi
"$system_python" -I -c 'import sys, venv; assert sys.version_info[:2] == (3, 10)'
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader; then
    printf '%s\n' 'NVIDIA GPU/driver is unavailable. On a rented server/container, ask the platform to expose the GPU and provide a working host driver. This script does not install drivers or reboot.' >&2
    exit 1
fi

environment="$repo/.venv-fwt"
marker="$environment/.fwt-managed-environment-v1"
marker_text="fwt-ubuntu22-python310-torch271-cu128-v1|$repo"
if [[ -e "$environment" || -L "$environment" ]]; then
    if [[ -L "$environment" || ! -d "$environment" || -L "$marker" || ! -f "$marker" || "$(cat -- "$marker")" != "$marker_text" ]]; then
        printf '%s\n' 'Refusing to overwrite an unknown .venv-fwt. Move it aside yourself after inspecting it; no files were deleted.' >&2
        exit 1
    fi
else
    # Mark only the directory this invocation created. Interrupted setup can be rerun.
    mkdir -- "$environment"
    printf '%s\n' "$marker_text" > "$marker"
fi
# Re-running venv without --clear repairs an interrupted managed setup and
# preserves installed packages. The system interpreter is never the active venv.
"$system_python" -I -m venv "$environment"
python="$environment/bin/python"
if [[ ! -x "$python" ]]; then
    printf '%s\n' 'Managed environment is incomplete; inspect .venv-fwt before retrying.' >&2
    exit 1
fi
"$python" -I -c 'import pathlib, sys; expected=pathlib.Path(sys.argv[1]).resolve(); assert sys.version_info[:2] == (3,10) and pathlib.Path(sys.prefix).resolve() == expected and sys.prefix != sys.base_prefix, "Expected dedicated Python 3.10 venv"' "$environment"

# Absolute venv interpreter + isolated pip: never system/global pip or a user index.
"$python" -I -m pip --isolated --require-virtualenv install --no-cache-dir --disable-pip-version-check --no-input \
    --index-url https://download.pytorch.org/whl/cu128 -r "$repo/requirements-torch.txt"
"$python" -I -m pip --isolated --require-virtualenv install --no-cache-dir --disable-pip-version-check --no-input \
    --index-url https://pypi.org/simple -r "$repo/requirements.txt"
"$python" -I "$repo/scripts/check_environment.py" --gpu-test --device-index "$gpu_index"
printf '%s\n' 'Environment ready. Activate it with: source .venv-fwt/bin/activate' \
    'No data was downloaded and no training was started. Follow README for the next explicit commands.'
