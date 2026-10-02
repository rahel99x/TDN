#!/usr/bin/env bash
# Run by the allocated CPU setup stage, or with an explicit local CPU test override.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" ]]; then
    tdn_runtime_policy
    tdn_allocation
    [[ "${TDN_DEVICE:-cpu}" == cpu ]] || tdn_die 'Environment setup belongs in a CPU task'
else
    tdn_local_policy
fi
tdn_prepare_env
tdn_load_python_module
base_python="$(command -v python3)"
: "${TORCH_VERSION:?Choose an exact tested torch version after the driver audit}"
: "${TORCH_WHEEL_INDEX:?Choose an official compatible PyTorch wheel index}"
[[ "$TORCH_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+][A-Za-z0-9.]+)?$ ]] || tdn_die 'TORCH_VERSION must be an exact release'
[[ "$TORCH_WHEEL_INDEX" =~ ^https://download\.pytorch\.org/whl/(cpu|cu[0-9]+)$ ]] ||
    tdn_die 'Use the official HTTPS CPU or CUDA wheel index'
if [[ -e "$TDN_ENV_PREFIX" ]]; then
    [[ -f "$TDN_ENV_PREFIX/pyvenv.cfg" && -x "$TDN_ENV_PREFIX/bin/python" ]] ||
        tdn_die 'Existing .venv is not a usable Python venv; preserve it and resolve manually'
    "$TDN_ENV_PREFIX/bin/python" - "$base_python" <<'PY'
import subprocess, sys
version = subprocess.check_output([sys.argv[1], '-c', 'import sys; print(sys.version_info[:3])'], text=True).strip()
if str(sys.version_info[:3]) != version or sys.prefix == sys.base_prefix:
    raise SystemExit('Existing venv interpreter is incompatible; it was not overwritten')
PY
else
    "$base_python" -m venv "$TDN_ENV_PREFIX"
fi
python="$(tdn_python)"
for directory in bin lib include; do tdn_inside "$TDN_ENV_PREFIX/$directory" >/dev/null; done
"$python" -m pip install 'setuptools==80.9.0' 'wheel==0.45.1'
"$python" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_WHEEL_INDEX"
"$python" -m pip install -r "$TDN_REPO_ROOT/requirements.txt"
"$python" -m pip install --no-deps --no-build-isolation -e "$TDN_REPO_ROOT"
"$python" -m pip check
tdn_mkdir "$TDN_REPO_ROOT/requirements"
freeze_path="$(tdn_inside "$TDN_REPO_ROOT/requirements/environment-freeze.txt")"
versions_path="$(tdn_inside "$TDN_REPO_ROOT/requirements/environment-versions.json")"
"$python" -m pip freeze > "$freeze_path"
"$python" - "$versions_path" <<'PY'
import importlib.metadata, json, os, platform, sys
packages = ('torch', 'numpy', 'scipy', 'PyYAML', 'pytest', 'mpmath', 'matplotlib', 'setuptools', 'wheel')
report = {'category': 'newly_measured_result', 'python': platform.python_version(),
          'python_executable': sys.executable, 'venv': sys.prefix,
          'torch_wheel_index': os.environ['TORCH_WHEEL_INDEX'],
          'packages': {name: importlib.metadata.version(name) for name in packages},
          'note': 'Installed software only; CUDA operation requires an allocated GPU preflight.'}
with open(sys.argv[1], 'w') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
PY
printf 'Installed Python venv and recorded the dependency freeze under the project.\n'
