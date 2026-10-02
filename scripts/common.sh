#!/usr/bin/env bash
# Shared root, path, cache and allocation policy. Safe to source repeatedly.
set -euo pipefail
TDN_CARC_ROOT=/home1/aadaniel/projects/TDN
TDN_EXPECTED_USER=aadaniel
TDN_EXPECTED_ACCOUNT=anakano_81
_tdn_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TDN_REPO_ROOT="$(cd -- "$_tdn_script_dir/.." && pwd -P)"
export TDN_REPO_ROOT TDN_EXPECTED_USER TDN_EXPECTED_ACCOUNT

tdn_die() { printf 'TDN: %s\n' "$*" >&2; exit 2; }
tdn_inside() {
    local value="${1:?path required}" resolved
    [[ "$value" == /* ]] || tdn_die "Path must be absolute: $value"
    resolved="$(realpath -m -- "$value")"
    [[ "$resolved" == "$TDN_REPO_ROOT" || "$resolved" == "$TDN_REPO_ROOT/"* ]] ||
        tdn_die "Path leaves the project, including through a symlink: $value"
    printf '%s\n' "$resolved"
}
tdn_mkdir() {
    local path
    for path in "$@"; do
        path="$(tdn_inside "$path")"
        mkdir -p -- "$path"
        tdn_inside "$path" >/dev/null
        [[ -w "$path" ]] || tdn_die "Project path is not writable: $path"
    done
}
tdn_runtime_policy() {
    [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" ]] || tdn_die "CARC execution requires $TDN_CARC_ROOT"
    [[ "$(id -un)" == "$TDN_EXPECTED_USER" ]] || tdn_die "CARC execution requires user $TDN_EXPECTED_USER"
    [[ -z "${TDN_LOCAL_TEST_ROOT:-}" ]] || tdn_die 'Local test override cannot submit or run CARC jobs'
    [[ "${CARC_ACCOUNT:-$TDN_EXPECTED_ACCOUNT}" == "$TDN_EXPECTED_ACCOUNT" ]] ||
        tdn_die "Charging account must be $TDN_EXPECTED_ACCOUNT"
}
tdn_local_policy() {
    if [[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]]; then
        [[ "${TDN_LOCAL_TEST_ROOT:-}" == "$TDN_REPO_ROOT" ]] ||
            tdn_die 'Outside CARC, explicitly set TDN_LOCAL_TEST_ROOT to this checkout for CPU validation'
    else
        tdn_runtime_policy
    fi
}
tdn_load_python_module() {
    if [[ -n "${TDN_PYTHON_MODULE:-}" ]]; then
        [[ "${TDN_PYTHON_MODULE,,}" != *conda* ]] || tdn_die 'Choose a Python interpreter module'
        if [[ "${TDN_LOADED_PYTHON_MODULE:-}" != "$TDN_PYTHON_MODULE" ]]; then
            type module >/dev/null 2>&1 || tdn_die 'Selected Python module requires a CARC module shell'
            module load "$TDN_PYTHON_MODULE"
            export TDN_LOADED_PYTHON_MODULE="$TDN_PYTHON_MODULE"
        fi
    fi
    python3 -c 'import sys; from pathlib import Path; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use tested Python 3.11-3.13"; assert not (Path(sys.base_prefix)/"conda-meta").exists(), "Use a non-Conda base Python"'
}
tdn_prepare_env() {
    local cache="$TDN_REPO_ROOT/.cache"
    [[ -z "${CONDA_PREFIX:-}" && "${CONDA_SHLVL:-0}" == 0 ]] || tdn_die 'Deactivate the active Conda environment before using the Python venv'
    tdn_mkdir "$cache" "$cache/temp" "$cache/pip" "$cache/xdg" "$cache/torch" \
        "$cache/inductor" "$cache/triton" "$cache/extensions" "$cache/matplotlib" \
        "$cache/cuda" "$cache/pycache" "$cache/pytest" "$cache/xdg-config" "$cache/xdg-data" \
        "$cache/xdg-state" "$TDN_REPO_ROOT/logs" "$TDN_REPO_ROOT/runs"
    export TMPDIR="$cache/temp" TMP="$cache/temp" TEMP="$cache/temp"
    export PIP_CACHE_DIR="$cache/pip" XDG_CACHE_HOME="$cache/xdg" TORCH_HOME="$cache/torch"
    export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
    export TORCH_EXTENSIONS_DIR="$cache/extensions" MPLCONFIGDIR="$cache/matplotlib"
    export CUDA_CACHE_PATH="$cache/cuda" PYTHONPYCACHEPREFIX="$cache/pycache"
    export XDG_CONFIG_HOME="$cache/xdg-config" XDG_DATA_HOME="$cache/xdg-data" XDG_STATE_HOME="$cache/xdg-state"
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PIP_NO_INPUT=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    unset PIP_TARGET PIP_PREFIX PYTHONUSERBASE PYTHONPATH
    export PIP_CONFIG_FILE=/dev/null PIP_USER=0
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TDN_ENV_PREFIX="$TDN_REPO_ROOT/.venv"
    tdn_inside "$TDN_ENV_PREFIX" >/dev/null
}
tdn_python() {
    [[ -x "$TDN_REPO_ROOT/.venv/bin/python" ]] || tdn_die 'Python venv missing; run allocated setup first'
    [[ -f "$TDN_REPO_ROOT/.venv/pyvenv.cfg" ]] || tdn_die 'Project interpreter is not a Python venv'
    tdn_inside "$TDN_REPO_ROOT/.venv" >/dev/null
    "$TDN_REPO_ROOT/.venv/bin/python" -c 'import sys; from pathlib import Path; expected=Path(sys.argv[1]).resolve(); assert Path(sys.prefix).resolve()==expected and sys.prefix!=sys.base_prefix, "Expected the project Python venv"; assert not (Path(sys.base_prefix)/"conda-meta").exists(), "Use a non-Conda base Python"' "$TDN_REPO_ROOT/.venv"
    printf '%s\n' "$TDN_REPO_ROOT/.venv/bin/python"
}
tdn_allocation() {
    [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] ||
        tdn_die 'Work must run inside an srun task in a Slurm allocation'
    [[ "${SLURM_JOB_ACCOUNT:-}" == "$TDN_EXPECTED_ACCOUNT" ]] ||
        tdn_die 'Slurm allocation does not confirm the required charging account'
}
