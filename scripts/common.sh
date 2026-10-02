#!/usr/bin/env bash
# Shared root, path, cache and allocation policy. Safe to source repeatedly.
# Executable entrypoints own strict mode. Sourcing this library must not change
# a login shell's options or terminate it when a policy check fails.
if [ -z "${BASH_VERSION:-}" ]; then
    printf 'TDN: common.sh requires Bash; run the submission scripts with bash.\n' >&2
    return 2
fi
TDN_CARC_ROOT=/home1/aadaniel/projects/TDN
TDN_EXPECTED_USER=aadaniel
TDN_EXPECTED_ACCOUNT=anakano_81
_tdn_script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 2
TDN_REPO_ROOT="$(CDPATH= cd -- "$_tdn_script_dir/.." && pwd -P)" || return 2
export TDN_REPO_ROOT TDN_EXPECTED_USER TDN_EXPECTED_ACCOUNT

tdn_die() {
    printf 'TDN: %s\n' "$*" >&2
    if [[ $- == *i* ]]; then return 2; fi
    exit 2
}
tdn_inside() {
    local value="${1:-}" resolved
    [[ "$value" == /* ]] || { tdn_die "Path must be absolute: $value"; return 2; }
    resolved="$(realpath -m -- "$value")" || { tdn_die "Cannot resolve path: $value"; return 2; }
    [[ "$resolved" == "$TDN_REPO_ROOT" || "$resolved" == "$TDN_REPO_ROOT/"* ]] || {
        tdn_die "Path leaves the project, including through a symlink: $value"
        return 2
    }
    printf '%s\n' "$resolved"
}
tdn_mkdir() {
    local path
    for path in "$@"; do
        path="$(tdn_inside "$path")" || return 2
        mkdir -p -- "$path" || { tdn_die "Cannot create project path: $path"; return 2; }
        tdn_inside "$path" >/dev/null || return 2
        [[ -w "$path" ]] || { tdn_die "Project path is not writable: $path"; return 2; }
    done
}
tdn_runtime_policy() {
    [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" ]] || { tdn_die "CARC execution requires $TDN_CARC_ROOT"; return 2; }
    [[ "$(id -un)" == "$TDN_EXPECTED_USER" ]] || { tdn_die "CARC execution requires user $TDN_EXPECTED_USER"; return 2; }
    [[ -z "${TDN_LOCAL_TEST_ROOT:-}" ]] || { tdn_die 'Local test override cannot submit or run CARC jobs'; return 2; }
    [[ "${CARC_ACCOUNT:-$TDN_EXPECTED_ACCOUNT}" == "$TDN_EXPECTED_ACCOUNT" ]] || {
        tdn_die "Charging account must be $TDN_EXPECTED_ACCOUNT"
        return 2
    }
}
tdn_local_policy() {
    if [[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]]; then
        [[ "${TDN_LOCAL_TEST_ROOT:-}" == "$TDN_REPO_ROOT" ]] || {
            tdn_die 'Outside CARC, explicitly set TDN_LOCAL_TEST_ROOT to this checkout for CPU validation'
            return 2
        }
    else
        tdn_runtime_policy || return 2
    fi
}
tdn_load_python_module() {
    if [[ -n "${TDN_PYTHON_MODULE:-}" ]]; then
        [[ "${TDN_PYTHON_MODULE,,}" != *conda* ]] || { tdn_die 'Choose a Python interpreter module'; return 2; }
        if [[ "${TDN_LOADED_PYTHON_MODULE:-}" != "$TDN_PYTHON_MODULE" ]]; then
            type module >/dev/null 2>&1 || { tdn_die 'Selected Python module requires a CARC module shell'; return 2; }
            module load "$TDN_PYTHON_MODULE" || { tdn_die "Cannot load Python module: $TDN_PYTHON_MODULE"; return 2; }
            export TDN_LOADED_PYTHON_MODULE="$TDN_PYTHON_MODULE"
        fi
    fi
    python3 -c 'import sys; from pathlib import Path; assert (3,11) <= sys.version_info[:2] <= (3,13), "Use tested Python 3.11-3.13"; assert not (Path(sys.base_prefix)/"conda-meta").exists(), "Use a non-Conda base Python"' || {
        tdn_die 'Standalone Python 3.11-3.13 is required'
        return 2
    }
}
tdn_prepare_env() {
    local cache="$TDN_REPO_ROOT/.cache"
    [[ -z "${CONDA_PREFIX:-}" && "${CONDA_SHLVL:-0}" == 0 ]] || { tdn_die 'Deactivate the active Conda environment before using the Python venv'; return 2; }
    tdn_mkdir "$cache" "$cache/temp" "$cache/pip" "$cache/xdg" "$cache/torch" \
        "$cache/inductor" "$cache/triton" "$cache/extensions" "$cache/matplotlib" \
        "$cache/cuda" "$cache/pycache" "$cache/pytest" "$cache/xdg-config" "$cache/xdg-data" \
        "$cache/xdg-state" "$TDN_REPO_ROOT/logs" "$TDN_REPO_ROOT/runs" || return 2
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
    tdn_inside "$TDN_ENV_PREFIX" >/dev/null || return 2
}
tdn_python() {
    [[ -x "$TDN_REPO_ROOT/.venv/bin/python" ]] || { tdn_die 'Python venv missing; run allocated setup first'; return 2; }
    [[ -f "$TDN_REPO_ROOT/.venv/pyvenv.cfg" ]] || { tdn_die 'Project interpreter is not a Python venv'; return 2; }
    tdn_inside "$TDN_REPO_ROOT/.venv" >/dev/null || return 2
    "$TDN_REPO_ROOT/.venv/bin/python" -c 'import sys; from pathlib import Path; expected=Path(sys.argv[1]).resolve(); assert Path(sys.prefix).resolve()==expected and sys.prefix!=sys.base_prefix, "Expected the project Python venv"; assert not (Path(sys.base_prefix)/"conda-meta").exists(), "Use a non-Conda base Python"' "$TDN_REPO_ROOT/.venv" || {
        tdn_die 'Project Python venv validation failed'
        return 2
    }
    printf '%s\n' "$TDN_REPO_ROOT/.venv/bin/python"
}
tdn_allocation() {
    [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_STEP_ID:-}" ]] || {
        tdn_die 'Work must run inside an srun task in a Slurm allocation'
        return 2
    }
    [[ "${SLURM_JOB_ACCOUNT:-}" == "$TDN_EXPECTED_ACCOUNT" ]] || {
        tdn_die 'Slurm allocation does not confirm the required charging account'
        return 2
    }
}
