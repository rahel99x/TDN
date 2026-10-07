#!/usr/bin/env bash
# Both the sbatch entry point and the single real allocated srun task.
set -euo pipefail
: "${TDN_REPO_ROOT:?Use scripts/fedora_slurm.sh run}" "${TDN_PREMIX_WORKFLOW:?}" "${TDN_PREMIX_STAGE:?}"
[[ "$TDN_EXECUTION_MODE" == desktop-slurm ]] || { printf 'Expected explicit desktop-slurm mode\n' >&2; exit 2; }
[[ "$TDN_REPO_ROOT" != /home1/aadaniel/projects/TDN ]] || { printf 'Use the separate CARC workflow\n' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'Worker requires an actual Slurm allocation\n' >&2; exit 2; }
case "$TDN_PREMIX_STAGE" in accuracy|scaling|prepare|neural) ;; *) exit 2 ;; esac
cd -- "$TDN_REPO_ROOT"
# Configure the same contained storage before any Python, pytest or compiler work.
export TDN_PROJECT_ROOT="$TDN_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="$TDN_REPO_ROOT/.runtime/tmp" TMP="$TDN_REPO_ROOT/.runtime/tmp" TEMP="$TDN_REPO_ROOT/.runtime/tmp"
export PIP_CACHE_DIR="$TDN_REPO_ROOT/.runtime/cache/pip" XDG_CACHE_HOME="$TDN_REPO_ROOT/.runtime/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache" MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Resolve every storage path before writing, including symlink containment.
"$TDN_REPO_ROOT/.venv/bin/python" -c 'from tdn.runtime.storage import configure_storage; configure_storage()'
if [[ "${1:-}" != --task ]]; then
    gres=none
    if [[ "$TDN_PREMIX_STAGE" == neural ]]; then gres="${TDN_FEDORA_GPU_GRES:?}"; fi
    exec srun --ntasks=1 --cpus-per-task=4 --unbuffered --export=ALL --gres="$gres" bash "$TDN_REPO_ROOT/scripts/fedora_slurm_worker.sh" --task
fi
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || { printf 'Worker requires a real numeric srun step\n' >&2; exit 2; }
unset TDN_REQUIRE_GPU_TESTS TDN_GATE_REPORT TDN_CALIBRATION_REPORT TDN_DESKTOP_CUDA_DEVICE TDN_TOWER_DIR
exec "$TDN_REPO_ROOT/.venv/bin/python" "$TDN_REPO_ROOT/scripts/fedora_workflow.py" worker \
    --workflow "$TDN_PREMIX_WORKFLOW" --stage "$TDN_PREMIX_STAGE"
