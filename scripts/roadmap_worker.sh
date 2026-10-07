#!/usr/bin/env bash
# One real allocated srun task; no scheduler or CUDA spoofing.
set -euo pipefail
: "${TDN_REPO_ROOT:?Use scripts/fedora_roadmap.sh run}" "${TDN_ROADMAP_WORKFLOW:?}" "${TDN_ROADMAP_STAGE:?}" "${TDN_ROADMAP_CPUS:?}"
[[ "${TDN_EXECUTION_MODE:-}" == desktop-slurm ]] || { printf 'Expected explicit desktop-slurm mode\n' >&2; exit 2; }
[[ "$TDN_REPO_ROOT" != /home1/aadaniel/projects/TDN ]] || { printf 'Use the separate CARC workflow\n' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf 'Worker requires an actual Slurm allocation\n' >&2; exit 2; }
case "$TDN_ROADMAP_STAGE" in audit|headroom|prepare|train|confirm_prepare|confirm|policy|transfer|scaling|report) ;; *) exit 2 ;; esac
[[ "$TDN_ROADMAP_CPUS" =~ ^[1-8]$ ]] || { printf 'Invalid stage CPU request\n' >&2; exit 2; }
cd -- "$TDN_REPO_ROOT"
export TDN_PROJECT_ROOT="$TDN_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="$TDN_REPO_ROOT/.runtime/tmp" TMP="$TDN_REPO_ROOT/.runtime/tmp" TEMP="$TDN_REPO_ROOT/.runtime/tmp"
export PIP_CACHE_DIR="$TDN_REPO_ROOT/.runtime/cache/pip" XDG_CACHE_HOME="$TDN_REPO_ROOT/.runtime/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache" MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
# Single-thread deterministic kernels retain comparable complete-rollout timings;
# allocated CPU headroom covers independent FP64 teachers, tests and reporting.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$TDN_REPO_ROOT/.venv/bin/python" -c 'from tdn.runtime.storage import configure_storage; configure_storage()'
if [[ "${1:-}" != --task ]]; then
    gres=none
    case "$TDN_ROADMAP_STAGE" in train|confirm|policy|scaling) gres="${TDN_FEDORA_GPU_GRES:?}" ;; esac
    exec srun --ntasks=1 --cpus-per-task="$TDN_ROADMAP_CPUS" --unbuffered --export=ALL --gres="$gres" \
        bash "$TDN_REPO_ROOT/scripts/roadmap_worker.sh" --task
fi
[[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || { printf 'Worker requires a real numeric srun step\n' >&2; exit 2; }
unset TDN_REQUIRE_GPU_TESTS TDN_GATE_REPORT TDN_CALIBRATION_REPORT TDN_DESKTOP_CUDA_DEVICE TDN_TOWER_DIR
unset TDN_AGENDA_WORKFLOW TDN_AGENDA_STAGE TDN_AGENDA_PROTOCOL_SHA256
unset TDN_PREMIX_WORKFLOW TDN_PREMIX_STAGE TDN_PREMIX_PROTOCOL_SHA256
unset TDN_CONSISTENCY_WORKFLOW TDN_CONSISTENCY_STAGE TDN_CONSISTENCY_PROTOCOL_SHA256
exec "$TDN_REPO_ROOT/.venv/bin/python" "$TDN_REPO_ROOT/scripts/roadmap_workflow.py" worker \
    --workflow "$TDN_ROADMAP_WORKFLOW" --stage "$TDN_ROADMAP_STAGE"
