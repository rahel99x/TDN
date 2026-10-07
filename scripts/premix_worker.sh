#!/usr/bin/env bash
# This file is both the sbatch script and its single allocated srun task entry.
set -euo pipefail
source "${TDN_REPO_ROOT:?Use scripts/carc_premix.sh}/scripts/common.sh"
source "$TDN_REPO_ROOT/scripts/carc_defaults.sh"
tdn_runtime_policy
[[ "${SLURM_JOB_ACCOUNT:-}" == anakano_81 ]] || tdn_die 'Wrong allocation account'
[[ -z "${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}${TDN_LOCAL_TEST_ROOT:-}" ]] || tdn_die 'Clear local/desktop overrides'
: "${TDN_PREMIX_WORKFLOW:?}" "${TDN_PREMIX_STAGE:?}"
case "$TDN_PREMIX_STAGE" in accuracy|scaling|prepare|neural) ;; *) tdn_die 'Unknown premix stage' ;; esac
tdn_prepare_env
cd "$TDN_REPO_ROOT"
if [[ "${1:-}" != --task ]]; then
    tdn_load_python_module
    gpu=()
    if [[ "$TDN_PREMIX_STAGE" == neural ]]; then
        gpu=(--gpus-per-task=a100:1)
        export TORCH_CUDA_ARCH_LIST=8.0
    fi
    exec srun --ntasks=1 --unbuffered "${gpu[@]}" bash "$TDN_REPO_ROOT/scripts/premix_worker.sh" --task
fi
tdn_allocation
unset TDN_REQUIRE_GPU_TESTS TDN_GATE_REPORT TDN_CALIBRATION_REPORT TDN_DESKTOP_CUDA_DEVICE TDN_TOWER_DIR
exec python3 "$TDN_REPO_ROOT/scripts/premix_workflow.py" worker \
    --workflow "$TDN_PREMIX_WORKFLOW" --stage "$TDN_PREMIX_STAGE"
