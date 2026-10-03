#!/usr/bin/env bash
# Resources and absolute logs are supplied by carc.sh; one A100 allocation.
#SBATCH --job-name=tdn-carc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --signal=USR1@180
#SBATCH --open-mode=append
set -euo pipefail
source "${TDN_REPO_ROOT:?Submission wrapper must export the repository root}/scripts/common.sh"
tdn_runtime_policy
: "${SLURM_JOB_ID:?Run with scripts/carc.sh}"
[[ "${SLURM_JOB_ACCOUNT:-}" == anakano_81 ]] || tdn_die 'Wrong allocation account'
tdn_prepare_env
tdn_load_python_module
export TDN_WORKFLOW_PHASE=gpu TORCH_CUDA_ARCH_LIST=8.0
cd "$TDN_REPO_ROOT"
# Slurm controls visibility; preflight and every GPU stage share this task.
exec srun --ntasks=1 --gpus-per-task=a100:1 --unbuffered bash "$TDN_REPO_ROOT/scripts/carc_worker.sh"
