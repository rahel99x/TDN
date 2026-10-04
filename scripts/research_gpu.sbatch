#!/usr/bin/env bash
# One explicit, bounded A100 benchmark after a completed CPU hypothesis screen.
#SBATCH --job-name=tdn-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --signal=TERM@120
#SBATCH --open-mode=append
set -euo pipefail
source "${TDN_REPO_ROOT:?Use scripts/carc_research.sh}/scripts/common.sh"
source "$TDN_REPO_ROOT/scripts/carc_defaults.sh"
tdn_runtime_policy
[[ "${SLURM_JOB_ACCOUNT:-}" == anakano_81 ]] || tdn_die 'Wrong allocation account'
tdn_prepare_env
tdn_load_python_module
export TDN_RESEARCH_PHASE=gpu TORCH_CUDA_ARCH_LIST=8.0
cd "$TDN_REPO_ROOT"
# Preflight, checkpoint loading and benchmarking share this exact srun task.
exec srun --ntasks=1 --gpus-per-task=a100:1 --unbuffered bash "$TDN_REPO_ROOT/scripts/research_worker.sh"
