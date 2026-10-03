#!/usr/bin/env bash
# Resources and absolute logs are supplied by carc.sh; one CPU allocation.
#SBATCH --job-name=tdn-carc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@180
#SBATCH --open-mode=append
set -euo pipefail
source "${TDN_REPO_ROOT:?Submission wrapper must export the repository root}/scripts/common.sh"
tdn_runtime_policy
: "${SLURM_JOB_ID:?Run with scripts/carc.sh}"
[[ "${SLURM_JOB_ACCOUNT:-}" == anakano_81 ]] || tdn_die 'Wrong allocation account'
tdn_prepare_env
tdn_load_python_module
export TDN_WORKFLOW_PHASE=cpu
cd "$TDN_REPO_ROOT"
exec srun --ntasks=1 --unbuffered bash "$TDN_REPO_ROOT/scripts/carc_worker.sh"
