#!/usr/bin/env bash
# Resources, exact account, partition and output paths come from submit.sh.
#SBATCH --job-name=tdn-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@180
#SBATCH --open-mode=append
set -euo pipefail
source "${TDN_REPO_ROOT:?Submission wrapper must export the repository root}/scripts/common.sh"
tdn_runtime_policy
: "${SLURM_JOB_ID:?Run with the submission wrapper}"
[[ "${SLURM_JOB_ACCOUNT:-}" == anakano_81 ]] || tdn_die 'Wrong allocation account'
tdn_prepare_env
tdn_load_python_module
cd "$TDN_REPO_ROOT"
exec srun --ntasks=1 --unbuffered bash "$TDN_REPO_ROOT/scripts/run_stage.sh"
