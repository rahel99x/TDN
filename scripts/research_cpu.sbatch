#!/usr/bin/env bash
# Immutable resource arguments and project-local log paths come from the wrapper.
#SBATCH --job-name=tdn-research-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=TERM@120
#SBATCH --open-mode=append
set -euo pipefail
source "${TDN_REPO_ROOT:?Use scripts/carc_research.sh}/scripts/common.sh"
source "$TDN_REPO_ROOT/scripts/carc_defaults.sh"
tdn_runtime_policy
[[ "${SLURM_JOB_ACCOUNT:-}" == anakano_81 ]] || tdn_die 'Wrong allocation account'
tdn_prepare_env
tdn_load_python_module
export TDN_RESEARCH_PHASE=cpu
cd "$TDN_REPO_ROOT"
# Signal the Python coordinator, which records interruption and terminates its child.
exec srun --ntasks=1 --unbuffered bash "$TDN_REPO_ROOT/scripts/research_worker.sh"
