#!/usr/bin/env bash
# Local CPU implementation checks; native work uses the separate Raven worker.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/scripts/common.sh"
[[ "$root" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use the separate CARC allocation workflow'
if [[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${SLURM_JOB_ACCOUNT:-}" ]]; then
    tdn_die 'The local CPU wrapper requires an unallocated development shell; use the native Raven worker inside Slurm'
fi
tdn_prepare_env
python="$(tdn_python)"
export PYTHONDONTWRITEBYTECODE=1 TDN_EXECUTION_MODE=local_cpu
tdn_mkdir "$root/.runtime/locks"
exec 8<>"$root/.cache/carc-phase.lock"
flock -s -n 8 || tdn_die 'Project virtual environment is being modified; retry after setup'
exec 9>"$root/.runtime/locks/response-preconditioner-local.lock"
flock -n 9 || tdn_die 'Another response-preconditioner local verification is active'
cd -- "$root"
exec "$python" "$root/scripts/response_preconditioner.py" "$@"
