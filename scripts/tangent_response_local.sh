#!/usr/bin/env bash
# The reduced CPU implementation rehearsal has no Slurm/GPU readiness meaning.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/scripts/common.sh"
[[ "$root" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use a real CARC allocation through its separate workflow'
if [[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${SLURM_JOB_ACCOUNT:-}" ]]; then
    printf 'Tangent response: CPU implementation smoke must run outside an allocation.\n' >&2
    exit 2
fi
tdn_prepare_env
python="$(tdn_python)"
export PYTHONDONTWRITEBYTECODE=1
export TDN_EXECUTION_MODE=local_cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$root/.runtime/cache" MPLCONFIGDIR="$root/.runtime/matplotlib"
export TMPDIR="$root/.runtime/tmp"
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$root/.runtime/locks"
exec 8<>"$root/.cache/carc-phase.lock"
flock -s -n 8 || tdn_die 'Project virtual environment is being modified; retry after setup'
exec 9>"$root/.runtime/locks/tangent-response-cpu-smoke.lock"
flock -n 9 || { printf 'Tangent response: another CPU rehearsal is active.\n' >&2; exit 2; }
cd -- "$root"
exec "$python" "$root/scripts/tangent_response.py" "$@"
