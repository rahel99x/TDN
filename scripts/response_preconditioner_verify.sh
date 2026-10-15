#!/usr/bin/env bash
#SBATCH --job-name=tdn-response-preconditioner-verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --output=logs/response-preconditioner/%x-%j.out
#SBATCH --error=logs/response-preconditioner/%x-%j.err
# One bounded CPU implementation check in Raven's actual Slurm allocation.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: raven launch scripts/response_preconditioner_verify.sh \
  --project "$PWD" --recipe response-preconditioner \
  --sbatch=--partition=local --sbatch="--chdir=$PWD" \
  -- --contract PATH [--mode all|verify|smoke]
Create PATH first using the project venv:
  .venv/bin/python scripts/response_preconditioner_native.py prepare
The contract freezes source, software and Fedora profile before submission.
This is CPU implementation evidence only; no historical gate is promoted.
Use scripts/response_preconditioner.py for separate local CPU verification.
HELP
    exit 0
fi
[[ -n "${RAVEN_LAUNCH_PLAN:-}" && -n "${RAVEN_RUN_DIR:-}" ]] || {
    printf 'TDN preconditioner: use raven launch for this native entry point.\n' >&2; exit 2;
}
[[ "${RAVEN_SOURCE_SCRIPT:-}" == /* && -f "$RAVEN_SOURCE_SCRIPT" ]] || {
    printf 'TDN preconditioner: Raven original source script is missing.\n' >&2; exit 2;
}
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'TDN preconditioner: an actual Slurm allocation is required; omit Raven --local.\n' >&2; exit 2;
}
root="$(cd -- "$(dirname -- "$RAVEN_SOURCE_SCRIPT")/.." && pwd -P)"
[[ "$RAVEN_SOURCE_SCRIPT" == "$root/scripts/response_preconditioner_verify.sh" && "$root" != /home1/aadaniel/projects/TDN ]] || exit 2
[[ -x "$root/.venv/bin/python" ]] || {
    printf 'TDN preconditioner: use this checkout\047s standalone .venv.\n' >&2; exit 2;
}
cd -- "$root"
export TDN_PROJECT_ROOT="$root" TDN_REPO_ROOT="$root" TDN_EXECUTION_MODE=desktop-slurm
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="$root/.runtime/tmp" TMP="$root/.runtime/tmp" TEMP="$root/.runtime/tmp"
export XDG_CACHE_HOME="$root/.runtime/cache" PIP_CACHE_DIR="$root/.runtime/cache/pip"
export MPLCONFIGDIR="$root/.runtime/cache/matplotlib" MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TDN_RESPONSE_PRECONDITIONER_SNAPSHOT="${BASH_SOURCE[0]}"
exec srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=none --export=ALL --unbuffered \
    "$root/.venv/bin/python" "$root/scripts/response_preconditioner_native.py" run "$@"
