#!/usr/bin/env bash
#SBATCH --job-name=tdn-response-preconditioner-r7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=none
#SBATCH --output=logs/response-preconditioner-r7/%x-%j.out
#SBATCH --error=logs/response-preconditioner-r7/%x-%j.err
# One bounded R7 development study, following mandatory implementation checks.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: raven launch scripts/response_preconditioner_r7.sh \
  --project "$PWD" --recipe response-preconditioner-r7 \
  --sbatch=--partition=local --sbatch="--chdir=$PWD" \
  -- --contract PATH [--mode all|verify]
Prepare PATH first using this checkout's standalone venv:
  .venv/bin/python scripts/response_preconditioner_r7_native.py prepare
The default runs implementation verification, then the frozen bounded R7 study.
Scientific acquisition requires actual native Raven/Slurm; omit --local.
No historical scientific gate or GPU readiness is promoted by completion.
HELP
    exit 0
fi
[[ -n "${RAVEN_LAUNCH_PLAN:-}" && -n "${RAVEN_RUN_DIR:-}" ]] || {
    printf 'TDN R7: use raven launch for this native entry point.\n' >&2; exit 2;
}
[[ "${RAVEN_SOURCE_SCRIPT:-}" == /* && -f "$RAVEN_SOURCE_SCRIPT" ]] || {
    printf 'TDN R7: Raven original source script is missing.\n' >&2; exit 2;
}
[[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'TDN R7: an actual Slurm allocation is required; omit Raven --local.\n' >&2; exit 2;
}
root="$(cd -- "$(dirname -- "$RAVEN_SOURCE_SCRIPT")/.." && pwd -P)"
[[ "$RAVEN_SOURCE_SCRIPT" == "$root/scripts/response_preconditioner_r7.sh" && "$root" != /home1/aadaniel/projects/TDN ]] || exit 2
[[ -x "$root/.venv/bin/python" ]] || {
    printf 'TDN R7: use this checkout\047s standalone .venv.\n' >&2; exit 2;
}
cd -- "$root"
export TDN_PROJECT_ROOT="$root" TDN_REPO_ROOT="$root" TDN_EXECUTION_MODE=desktop-slurm
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="$root/.runtime/tmp" TMP="$root/.runtime/tmp" TEMP="$root/.runtime/tmp"
export XDG_CACHE_HOME="$root/.runtime/cache" PIP_CACHE_DIR="$root/.runtime/cache/pip"
export MPLCONFIGDIR="$root/.runtime/cache/matplotlib" MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TDN_RESPONSE_PRECONDITIONER_R7_SNAPSHOT="${BASH_SOURCE[0]}"
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
exec srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=none --export=ALL --unbuffered \
    "$root/.venv/bin/python" "$root/scripts/response_preconditioner_r7_native.py" run "$@"
