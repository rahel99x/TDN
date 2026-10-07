#!/usr/bin/env bash
# Project-contained sequential CPU integration for the consistency experiment.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash scripts/consistency_local.sh [--full] [--run-dir PROJECT_PATH]\nDefault: smoke profile; audit, prepare, neural sequentially on CPU.\n'
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use an allocated workflow on CARC; local computation cannot run on its login node'
[[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] ||
    tdn_die 'Local consistency requires a separate CPU checkout without Slurm or desktop overrides'
profile=smoke
run_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full) profile=full; shift ;;
        --smoke) profile=smoke; shift ;;
        --run-dir) [[ $# -ge 2 ]] || tdn_die '--run-dir needs a path'; run_dir="$2"; shift 2 ;;
        *) tdn_die "Unknown option: $1" ;;
    esac
done
export TDN_LOCAL_TEST_ROOT="$TDN_REPO_ROOT"
unset TDN_TOWER_DIR TDN_CONSISTENCY_WORKFLOW TDN_CONSISTENCY_PROTOCOL_SHA256 TDN_CONSISTENCY_STAGE
tdn_prepare_env
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
if [[ -z "$run_dir" ]]; then
    run_dir="$TDN_REPO_ROOT/runs/consistency-local-$(date -u +%Y%m%dT%H%M%S%N)"
fi
[[ "$run_dir" == /* ]] || run_dir="$TDN_REPO_ROOT/$run_dir"
run_dir="$(tdn_inside "$run_dir")"
[[ ! -e "$run_dir" ]] || tdn_die 'Run path exists; preserve it and choose a fresh directory'
# The shared readable lock prevents setup from replacing a venv in active use.
exec 9<> "$TDN_REPO_ROOT/.cache/carc-phase.lock"
flock -s -n 9 || tdn_die 'Another workflow is modifying the project venv; preserve it and retry after setup finishes'
mkdir -p "$run_dir"
for stage in audit prepare neural; do
    extra=()
    if [[ "$stage" != audit ]]; then extra+=(--audit-dir "$run_dir/audit"); fi
    if [[ "$stage" == neural ]]; then extra+=(--dataset-dir "$run_dir/prepare"); fi
    "$python" "$TDN_REPO_ROOT/scripts/consistency.py" --stage "$stage" --profile "$profile" \
        --run-dir "$run_dir/$stage" --device cpu --local-root "$TDN_REPO_ROOT" "${extra[@]}"
done
printf 'Local CPU consistency complete: %s\n' "$run_dir"
