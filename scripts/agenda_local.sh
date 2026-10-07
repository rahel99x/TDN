#!/usr/bin/env bash
# Project-contained CPU correctness smoke/development; fresh full cohort stays allocated.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash scripts/agenda_local.sh [--smoke|--development] [--run-dir PROJECT_PATH]\nDefault: smoke. All eight stages run sequentially on CPU; this cannot validate CUDA or fresh confirmation.\n'
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use an allocated workflow on CARC; local computation cannot run on its login node'
[[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] ||
    tdn_die 'Local agenda requires a separate CPU checkout without Slurm or desktop overrides'
profile=smoke
run_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --development) profile=development; shift ;;
        --smoke) profile=smoke; shift ;;
        --full) tdn_die 'Fresh full confirmation requires the allocated Fedora workflow' ;;
        --run-dir) [[ $# -ge 2 ]] || tdn_die '--run-dir needs a path'; run_dir="$2"; shift 2 ;;
        *) tdn_die "Unknown option: $1" ;;
    esac
done
export TDN_LOCAL_TEST_ROOT="$TDN_REPO_ROOT"
unset TDN_TOWER_DIR TDN_AGENDA_WORKFLOW TDN_AGENDA_PROTOCOL_SHA256 TDN_AGENDA_STAGE TDN_AGENDA_CPUS
unset TDN_CONSISTENCY_WORKFLOW TDN_CONSISTENCY_PROTOCOL_SHA256 TDN_CONSISTENCY_STAGE
unset TDN_PREMIX_WORKFLOW TDN_PREMIX_PROTOCOL_SHA256 TDN_PREMIX_STAGE
tdn_prepare_env
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
if [[ -z "$run_dir" ]]; then
    run_dir="$TDN_REPO_ROOT/runs/agenda-local-$(date -u +%Y%m%dT%H%M%S%N)"
fi
[[ "$run_dir" == /* ]] || run_dir="$TDN_REPO_ROOT/$run_dir"
run_dir="$(tdn_inside "$run_dir")"
[[ ! -e "$run_dir" ]] || tdn_die 'Run path exists; preserve it and choose a fresh directory'
exec 9<> "$TDN_REPO_ROOT/.cache/carc-phase.lock"
flock -s -n 9 || tdn_die 'Another workflow is modifying the project venv; preserve it and retry after setup finishes'
mkdir -p "$run_dir"
prior=()
for stage in structure prepare controls optimize compression kernel confirm policy; do
    extra=("${prior[@]}")
    case "$stage" in structure|prepare) ;; *) extra+=(--dataset-dir "$run_dir/prepare") ;; esac
    "$python" "$TDN_REPO_ROOT/scripts/agenda.py" --stage "$stage" --profile "$profile" \
        --run-dir "$run_dir/$stage" --device cpu --local-root "$TDN_REPO_ROOT" "${extra[@]}"
    prior+=(--prerequisite-dir "$stage=$run_dir/$stage")
done
printf 'Local CPU agenda complete: %s\n' "$run_dir"
