#!/usr/bin/env bash
# Explicit cloud/local CPU diagnostic in the existing project Python venv.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
if [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" ]]; then
    tdn_die 'Use bash scripts/carc_research.sh --submit on CARC; local research must not run on its login node'
fi
[[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}" ]] ||
    tdn_die 'Local research requires a separate CPU checkout without Slurm or desktop overrides'
export TDN_LOCAL_TEST_ROOT="$TDN_REPO_ROOT"
tdn_prepare_env
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
smoke=()
run_dir=""
config="$TDN_REPO_ROOT/configs/research.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) smoke=(--smoke); shift ;;
        --run-dir) [[ $# -ge 2 ]] || tdn_die '--run-dir needs a path'; run_dir="$2"; shift 2 ;;
        --config) [[ $# -ge 2 ]] || tdn_die '--config needs a path'; config="$2"; shift 2 ;;
        --help|-h)
            printf 'Usage: bash scripts/research_local.sh [--smoke] [--run-dir PROJECT_PATH] [--config PROJECT_PATH]\n'
            exit 0 ;;
        *) tdn_die "Unknown option: $1" ;;
    esac
done
if [[ -z "$run_dir" ]]; then
    run_dir="$TDN_REPO_ROOT/runs/research-local-$(date -u +%Y%m%dT%H%M%S%N)"
fi
[[ "$run_dir" == /* ]] || run_dir="$TDN_REPO_ROOT/$run_dir"
[[ "$config" == /* ]] || config="$TDN_REPO_ROOT/$config"
run_dir="$(tdn_inside "$run_dir")"
config="$(tdn_inside "$config")"
[[ ! -e "$run_dir" ]] || tdn_die 'Run path already exists; preserve it and choose a fresh path'
exec "$python" "$TDN_REPO_ROOT/scripts/research.py" run \
    --config "$config" --run-dir "$run_dir" --device cpu "${smoke[@]}"
