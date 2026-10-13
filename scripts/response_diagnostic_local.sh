#!/usr/bin/env bash
# Reduced CPU implementation smoke; never substitutes for native calibration.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash scripts/response_diagnostic_local.sh [--smoke] [--run-dir PROJECT_PATH] [--targets TARGET ...]\nReduced CPU diagnostic smoke only; native diagnostic work requires Fedora Slurm.\n'
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use a real allocation on CARC'
[[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] || tdn_die 'Local CPU diagnostic requires a separate checkout without Slurm overrides'
run_dir=""
targets=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke|--diagnostic-smoke) shift ;;
        --run-dir) [[ $# -ge 2 ]] || tdn_die '--run-dir needs a path'; run_dir="$2"; shift 2 ;;
        --targets)
            shift; count=0
            while [[ $# -gt 0 && "$1" != --* ]]; do targets+=("$1"); shift; count=$((count+1)); done
            [[ "$count" -gt 0 ]] || tdn_die '--targets needs at least one target' ;;
        --diagnostic|--pilot|--development|--confirmation|--full)
            tdn_die 'Full diagnostic and native calibration require allocated Fedora Slurm' ;;
        *) tdn_die "Unknown option: $1" ;;
    esac
done
export TDN_LOCAL_TEST_ROOT="$TDN_REPO_ROOT"
unset TDN_TOWER_DIR TDN_FRONTIER_WORKFLOW TDN_ROADMAP_WORKFLOW TDN_AGENDA_WORKFLOW
unset TDN_PREMIX_WORKFLOW TDN_CONSISTENCY_WORKFLOW TDN_PORTFOLIO_WORKFLOW
unset TDN_ADVANCE_WORKFLOW TDN_ADVANCE_STAGE TDN_ADVANCE_PROTOCOL_SHA256
unset TDN_RESOLVE_WORKFLOW TDN_RESOLVE_STAGE TDN_RESOLVE_PROTOCOL_SHA256
unset TDN_RESPONSE_WORKFLOW TDN_RESPONSE_STAGE TDN_RESPONSE_PROTOCOL_SHA256 TDN_RESPONSE_POINTER
unset TDN_REQUIRE_GPU_TESTS TDN_REQUIRE_PORTFOLIO_GPU_TESTS TDN_REQUIRE_ADVANCE_GPU_TESTS
unset TDN_REQUIRE_RESOLVE_GPU_TESTS TDN_REQUIRE_RESPONSE_GPU_TESTS
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
tdn_prepare_env
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
if [[ -z "$run_dir" ]]; then run_dir="$TDN_REPO_ROOT/runs/response-diagnostic-local-$(date -u +%Y%m%dT%H%M%S%N)"; fi
[[ "$run_dir" == /* ]] || run_dir="$TDN_REPO_ROOT/$run_dir"
run_dir="$(tdn_inside "$run_dir")"
[[ ! -e "$run_dir" ]] || tdn_die 'Run path exists; preserve it and choose a fresh directory'
exec 9<> "$TDN_REPO_ROOT/.cache/carc-phase.lock"
flock -s -n 9 || tdn_die 'Project venv is being modified; retry after setup'
args=(--run-dir "$run_dir")
if [[ ${#targets[@]} -gt 0 ]]; then args+=(--targets "${targets[@]}"); fi
exec "$python" "$TDN_REPO_ROOT/scripts/response_diagnostic_cpu.py" "${args[@]}"
