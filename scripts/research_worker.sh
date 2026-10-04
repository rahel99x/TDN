#!/usr/bin/env bash
# Allocated worker. The existing shared venv is verified, never installed here.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tdn_runtime_policy
tdn_allocation
tdn_prepare_env
[[ -z "${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] || tdn_die 'Clear desktop/local overrides'
: "${TDN_RESEARCH_WORKFLOW:?}" "${TDN_RESEARCH_PHASE:?}"
case "$TDN_RESEARCH_PHASE" in cpu|gpu) ;; *) tdn_die 'Unknown research phase' ;; esac
unset TDN_REQUIRE_GPU_TESTS TDN_GATE_REPORT TDN_CALIBRATION_REPORT TDN_DESKTOP_CUDA_DEVICE
cd "$TDN_REPO_ROOT"
# The stdlib coordinator starts reporting before locking/verifying the venv.
# It holds the shared lock while every numerical child runs in project .venv.
exec python3 "$TDN_REPO_ROOT/scripts/research_workflow.py" worker \
    --workflow "$TDN_RESEARCH_WORKFLOW" --phase "$TDN_RESEARCH_PHASE"
