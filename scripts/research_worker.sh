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
command -v flock >/dev/null || tdn_die 'Research worker requires flock'
# Shared readers may run concurrently; existing CARC installers take this same
# lock exclusively. This protects the venv without limiting pending jobs.
lock_path="$(tdn_inside "$TDN_REPO_ROOT/.cache/carc-phase.lock")"
exec {research_lock_fd}>"$lock_path"
flock --shared --nonblock "$research_lock_fd" ||
    tdn_die 'Another workflow is modifying/using the venv exclusively; preserve this run and submit a fresh one later'
unset TDN_REQUIRE_GPU_TESTS TDN_GATE_REPORT TDN_CALIBRATION_REPORT TDN_DESKTOP_CUDA_DEVICE
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
exec "$python" "$TDN_REPO_ROOT/scripts/research_workflow.py" worker \
    --workflow "$TDN_RESEARCH_WORKFLOW" --phase "$TDN_RESEARCH_PHASE"
