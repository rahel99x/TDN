#!/usr/bin/env bash
# One login-node entrypoint; substantial setup and numerics stay in allocations.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$TDN_REPO_ROOT/scripts/carc_defaults.sh"

if [[ $# == 0 ]]; then set -- start; fi
if [[ "${1:-}" == --* && "${1:-}" != --help ]]; then set -- start "$@"; fi
submit=false
dry_run=false
for argument in "$@"; do
    if [[ "$argument" == --submit ]]; then submit=true; fi
    if [[ "$argument" == --dry-run ]]; then dry_run=true; fi
done
if [[ "$dry_run" == true ]]; then submit=false; fi
if [[ "$submit" == true ]]; then
    tdn_runtime_policy
    [[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}" ]] ||
        tdn_die 'Submit the CARC workflow from the login shell, outside an allocation'
    [[ -z "${TDN_EXECUTION_MODE:-}" ]] ||
        tdn_die 'Unset TDN_EXECUTION_MODE before using the CARC workflow'
    tdn_prepare_env
    # Each allocated stage owns its test/report flags. An old interactive GPU
    # test flag must not turn the CPU prerequisite suite into a GPU request.
    unset TDN_REQUIRE_GPU_TESTS TDN_GATE_REPORT TDN_CALIBRATION_REPORT TDN_PROJECT_ROOT
    unset TDN_DESKTOP_CUDA_DEVICE
fi
# The default module is already observed on CARC. Dry runs elsewhere need only
# their existing standalone Python; they never install or request a GPU.
if [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" && "${1:-}" != --help ]]; then
    tdn_load_python_module
fi
python3 -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11 or newer is required"'
exec python3 "$TDN_REPO_ROOT/scripts/carc_workflow.py" "$@"
