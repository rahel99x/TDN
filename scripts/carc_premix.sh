#!/usr/bin/env bash
# Login controller: run explicitly submits the fixed bounded four-job DAG.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
source "$TDN_REPO_ROOT/scripts/carc_defaults.sh"
if [[ $# == 0 ]]; then set -- plan; fi
if [[ "${1:-}" == run ]]; then
    tdn_runtime_policy
    [[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] ||
        tdn_die 'Submit premix from the CARC login shell with local/desktop overrides cleared'
    tdn_prepare_env
    tdn_load_python_module
fi
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$TDN_REPO_ROOT/scripts/premix_workflow.py" "$@"
