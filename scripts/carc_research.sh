#!/usr/bin/env bash
# Login-node controller only; all numerical work belongs to an allocated task.
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
    [[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] ||
        tdn_die 'Submit research from the CARC login shell with local/desktop overrides cleared'
    tdn_prepare_env
fi
if [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" && "${1:-}" != --help ]]; then
    tdn_load_python_module
fi
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$TDN_REPO_ROOT/scripts/research_workflow.py" "$@"
