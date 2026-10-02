#!/usr/bin/env bash
# Six tiny CPU controls, serial and bounded; no scheduler submission.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tdn_local_policy
if [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" ]]; then tdn_allocation; fi
tdn_prepare_env
[[ $# -eq 3 ]] || tdn_die 'Usage: compare_local.sh CONFIG DATASET NEW_RUN_DIR'
config="$(tdn_inside "$(realpath -m -- "$1")")"
dataset="$(tdn_inside "$(realpath -m -- "$2")")"
run="$(tdn_inside "$(realpath -m -- "$3")")"
[[ ! -e "$run" ]] || tdn_die 'Use a new run directory to preserve results'
cd "$TDN_REPO_ROOT"
exec "$(tdn_python)" -m tdn.cli compare --config "$config" --dataset "$dataset" --run-dir "$run" --device cpu
