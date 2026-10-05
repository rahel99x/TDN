#!/usr/bin/env bash
# Non-CARC CPU development in the existing project Python venv.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/neural_replication_local.sh [--smoke] [--run-dir PROJECT_PATH] [--config PROJECT_PATH]

Run configs/neural-replication.yaml in the existing project venv on local CPU.
Uses research_local.sh storage and runtime checks; refuses CARC login-node use.
HELP
    exit 0
fi
custom_config=false
for argument in "$@"; do
    if [[ "$argument" == --config || "$argument" == --config=* ]]; then
        custom_config=true
    fi
done
if [[ "$custom_config" == false ]]; then
    set -- --config "$script_dir/../configs/neural-replication.yaml" "$@"
fi
exec bash "$script_dir/research_local.sh" "$@"
