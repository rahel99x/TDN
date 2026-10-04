#!/usr/bin/env bash
# Explicit non-CARC CPU development through the existing project venv wrapper.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/neural_benchmarks_local.sh [--smoke] [--run-dir PROJECT_PATH] [--config PROJECT_PATH]

Run configs/neural-benchmarks.yaml in the existing project venv on local CPU.
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
    set -- --config "$script_dir/../configs/neural-benchmarks.yaml" "$@"
fi
exec bash "$script_dir/research_local.sh" "$@"
