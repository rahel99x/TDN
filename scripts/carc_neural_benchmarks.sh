#!/usr/bin/env bash
# Thin login-node interface: existing research controller owns CARC policy.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/carc_neural_benchmarks.sh [start] [--submit] [--smoke] [--run-id ID] [--config PATH]
       bash scripts/carc_neural_benchmarks.sh status|logs|collect [RUN_ID|latest]
       bash scripts/carc_neural_benchmarks.sh benchmark CPU_RUN_ID [--submit] [--run-id ID]

Defaults to a preview with configs/neural-benchmarks.yaml. Add --submit explicitly.
All commands delegate to carc_research.sh; latest is the shared research pointer.
Benchmark inherits the immutable CPU config and uses frozen checkpoints.
HELP
    exit 0
fi
if [[ $# == 0 ]]; then set -- start; fi
if [[ "$1" == --* ]]; then set -- start "$@"; fi
if [[ "$1" == start ]]; then
    custom_config=false
    neural_marker=false
    for argument in "$@"; do
        if [[ "$argument" == --config || "$argument" == --config=* ]]; then
            custom_config=true
        fi
        if [[ "$argument" == --neural-benchmarks ]]; then neural_marker=true; fi
    done
    if [[ "$custom_config" == false ]]; then
        shift
        set -- start --config "$script_dir/../configs/neural-benchmarks.yaml" "$@"
    fi
    if [[ "$neural_marker" == false ]]; then
        shift
        set -- start --neural-benchmarks "$@"
    fi
fi
exec bash "$script_dir/carc_research.sh" "$@"
