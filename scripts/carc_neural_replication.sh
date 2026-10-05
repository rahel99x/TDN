#!/usr/bin/env bash
# Paired three-seed CPU study; existing controller owns CARC policy.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/carc_neural_replication.sh [start] [--submit] [--smoke] [--run-id ID] [--config PATH]
       bash scripts/carc_neural_replication.sh status|logs|collect [RUN_ID|latest]
       bash scripts/carc_neural_replication.sh benchmark CPU_RUN_ID [--submit] [--run-id ID]

Defaults to a preview with configs/neural-replication.yaml. Add --submit explicitly.
One bounded CPU job trains paired seeds and evaluates fresh diagnostic parents.
Optional A100 inference inherits sealed CPU config and frozen checkpoints.
All commands delegate to carc_research.sh; latest is the shared research pointer.
HELP
    exit 0
fi
if [[ $# == 0 ]]; then set -- start; fi
if [[ "$1" == --* ]]; then set -- start "$@"; fi
if [[ "$1" == start ]]; then
    custom_config=false
    replication_marker=false
    for argument in "$@"; do
        if [[ "$argument" == --config || "$argument" == --config=* ]]; then
            custom_config=true
        fi
        if [[ "$argument" == --neural-replication ]]; then replication_marker=true; fi
    done
    if [[ "$custom_config" == false ]]; then
        shift
        set -- start --config "$script_dir/../configs/neural-replication.yaml" "$@"
    fi
    if [[ "$replication_marker" == false ]]; then
        shift
        set -- start --neural-replication "$@"
    fi
fi
exec bash "$script_dir/carc_research.sh" "$@"
