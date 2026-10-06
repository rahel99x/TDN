#!/usr/bin/env bash
# CPU-only, training-free work-precision experiments; shared controller owns CARC policy.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/carc_work_precision.sh [start] [--submit] [--smoke] [--run-id ID] [--config PATH]
       bash scripts/carc_work_precision.sh status|logs|collect [RUN_ID|latest]

Defaults to a read-only preview with configs/work-precision.yaml.
Add --submit explicitly for one CPU-only job: 4 CPUs, 16 GiB, 30 minutes.
Numerical experiments have a 1,200-second cap. No neural training or GPU successor.
Uses the existing verified project venv, account and allocation checks.
Status/logs/collect delegate to carc_research.sh; latest is the shared research pointer.
HELP
    exit 0
fi
if [[ $# == 0 ]]; then set -- start; fi
if [[ "$1" == --* ]]; then set -- start "$@"; fi
if [[ "$1" == benchmark ]]; then
    printf '%s\n' 'TDN work-precision: CPU-only experiments have no GPU benchmark successor.' >&2
    exit 2
fi
if [[ "$1" == start ]]; then
    custom_config=false
    precision_marker=false
    for argument in "$@"; do
        if [[ "$argument" == --config || "$argument" == --config=* ]]; then
            custom_config=true
        fi
        if [[ "$argument" == --work-precision ]]; then precision_marker=true; fi
    done
    if [[ "$custom_config" == false ]]; then
        shift
        set -- start --config "$script_dir/../configs/work-precision.yaml" "$@"
    fi
    if [[ "$precision_marker" == false ]]; then
        shift
        set -- start --work-precision "$@"
    fi
fi
exec bash "$script_dir/carc_research.sh" "$@"
