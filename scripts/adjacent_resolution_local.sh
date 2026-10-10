#!/usr/bin/env bash
# Actual 64/128 CPU plumbing only; native full uses fedora_adjacent_resolution.sh.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash scripts/adjacent_resolution_local.sh [--run-dir PROJECT_PATH]\nCPU-only resolution smoke; all full profiles require actual Fedora Slurm.\n'
    exit 0
fi
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir) [[ $# -ge 2 ]] || { printf '%s\n' '--run-dir needs a path' >&2; exit 2; }; args+=("$1" "$2"); shift 2 ;;
        *) printf 'Unsupported resolution-local option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
exec bash "$root/scripts/adjacent_local.sh" --resolution-smoke "${args[@]}"
