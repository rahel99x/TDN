#!/usr/bin/env bash
# Sequential local CPU implementation check. Never fabricate Slurm/GPU variables.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tdn_local_policy
if [[ "$TDN_REPO_ROOT" == "$TDN_CARC_ROOT" ]]; then tdn_allocation; fi
tdn_prepare_env
python="$(tdn_python)"
config="$(tdn_inside "$(realpath -m -- "${1:-$TDN_REPO_ROOT/configs/smoke.yaml}")")"
run="$(tdn_inside "$(realpath -m -- "${2:-$TDN_REPO_ROOT/runs/local-smoke-$(date -u +%Y%m%dT%H%M%SZ)}")")"
[[ ! -e "$run" ]] || tdn_die 'Smoke run already exists; choose a fresh output path'
tdn_mkdir "$run"
cd "$TDN_REPO_ROOT"
export TDN_GATE_REPORT="$run/audit/audit.json"
"$python" -m pytest -q -m 'not gpu' --basetemp "$run/pytest-work"
for stage in audit generate train evaluate benchmark; do
    args=("$python" -u -m tdn.cli "$stage" --config "$config" --run-dir "$run/$stage"
          --dataset "$run/dataset" --device cpu --resume none)
    if [[ "$stage" == evaluate || "$stage" == benchmark ]]; then
        checkpoint="$run/train/checkpoints/best.pt"
        if [[ ! -f "$checkpoint" ]]; then
            checkpoint="$run/train/checkpoints/last.pt"
            printf 'Implementation smoke uses last checkpoint; no feasibility claim is implied.\n'
        fi
        args=("$python" -u -m tdn.cli "$stage" --config "$config" --run-dir "$run/$stage"
              --dataset "$run/dataset" --device cpu --resume "$checkpoint")
    fi
    "${args[@]}"
done
printf 'Sequential CPU smoke completed: %s\n' "$run"
