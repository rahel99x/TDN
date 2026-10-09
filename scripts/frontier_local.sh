#!/usr/bin/env bash
# Project-contained CPU correctness smoke/development; fresh full cohort stays allocated.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash scripts/frontier_local.sh [--smoke|--development] [--run-dir PROJECT_PATH] [--recover ORIGIN_RUN]\nDefault: smoke. Bounded parts run sequentially on CPU; this cannot validate CUDA or fresh confirmation.\n'
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use an allocated workflow on CARC; local computation cannot run on its login node'
[[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] ||
    tdn_die 'Local frontier requires a separate CPU checkout without Slurm or desktop overrides'
profile=smoke
run_dir=""
recovery_origin=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --development) profile=development; shift ;;
        --smoke) profile=smoke; shift ;;
        --full) tdn_die 'Fresh full confirmation requires the allocated Fedora workflow' ;;
        --run-dir) [[ $# -ge 2 ]] || tdn_die '--run-dir needs a path'; run_dir="$2"; shift 2 ;;
        --recover) [[ $# -ge 2 ]] || tdn_die '--recover needs an origin run'; recovery_origin="$2"; shift 2 ;;
        *) tdn_die "Unknown option: $1" ;;
    esac
done
export TDN_LOCAL_TEST_ROOT="$TDN_REPO_ROOT"
unset TDN_ROADMAP_WORKFLOW TDN_ROADMAP_STAGE TDN_ROADMAP_PROTOCOL_SHA256 TDN_ROADMAP_CPUS
unset TDN_AGENDA_WORKFLOW TDN_AGENDA_PROTOCOL_SHA256 TDN_AGENDA_STAGE TDN_AGENDA_CPUS
unset TDN_TOWER_DIR TDN_FRONTIER_WORKFLOW TDN_FRONTIER_PROTOCOL_SHA256 TDN_FRONTIER_STAGE TDN_FRONTIER_CPUS
unset TDN_CONSISTENCY_WORKFLOW TDN_CONSISTENCY_PROTOCOL_SHA256 TDN_CONSISTENCY_STAGE
unset TDN_PREMIX_WORKFLOW TDN_PREMIX_PROTOCOL_SHA256 TDN_PREMIX_STAGE
tdn_prepare_env
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
if [[ -z "$run_dir" ]]; then
    run_dir="$TDN_REPO_ROOT/runs/frontier-local-$(date -u +%Y%m%dT%H%M%S%N)"
fi
[[ "$run_dir" == /* ]] || run_dir="$TDN_REPO_ROOT/$run_dir"
run_dir="$(tdn_inside "$run_dir")"
[[ ! -e "$run_dir" ]] || tdn_die 'Run path exists; preserve it and choose a fresh directory'
exec 9<> "$TDN_REPO_ROOT/.cache/carc-phase.lock"
flock -s -n 9 || tdn_die 'Another workflow is modifying the project venv; preserve it and retry after setup finishes'
mkdir -p "$run_dir"
recovery_args=()
if [[ -n "$recovery_origin" ]]; then
    "$python" - "$recovery_origin" "$run_dir/recovery.json" "$profile" <<'PY'
import sys
from pathlib import Path
from tdn.analysis.frontier.protocol import build_protocol
from tdn.analysis.frontier.recovery import create_recovery
from tdn.runtime.metadata import software_metadata, write_json
bridge = create_recovery(Path(sys.argv[1]), build_protocol(sys.argv[3]), software_metadata(), None)
write_json(Path(sys.argv[2]), bridge)
PY
    recovery_args=(--recovery-manifest "$run_dir/recovery.json")
fi
prior=()
failed=0
for stage in audit screen prepare train confirm_prepare confirm scaling policy report; do
    extra=("${prior[@]}")
    stage_dir="$run_dir/$stage"
    if [[ -n "$recovery_origin" && "$stage" =~ ^(audit|screen|prepare|train|confirm_prepare)$ ]]; then
        stage_dir="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_paths"][sys.argv[2]])' "$run_dir/recovery.json" "$stage")"
        printf 'Reusing sealed %s: %s\n' "$stage" "$stage_dir"
        prior+=(--prerequisite-dir "$stage=$stage_dir")
        continue
    fi
    if [[ "$stage" == confirm && "$failed" == 0 ]]; then
        mapfile -t parts < <("$python" -c 'import sys; from tdn.analysis.frontier.protocol import build_protocol; from tdn.analysis.frontier.partition import plan_shards; print("\n".join(p["shard_id"] for p in plan_shards(build_protocol(sys.argv[1]))))' "$profile")
        [[ ${#parts[@]} -gt 0 ]] || tdn_die 'Missing confirmation execution plan'
        for part in "${parts[@]}"; do
            inherited_part=""
            if [[ -n "$recovery_origin" ]]; then
                inherited_part="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_paths"].get(sys.argv[2], ""))' "$run_dir/recovery.json" "$part")"
            fi
            if [[ -n "$inherited_part" ]]; then
                printf 'Reusing sealed %s: %s\n' "$part" "$inherited_part"
                extra+=(--confirm-shard-dir "$part=$inherited_part")
                continue
            fi
            "$python" "$TDN_REPO_ROOT/scripts/frontier.py" --stage confirm --profile "$profile" \
                --run-dir "$run_dir/$part" --confirm-shard "$part" --device cpu --local-root "$TDN_REPO_ROOT" \
                "${prior[@]}" "${recovery_args[@]}" || failed=1
            extra+=(--confirm-shard-dir "$part=$run_dir/$part")
            [[ "$failed" == 0 ]] || break
        done
    fi
    if [[ "$failed" == 0 || "$stage" == report ]]; then
        "$python" "$TDN_REPO_ROOT/scripts/frontier.py" --stage "$stage" --profile "$profile" \
            --run-dir "$stage_dir" --device cpu --local-root "$TDN_REPO_ROOT" "${extra[@]}" "${recovery_args[@]}" || failed=1
    fi
    prior+=(--prerequisite-dir "$stage=$stage_dir")
done
printf 'Local CPU frontier complete (runtime failure=%s): %s\n' "$failed" "$run_dir"
exit "$failed"
