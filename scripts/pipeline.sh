#!/usr/bin/env bash
# Serial afterok DAG. Default is a readiness dry-run, not a research campaign.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 1 ]] || tdn_die 'Usage: pipeline.sh CONFIG [--submit] [--pilot-budget STEPS] [--pipeline-id NAME]'
config="$1"; shift
submit=false; budget=""; pipeline_id="readiness-$(date -u +%Y%m%dT%H%M%SZ)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) submit=true; shift ;;
        --dry-run) submit=false; shift ;;
        --pilot-budget|--pipeline-id)
            [[ $# -ge 2 ]] || tdn_die "Missing value for $1"
            if [[ "$1" == --pilot-budget ]]; then budget="$2"; else pipeline_id="$2"; fi
            shift 2 ;;
        *) tdn_die "Unknown option $1" ;;
    esac
done
[[ "$pipeline_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || tdn_die 'Use a simple unique pipeline identifier'
if [[ -n "$budget" ]]; then
    [[ "$budget" =~ ^[1-9][0-9]*$ && "$budget" -le 1000 ]] || tdn_die 'Pilot budget must be 1-1000 optimizer steps'
fi
config="$(tdn_inside "$(realpath -m -- "$config")")"
base="$(tdn_inside "$TDN_REPO_ROOT/runs/$pipeline_id")"
dataset="$(tdn_inside "$base/dataset")"
[[ ! -e "$base" ]] || tdn_die 'Pipeline directory already exists; preserve it and choose a new identifier'
stages=(cpu-tests audit generate gpu-tests calibrate)
if [[ -n "$budget" ]]; then stages+=(train evaluate benchmark); fi
printf 'Sequential DAG: '
printf '%s ' "${stages[@]}"
printf '\nEach stage depends on successful completion of the preceding stage.\n'
printf 'Paused (exit 75), failed scientific gates and failed GPU checks stop afterok successors.\n'
if [[ "$submit" == true ]]; then
    tdn_runtime_policy
    [[ -x "$TDN_REPO_ROOT/.venv/bin/python" ]] || tdn_die 'Complete the allocated setup stage first'
    if [[ -n "$budget" ]]; then
        python="$(tdn_python)"
        "$python" - "$config" "$budget" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
if config['purpose'] != 'development' or config['runtime']['confirmatory_authorized']:
    raise SystemExit('Only a development pilot is allowed by this wrapper')
steps = config['training']['max_steps']
if not isinstance(steps, int) or not 1 <= steps <= int(sys.argv[2]):
    raise SystemExit('Config optimizer steps exceed the explicit pilot budget; no jobs submitted')
grid = config['problem']['grid']
if len(grid) > 2 or any(n > 64 for n in grid):
    raise SystemExit('The initial bounded pilot is limited to small 1-D/2-D grids')
PY
    fi
fi
previous=""; previous_stage=""
for stage in "${stages[@]}"; do
    args=(bash "$TDN_REPO_ROOT/scripts/submit.sh" "$stage" "$config"
          --run-dir "$base/$stage" --dataset "$dataset" --gate-report "$base/audit/audit.json")
    if [[ "$stage" == train ]]; then args+=(--calibration-report "$base/calibrate/calibration.json"); fi
    if [[ "$stage" == train ]]; then args+=(--pilot-budget "$budget"); fi
    if [[ "$stage" == evaluate || "$stage" == benchmark ]]; then
        args+=(--resume "$base/train/checkpoints/best.pt")
    fi
    if [[ "$submit" == true ]]; then
        args+=(--submit)
        [[ -n "$previous" ]] && args+=(--dependency "afterok:$previous")
        output="$("${args[@]}")"
        printf '%s\n' "$output"
        job_line="${output##*$'\n'}"
        [[ "$job_line" =~ ^TDN_JOB_ID=([1-9][0-9]*)$ ]] || tdn_die 'Cannot safely chain a job without its actual scheduler ID'
        previous="${BASH_REMATCH[1]}"
    else
        [[ -n "$previous_stage" ]] && printf 'Planned dependency for %s: afterok:<actual-%s-job-id>\n' "$stage" "$previous_stage"
        "${args[@]}" --dry-run
    fi
    previous_stage="$stage"
done
