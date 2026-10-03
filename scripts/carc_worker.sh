#!/usr/bin/env bash
# Called inside the phase's single srun task, never directly on the login node.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tdn_runtime_policy
tdn_allocation
tdn_prepare_env
# One checkout owns one shared venv. Hold this lock through every child stage
# so a concurrent CPU installer cannot change software under a GPU phase.
command -v flock >/dev/null || tdn_die 'CARC grouped workflow requires flock'
phase_lock_path="$(tdn_inside "$TDN_REPO_ROOT/.cache/carc-phase.lock")"
exec {phase_lock_fd}>"$phase_lock_path"
flock --nonblock "$phase_lock_fd" ||
    tdn_die 'Another CARC phase is using this checkout; wait for it to finish'
: "${TDN_WORKFLOW:?}" "${TDN_WORKFLOW_ROOT:?}" "${TDN_WORKFLOW_PHASE:?}"
: "${TDN_CONFIG:?}" "${TDN_DATASET:?}" "${TDN_SETUP_MODE:?}"
TDN_WORKFLOW_ACTION="${TDN_WORKFLOW_ACTION:-start}"
case "$TDN_WORKFLOW_ACTION" in
    setup) [[ "$TDN_WORKFLOW_PHASE" == cpu ]] || tdn_die 'Setup-only workflow requires a CPU phase' ;;
    light)
        [[ "$TDN_WORKFLOW_PHASE" == cpu ]] || tdn_die 'Light screening has no GPU phase'
        [[ "$TDN_SETUP_MODE" == never ]] || tdn_die 'Light screening requires setup=never and an existing verified venv' ;;
    start) ;;
    *) tdn_die 'Grouped workflow action must be setup, light or start' ;;
esac
case "$TDN_WORKFLOW_PHASE" in
    cpu) export TDN_DEVICE=cpu ;;
    gpu) export TDN_DEVICE=cuda ;;
    *) tdn_die 'Grouped worker phase must be cpu or gpu' ;;
esac
case "$TDN_SETUP_MODE" in
    auto|always|never) ;;
    *) tdn_die 'Setup mode must be auto, always or never' ;;
esac
if [[ "$TDN_WORKFLOW_ACTION" == start ]]; then
    [[ "${TDN_PILOT_BUDGET:-}" =~ ^[1-9][0-9]*$ && "$TDN_PILOT_BUDGET" -le 1000 ]] ||
        tdn_die 'Grouped development workflow requires a budget of 1-1000 optimizer steps'
fi
TDN_WORKFLOW="$(tdn_inside "$TDN_WORKFLOW")"
TDN_WORKFLOW_ROOT="$(tdn_inside "$TDN_WORKFLOW_ROOT")"
TDN_CONFIG="$(tdn_inside "$TDN_CONFIG")"
TDN_DATASET="$(tdn_inside "$TDN_DATASET")"
[[ "$TDN_WORKFLOW" == "$TDN_WORKFLOW_ROOT/workflow.json" &&
   "$TDN_CONFIG" == "$TDN_WORKFLOW_ROOT/config.yaml" &&
   "$TDN_DATASET" == "$TDN_WORKFLOW_ROOT/dataset" ]] ||
    tdn_die 'Grouped worker paths must match the immutable workflow layout'
export TDN_WORKFLOW TDN_WORKFLOW_ROOT TDN_CONFIG TDN_DATASET
export TDN_GATE_REPORT="$TDN_WORKFLOW_ROOT/audit/audit.json"
export TDN_CALIBRATION_REPORT="$TDN_WORKFLOW_ROOT/calibrate/calibration.json"
controller="$TDN_REPO_ROOT/scripts/carc_workflow.py"
cd "$TDN_REPO_ROOT"

workflow_command() {
    python3 "$controller" "$1" --workflow "$TDN_WORKFLOW" --phase "$TDN_WORKFLOW_PHASE" "${@:2}"
}
workflow_command worker-verify

# The coordinator survives Slurm's warning while Python saves at a safe boundary.
# Traps interrupt Bash wait, so repeat wait while the child is still alive.
child_pid=""
current_stage=""
stop_requested=false
wait_interrupted=false
request_stop() {
    stop_requested=true
    wait_interrupted=true
    if [[ -n "$child_pid" ]]; then
        kill -s "$1" "$child_pid" 2>/dev/null || true
    fi
}
trap 'request_stop USR1' USR1
trap 'request_stop TERM' TERM
trap 'request_stop TERM' INT

mark_stage() {
    workflow_command worker-mark --stage "$1" --status "$2" --exit-code "$3"
}
wait_for_child() {
    local code
    while true; do
        wait_interrupted=false
        if wait "$child_pid"; then code=0; else code=$?; fi
        # A trap may interrupt wait just as the child exits. Repeat even if the
        # PID is already gone; Bash retains its real status for the next wait.
        if [[ "$wait_interrupted" == true ]]; then
            continue
        fi
        return "$code"
    done
}
run_stage() {
    local stage="$1" code
    if [[ "$stop_requested" == true ]]; then
        printf 'TDN: stop requested before %s; no successor stage started.\n' "$stage" >&2
        exit 75
    fi
    workflow_command worker-verify
    if workflow_command worker-stage-complete --stage "$stage"; then
        printf 'TDN: preserved verified completed stage %s.\n' "$stage"
        return 0
    fi
    if [[ "$stop_requested" == true ]]; then
        printf 'TDN: stop requested before %s; no successor stage started.\n' "$stage" >&2
        exit 75
    fi
    export TDN_STAGE="$stage" TDN_RUN_DIR="$TDN_WORKFLOW_ROOT/$stage"
    case "$stage" in
        train) export TDN_RESUME="$train_resume" ;;
        evaluate|benchmark) export TDN_RESUME="$TDN_WORKFLOW_ROOT/train/checkpoints/best.pt" ;;
        *) export TDN_RESUME=none ;;
    esac
    tdn_mkdir "$TDN_RUN_DIR"
    mark_stage "$stage" RUNNING 0
    current_stage="$stage"
    if [[ "$stop_requested" == true ]]; then
        mark_stage "$stage" FAILED 75
        current_stage=""
        printf 'TDN: stop requested before %s could start.\n' "$stage" >&2
        exit 75
    fi
    printf '\nTDN: starting %s in %s job %s step %s.\n' \
        "$stage" "$TDN_WORKFLOW_PHASE" "$SLURM_JOB_ID" "$SLURM_STEP_ID"
    bash "$TDN_REPO_ROOT/scripts/run_stage.sh" &
    child_pid=$!
    if [[ "$stop_requested" == true ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
    fi
    if wait_for_child; then code=0; else code=$?; fi
    child_pid=""
    if [[ "$code" == 0 ]]; then
        mark_stage "$stage" COMPLETED 0
        current_stage=""
        printf 'TDN: completed %s.\n' "$stage"
    elif [[ "$code" == 75 ]]; then
        mark_stage "$stage" PAUSED_NEEDS_RESUME 75
        current_stage=""
        printf 'TDN: %s paused; preserve this workflow and resume explicitly.\n' "$stage" >&2
        exit 75
    else
        mark_stage "$stage" FAILED "$code"
        current_stage=""
        printf 'TDN: %s failed with exit %s; no successor stage started.\n' "$stage" "$code" >&2
        exit "$code"
    fi
}
on_exit() {
    local code=$?
    trap - EXIT
    if [[ -n "$current_stage" ]]; then
        if [[ -n "$child_pid" ]]; then
            kill -TERM "$child_pid" 2>/dev/null || true
            wait "$child_pid" 2>/dev/null || true
        fi
        mark_stage "$current_stage" FAILED "$code" || true
    fi
    exit "$code"
}
trap on_exit EXIT

train_resume="${TDN_RESUME:-none}"
if [[ "$train_resume" != none ]]; then train_resume="$(tdn_inside "$train_resume")"; fi
if [[ "$TDN_WORKFLOW_PHASE" == cpu ]]; then
    if [[ "$stop_requested" == true ]]; then exit 75; fi
    case "$TDN_SETUP_MODE" in
        always) run_stage setup ;;
        auto)
            if workflow_command worker-venv-ready; then
                mark_stage setup COMPLETED 0
                printf 'TDN: reused the verified project Python venv; no installation needed.\n'
            else
                run_stage setup
            fi ;;
        never)
            workflow_command worker-venv-ready ||
                tdn_die 'Requested setup=never but the exact project venv is not ready'
            mark_stage setup COMPLETED 0 ;;
    esac
fi
workflow_command worker-venv-ready ||
    tdn_die 'The project venv does not match this workflow after CPU setup'
if [[ "$TDN_WORKFLOW_ACTION" == setup ]]; then
    if [[ "$stop_requested" == true ]]; then exit 75; fi
    printf 'TDN: completed allocated CPU setup-only workflow.\n'
    exit 0
fi
python="$(tdn_python)"
if [[ "$TDN_WORKFLOW_ACTION" == light ]]; then
    "$python" - "$TDN_CONFIG" <<'PY'
import sys
from tdn.config import load_config
from tdn.analysis.light_screen import validate_light_config

validate_light_config(load_config(sys.argv[1]))
PY
    for stage in light-tests light-screen; do run_stage "$stage"; done
    if [[ "$stop_requested" == true ]]; then exit 75; fi
    printf 'TDN: completed bounded CPU-only light screening workflow.\n'
    exit 0
fi
"$python" - "$TDN_CONFIG" "$TDN_PILOT_BUDGET" <<'PY'
import sys
from tdn.config import load_config

config = load_config(sys.argv[1])
if config["purpose"] != "development" or config["runtime"]["confirmatory_authorized"]:
    raise SystemExit("Grouped workflow authorizes development diagnostics only")
if not 1 <= config["training"]["max_steps"] <= int(sys.argv[2]):
    raise SystemExit("Configuration exceeds the explicitly authorized optimizer budget")
grid = config["problem"]["grid"]
if len(grid) > 2 or any(n > 64 for n in grid):
    raise SystemExit("Grouped workflow is limited to small 1-D/2-D grids")
PY
if [[ "$TDN_WORKFLOW_PHASE" == cpu ]]; then
    for stage in cpu-tests audit generate; do run_stage "$stage"; done
else
    for stage in gpu-tests calibrate train evaluate benchmark; do run_stage "$stage"; done
fi
if [[ "$stop_requested" == true ]]; then
    printf 'TDN: phase finished after a stop request; check completed stage markers before resuming.\n' >&2
    exit 75
fi
printf 'TDN: completed grouped %s workflow phase.\n' "$TDN_WORKFLOW_PHASE"
