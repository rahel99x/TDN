#!/usr/bin/env bash
# One foreground coordinator: fresh native calibration, diagnostic, sealed review.
# Interrupting this coordinator preserves submitted jobs and its receipt.
set -euo pipefail
source_entry="${BASH_SOURCE[0]}"
if [[ -n "${RAVEN_LAUNCH_PLAN:-}" ]]; then
    [[ "${RAVEN_SOURCE_SCRIPT:-}" == /* && -f "$RAVEN_SOURCE_SCRIPT" ]] || {
        printf 'TDN diagnostic: Raven must provide its original source script path.\n' >&2; exit 2;
    }
    source_entry="$RAVEN_SOURCE_SCRIPT"
fi
root="$(cd -- "$(dirname -- "$source_entry")/.." && pwd -P)"
launcher="$root/scripts/fedora_response_diagnostic.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash scripts/response_diagnostic_campaign.sh [--targets TARGET ...]
       [--updates FAMILY=N] [--trial-seconds FAMILY=S] [--chunk-seconds S]
Submits fresh native diagnostic smoke, waits for terminal jobs and verified
artifacts, then submits its calibrated diagnostic. Polls every 30 seconds.
Preserves partial reports; never retries, cancels jobs, or falls back to CPU.
Raven: raven launch scripts/response_diagnostic_campaign.sh --local
       --recipe response-diagnostic -- [campaign options]
HELP
    exit 0
fi
targets=(fd_same_grid galerkin_same_grid galerkin_projected_fine)
overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --targets)
            shift; targets=()
            while [[ $# -gt 0 && "$1" != --* ]]; do targets+=("$1"); shift; done
            [[ ${#targets[@]} -gt 0 ]] || { printf 'TDN diagnostic: --targets requires targets.\n' >&2; exit 2; } ;;
        --updates|--trial-seconds|--chunk-seconds)
            [[ $# -ge 2 ]] || { printf 'TDN diagnostic: %s requires a value.\n' "$1" >&2; exit 2; }
            overrides+=("$1" "$2"); shift 2 ;;
        *) printf 'TDN diagnostic: unknown campaign option %s.\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${SLURM_JOB_ACCOUNT:-}" ]]; then
    printf 'TDN diagnostic: Run the campaign coordinator outside Slurm; use raven launch with --local. Scientific workers retain their native Slurm allocations.\n' >&2
    exit 2
fi
python="${TDN_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then python="$root/.venv/bin/python"; else python=python3; fi
fi
cd "$root"
raven_active=0
raven_logging=0
if [[ -n "${RAVEN_LAUNCH_PLAN:-}" ]]; then
    "$python" "$root/scripts/response_diagnostic_raven.py" check --snapshot "${BASH_SOURCE[0]}"
    raven_active=1
fi
token="$("$python" - <<'PY'
from datetime import datetime,timezone
import uuid
print(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:12])
PY
)"
smoke_id="response-diagnostic-smoke-$token"
diagnostic_id="response-diagnostic-$token"
receipt_dir="$root/runs/.response-diagnostic-campaigns/$token"
mkdir -p -- "$receipt_dir"
receipt="$receipt_dir/receipt.json"
"$python" - "$receipt" "$root" "$smoke_id" "$diagnostic_id" "${#targets[@]}" "${targets[@]}" "${overrides[@]}" <<'PY'
from datetime import datetime,timezone
import json,os,sys
from pathlib import Path
path,root,smoke,diagnostic,n=sys.argv[1:6];n=int(n)
value={'schema':'tdn.response-diagnostic-campaign/v1','created_at':datetime.now(timezone.utc).isoformat(),
       'project_root':root,'smoke_run_id':smoke,'diagnostic_run_id':diagnostic,
       'targets':sys.argv[6:6+n],'diagnostic_overrides':sys.argv[6+n:],
       'status':'RUNNING','phase':'CREATED','events':[],
       'poll_interval_seconds':30,'automatic_retries':False,'cpu_fallback':False}
if os.environ.get('RAVEN_LAUNCH_PLAN'):
    plan=json.loads(Path(os.environ['RAVEN_LAUNCH_PLAN']).read_text())
    run=Path(os.environ['RAVEN_RUN_DIR'])
    value['raven_launch']={'project_root':root,'coordinator_run_id':run.name,
        'coordinator_run_dir':run.relative_to(Path(root)).as_posix(),'launch_token':plan['token']}
with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
PY
receipt_update() {
    "$python" - "$receipt" "$1" "$2" "${3:-}" <<'PY'
from datetime import datetime,timezone
import json,os,sys
from pathlib import Path
path=Path(sys.argv[1]);phase,status,detail=sys.argv[2:5]
value=json.loads(path.read_text());now=datetime.now(timezone.utc).isoformat()
value.update(phase=phase,status=status,updated_at=now)
value['events'].append({'time':now,'phase':phase,'status':status,'detail':detail})
partial=path.with_suffix('.json.partial');partial.write_text(json.dumps(value,indent=2)+'\n');os.replace(partial,path)
PY
    if [[ "$raven_active" == 1 ]]; then
        "$python" "$root/scripts/response_diagnostic_raven.py" publish --receipt "$receipt"
    fi
}
on_exit() {
    local code="$1" original_code="$1"
    trap - EXIT
    if [[ "$code" != 0 ]]; then receipt_update "${phase:-CREATED}" FAILED "Coordinator exit $code; submitted jobs were preserved" || true; fi
    if [[ "$raven_logging" == 1 ]]; then
        exec 1>&"$raven_stdout" 2>&"$raven_stderr"
        exec {raven_stdout}>&- {raven_stderr}>&-
        wait "$raven_out_pid" || { [[ "$code" != 0 ]] || code=2; }
        wait "$raven_err_pid" || { [[ "$code" != 0 ]] || code=2; }
        if [[ "$code" != "$original_code" ]]; then
            receipt_update "${phase:-CREATED}" FAILED "Coordinator exit $code; submitted jobs and log evidence were preserved" || true
        fi
    fi
    exit "$code"
}
trap 'on_exit "$?"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$raven_active" == 1 ]]; then
    "$python" "$root/scripts/response_diagnostic_raven.py" initialize --receipt "$receipt" --snapshot "${BASH_SOURCE[0]}"
    exec {raven_stdout}>&1 {raven_stderr}>&2
    exec > >(tee -- "$RAVEN_RUN_DIR/logs/campaign.stdout.log")
    raven_out_pid=$!
    exec 2> >(tee -- "$RAVEN_RUN_DIR/logs/campaign.stderr.log" >&2)
    raven_err_pid=$!
    raven_logging=1
fi
printf 'Calibration run: %s\nDiagnostic run: %s\nCampaign receipt: %s\n' "$smoke_id" "$diagnostic_id" "$receipt"
wait_terminal() {
    local run_id="$1" snapshot="$receipt_dir/$1-status.json" state
    while true; do
        bash "$launcher" status "$run_id" --json > "$snapshot.partial"
        mv -- "$snapshot.partial" "$snapshot"
        state="$("$python" - "$snapshot" "$run_id" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text())
if value.get('run_id')!=sys.argv[2] or not isinstance(value.get('terminal'),bool):
    raise SystemExit('Invalid diagnostic status identity or terminal flag')
if not isinstance(value.get('units'),dict) or not isinstance(value.get('failed_units'),list):
    raise SystemExit('Invalid diagnostic status units')
print('terminal' if value['terminal'] else 'pending')
PY
)"
        if [[ "$raven_active" == 1 ]]; then
            "$python" "$root/scripts/response_diagnostic_raven.py" publish --receipt "$receipt" --status-file "$snapshot"
        fi
        printf 'TDN diagnostic: %s %s; status snapshot %s\n' "$run_id" "$state" "$snapshot"
        if [[ "$state" == terminal ]]; then return 0; fi
        sleep 30
    done
}
finish_run() {
    local run_id="$1" result=0
    bash "$launcher" status "$run_id" || result=2
    bash "$launcher" validate "$run_id" > "$receipt_dir/$run_id-validation.json" || result=2
    "$python" - "$receipt_dir/$run_id-status.json" <<'PY' || result=2
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text())
if not value.get('completed') or value.get('failed_units'):
    raise SystemExit('Infrastructure incomplete; preserve the afterany report and inspect failed units')
PY
    # Export even negative science or a failed unit; exporter retains omissions.
    bash "$launcher" compact "$run_id" --mode review || result=2
    return "$result"
}
phase=SMOKE_PLAN
receipt_update "$phase" RUNNING
bash "$launcher" plan --diagnostic-smoke --run-id "$smoke_id" --targets "${targets[@]}" | tee "$receipt_dir/smoke-plan.txt"
phase=SMOKE_SUBMIT
receipt_update "$phase" RUNNING
smoke_submit=0
bash "$launcher" run --diagnostic-smoke --run-id "$smoke_id" --targets "${targets[@]}" || smoke_submit=$?
phase=SMOKE_WAIT
receipt_update "$phase" RUNNING "Submission exit $smoke_submit"
wait_terminal "$smoke_id"
smoke_result=0
finish_run "$smoke_id" || smoke_result=$?
if [[ "$smoke_submit" != 0 || "$smoke_result" != 0 ]]; then
    receipt_update SMOKE_VERIFICATION FAILED 'Calibration infrastructure did not verify; diagnostic not submitted'
    exit 2
fi
phase=DIAGNOSTIC_PLAN
receipt_update "$phase" RUNNING
bash "$launcher" plan --diagnostic --calibration "$smoke_id" --run-id "$diagnostic_id" --targets "${targets[@]}" "${overrides[@]}" | tee "$receipt_dir/diagnostic-plan.txt"
phase=DIAGNOSTIC_SUBMIT
receipt_update "$phase" RUNNING
diagnostic_submit=0
bash "$launcher" run --diagnostic --calibration "$smoke_id" --run-id "$diagnostic_id" --targets "${targets[@]}" "${overrides[@]}" || diagnostic_submit=$?
phase=DIAGNOSTIC_WAIT
receipt_update "$phase" RUNNING "Submission exit $diagnostic_submit"
wait_terminal "$diagnostic_id"
diagnostic_result=0
finish_run "$diagnostic_id" || diagnostic_result=$?
bash "$launcher" decision "$diagnostic_id" > "$receipt_dir/diagnostic-decision.json" || diagnostic_result=2
if [[ "$diagnostic_submit" != 0 || "$diagnostic_result" != 0 ]]; then
    receipt_update DIAGNOSTIC_VERIFICATION FAILED 'Infrastructure failed or artifacts did not verify; review evidence retained'
    exit 2
fi
phase=COMPLETED
receipt_update "$phase" COMPLETED 'Both native runs verified; scientific decision is recorded separately'
printf 'Diagnostic campaign evidence is complete: %s\nScientific decision: %s\nRaven project: %s/runs/%s/raven\n' "$receipt" "$receipt_dir/diagnostic-decision.json" "$root" "$diagnostic_id"
