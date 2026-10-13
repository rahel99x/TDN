#!/usr/bin/env bash
# Raven runs this submitting coordinator locally; science stays allocated.
set -euo pipefail
if [[ -n "${RAVEN_LAUNCH_PLAN:-}" ]]; then
    root="$(python3 - "$RAVEN_LAUNCH_PLAN" <<'PY'
import json,sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text())
root=plan.get('project_root')
if not isinstance(root,str) or not Path(root).is_absolute():raise SystemExit('Invalid Raven project root')
print(root)
PY
)"
else
    root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
fi
if [[ -n "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${SLURM_JOB_ACCOUNT:-}" ]]; then
    printf 'Tangent response: the coordinator requires Raven --local outside an allocation; native workers use Slurm.\n' >&2
    exit 2
fi
python="${TDN_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then python="$root/.venv/bin/python"; else python=python3; fi
fi
export PYTHONDONTWRITEBYTECODE=1
raven_logging=0
on_exit() {
    local code="$1"
    trap - EXIT
    if [[ "$raven_logging" == 1 ]]; then
        exec 1>&"$raven_stdout" 2>&"$raven_stderr"
        exec {raven_stdout}>&- {raven_stderr}>&-
        wait "$raven_out_pid" || { [[ "$code" != 0 ]] || code=2; }
        wait "$raven_err_pid" || { [[ "$code" != 0 ]] || code=2; }
    fi
    exit "$code"
}
trap 'on_exit "$?"' EXIT
if [[ -n "${RAVEN_LAUNCH_PLAN:-}" ]]; then
    "$python" "$root/scripts/tangent_response_raven.py" check --snapshot "${BASH_SOURCE[0]}"
    export TDN_TANGENT_CAPTURED_SCRIPT="${BASH_SOURCE[0]}"
    exec {raven_stdout}>&1 {raven_stderr}>&2
    exec > >(tee -- "$RAVEN_RUN_DIR/logs/campaign.stdout.log")
    raven_out_pid=$!
    exec 2> >(tee -- "$RAVEN_RUN_DIR/logs/campaign.stderr.log" >&2)
    raven_err_pid=$!
    raven_logging=1
fi
cd -- "$root"
"$python" "$root/scripts/tangent_response_workflow.py" campaign "$@"
