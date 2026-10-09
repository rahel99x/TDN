#!/usr/bin/env bash
# CPU smoke/development only; no scheduler identity or native-GPU claims.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf 'Usage: bash scripts/portfolio_local.sh [--smoke|--development] [--run-dir PROJECT_PATH]\nRuns the finite research DAG on CPU; fresh full confirmation requires Fedora Slurm.\n'
    exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ "$TDN_REPO_ROOT" != "$TDN_CARC_ROOT" ]] || tdn_die 'Use an allocated workflow on CARC'
[[ -z "${SLURM_JOB_ID:-}${SLURM_STEP_ID:-}${TDN_EXECUTION_MODE:-}${TDN_PROJECT_ROOT:-}" ]] || tdn_die 'Local CPU portfolio requires a separate checkout without Slurm overrides'
profile=smoke
run_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) profile=smoke; shift ;;
        --development) profile=development; shift ;;
        --full) tdn_die 'Fresh full confirmation requires allocated Fedora Slurm' ;;
        --run-dir) [[ $# -ge 2 ]] || tdn_die '--run-dir needs a path'; run_dir="$2"; shift 2 ;;
        *) tdn_die "Unknown option: $1" ;;
    esac
done
export TDN_LOCAL_TEST_ROOT="$TDN_REPO_ROOT"
unset TDN_TOWER_DIR TDN_FRONTIER_WORKFLOW TDN_ROADMAP_WORKFLOW TDN_AGENDA_WORKFLOW TDN_PREMIX_WORKFLOW TDN_CONSISTENCY_WORKFLOW
unset TDN_PORTFOLIO_WORKFLOW TDN_PORTFOLIO_STAGE TDN_PORTFOLIO_PROTOCOL_SHA256
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
tdn_prepare_env
python="$(tdn_python)"
cd "$TDN_REPO_ROOT"
if [[ -z "$run_dir" ]]; then run_dir="$TDN_REPO_ROOT/runs/portfolio-local-$(date -u +%Y%m%dT%H%M%S%N)"; fi
[[ "$run_dir" == /* ]] || run_dir="$TDN_REPO_ROOT/$run_dir"
run_dir="$(tdn_inside "$run_dir")"
[[ ! -e "$run_dir" ]] || tdn_die 'Run path exists; preserve it and choose a fresh directory'
exec 9<> "$TDN_REPO_ROOT/.cache/carc-phase.lock"
flock -s -n 9 || tdn_die 'Project venv is being modified; retry after setup'
"$python" - "$profile" "$run_dir" "$TDN_REPO_ROOT" <<'PY'
import json,subprocess,sys
from pathlib import Path
from tdn.analysis.portfolio.protocol import build_protocol,plan_units
from tdn.runtime.metadata import write_json
root=Path(sys.argv[3]);run=Path(sys.argv[2]);protocol=build_protocol(sys.argv[1]);run.mkdir(parents=True,exist_ok=False)
write_json(run/'local-protocol.json',protocol)
failed=False;states={}
for unit in plan_units(protocol):
    stage=unit['id']; prior=set()
    def ancestors(name):
        for parent in protocol['units'][name]['dependencies']:
            if parent not in prior: prior.add(parent);ancestors(parent)
    ancestors(stage)
    if stage!='report' and any(states.get(name)!='COMPLETED' for name in unit['dependencies']):
        states[stage]='BLOCKED';print(f'TDN portfolio: {stage}: BLOCKED by failed dependency',flush=True);continue
    command=[sys.executable,str(root/'scripts/portfolio.py'),'--stage',stage,'--profile',sys.argv[1],
        '--run-dir',str(run/stage),'--device','cpu','--local-root',str(root)]
    for name in protocol['units']:
        if name in prior: command += ['--prerequisite-dir',f'{name}={run/name}']
    result=subprocess.run(command,cwd=root,check=False)
    states[stage]='COMPLETED' if result.returncode==0 else 'FAILED';failed=failed or result.returncode!=0
    write_json(run/'local-state.json',states)
print(f'Local CPU portfolio complete (runtime failure={failed}): {run}',flush=True)
raise SystemExit(int(failed))
PY
