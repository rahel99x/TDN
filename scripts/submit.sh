#!/usr/bin/env bash
# Dry-run first; only an explicit --submit invokes sbatch.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
[[ $# -ge 2 ]] || tdn_die 'Usage: submit.sh STAGE CONFIG [--submit] [--dependency afterok:ID[:ID...]] [--run-dir PATH] [--dataset PATH] [--resume PATH] [--gate-report PATH] [--calibration-report PATH] [--pilot-budget STEPS]'
stage="$1"; config="$2"; shift 2
submit=false; dependency=""; run_dir=""; dataset=""; resume="${TDN_RESUME:-none}"; pilot_budget=""
gate_report="${TDN_GATE_REPORT:-}"; calibration_report="${TDN_CALIBRATION_REPORT:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit) submit=true; shift ;;
        --dry-run) submit=false; shift ;;
        --dependency|--run-dir|--dataset|--resume|--pilot-budget|--gate-report|--calibration-report)
            [[ $# -ge 2 ]] || tdn_die "Missing value for $1"
            case "$1" in
                --dependency) dependency="$2";; --run-dir) run_dir="$2";; --dataset) dataset="$2";;
                --resume) resume="$2";; --pilot-budget) pilot_budget="$2";;
                --gate-report) gate_report="$2";; --calibration-report) calibration_report="$2";;
            esac
            shift 2 ;;
        *) tdn_die "Unknown option $1" ;;
    esac
done
case "$stage" in
    setup) device=cpu; cpus=4; mem=16G; walltime=00:45:00 ;;
    cpu-tests|audit) device=cpu; cpus=4; mem=16G; walltime=00:30:00 ;;
    generate) device=cpu; cpus=8; mem=32G; walltime=01:00:00 ;;
    driver-audit) device=cuda; cpus=2; mem=4G; walltime=00:05:00 ;;
    gpu-tests) device=cuda; cpus=4; mem=16G; walltime=00:15:00 ;;
    calibrate) device=cuda; cpus=8; mem=64G; walltime=00:30:00 ;;
    train) device=cuda; cpus=8; mem=64G; walltime=04:00:00 ;;
    evaluate|benchmark) device=cuda; cpus=8; mem=64G; walltime=01:00:00 ;;
    *) tdn_die "Unknown stage $stage" ;;
esac
[[ "${CARC_ACCOUNT:-anakano_81}" == anakano_81 ]] || tdn_die 'Charging account must be anakano_81'
config="$(tdn_inside "$(realpath -m -- "$config")")"
[[ -f "$config" ]] || tdn_die 'Config is missing'
if [[ -n "$dependency" ]]; then
    [[ "$dependency" =~ ^afterok:[1-9][0-9]*(:[1-9][0-9]*)*$ ]] || tdn_die 'Only afterok dependencies on numeric real job IDs are accepted'
fi
if [[ -n "$pilot_budget" ]]; then
    [[ "$pilot_budget" =~ ^[1-9][0-9]*$ && "$pilot_budget" -le 1000 ]] || tdn_die 'Pilot budget must be 1-1000 optimizer steps'
fi
if [[ "$stage" == train && "$submit" == true && -z "$pilot_budget" ]]; then
    tdn_die 'Actual training requires an explicit --pilot-budget (maximum 1000 steps)'
fi
run_dir="$(tdn_inside "${run_dir:-$TDN_REPO_ROOT/runs/manual-$stage-$(date -u +%Y%m%dT%H%M%SZ)}")"
dataset="$(tdn_inside "${dataset:-$TDN_REPO_ROOT/data/pilot}")"
if [[ "$resume" != none ]]; then resume="$(tdn_inside "$resume")"; fi
[[ -z "$gate_report" ]] || gate_report="$(tdn_inside "$gate_report")"
[[ -z "$calibration_report" ]] || calibration_report="$(tdn_inside "$calibration_report")"
logroot="$(tdn_inside "$TDN_REPO_ROOT/logs")"
job_id_path="$(tdn_inside "$run_dir/slurm_job_id.txt")"
if [[ "$device" == cuda ]]; then
    partition=gpu
    batch="$TDN_REPO_ROOT/scripts/gpu.sbatch"
else
    partition="${TDN_CPU_PARTITION:-<discover-authorized-cpu-at-submit>}"
    batch="$TDN_REPO_ROOT/scripts/cpu.sbatch"
fi
if [[ "$submit" == true ]]; then
    tdn_runtime_policy
    command -v sbatch >/dev/null || tdn_die 'sbatch is unavailable'
    tdn_prepare_env
    tdn_load_python_module
    if [[ "$device" == cpu && -z "${TDN_CPU_PARTITION:-}" ]]; then
        partition="$(python3 "$TDN_REPO_ROOT/scripts/carc_check.py" --discover-cpu --walltime "$walltime" --cpus "$cpus" --mem-gib "${mem%G}")"
    fi
    [[ "$partition" =~ ^[A-Za-z0-9_.-]+$ ]] || tdn_die 'Invalid live partition name'
    tdn_mkdir "$logroot" "$run_dir"
    check_args=(python3 "$TDN_REPO_ROOT/scripts/carc_check.py" --partition "$partition"
                --walltime "$walltime" --cpus "$cpus" --mem-gib "${mem%G}" --output "$run_dir/slurm_policy.json")
    [[ "$device" == cuda ]] && check_args+=(--gpu)
    "${check_args[@]}"
    if [[ "$stage" != setup && "$stage" != driver-audit ]]; then
        python="$(tdn_python)"
        "$python" -c 'import tdn.cli' || tdn_die 'Install the package in its Python venv first'
        if [[ "$stage" == train ]]; then
            "$python" - "$config" "$pilot_budget" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
steps = config['training']['max_steps']
if not isinstance(steps, int) or not 1 <= steps <= int(sys.argv[2]):
    raise SystemExit('Config optimizer steps exceed the explicitly authorized pilot budget')
if config['purpose'] != 'development' or config['runtime']['confirmatory_authorized']:
    raise SystemExit('This submission wrapper authorizes development pilots only')
grid = config['problem']['grid']
if len(grid) > 2 or any(n > 64 for n in grid):
    raise SystemExit('The initial bounded pilot is limited to small 1-D/2-D grids')
PY
        fi
    fi
fi
export CARC_ACCOUNT=anakano_81 TDN_STAGE="$stage" TDN_CONFIG="$config"
export TDN_RUN_DIR="$run_dir" TDN_DATASET="$dataset" TDN_DEVICE="$device" TDN_RESUME="$resume"
export TDN_GATE_REPORT="$gate_report" TDN_CALIBRATION_REPORT="$calibration_report"
args=(sbatch --parsable --account=anakano_81 --partition="$partition" --nodes=1 --ntasks=1
      --job-name="tdn-$stage" --cpus-per-task="$cpus" --mem="$mem" --time="$walltime"
      --signal=USR1@180 --export=ALL --output="$logroot/%x-%j.out" --error="$logroot/%x-%j.err")
if [[ "$device" == cuda ]]; then args+=(--gpus-per-task=a100:1 --constraint=a100-40gb); fi
[[ -n "$dependency" ]] && args+=(--dependency="$dependency")
args+=("$batch")
printf 'Stage=%s device=%s account=anakano_81\nRun=%s\nDataset=%s\n' "$stage" "$device" "$run_dir" "$dataset"
printf 'Command: '; printf '%q ' "${args[@]}"; printf '\n'
if [[ "$submit" == false ]]; then
    printf 'DRY RUN: no allocation, live authorization check, installation or directories created.\n'
    exit 0
fi
job_output="$("${args[@]}")"
[[ "$job_output" =~ ^([1-9][0-9]*)(\;[A-Za-z0-9_.-]+)?$ ]] || tdn_die 'sbatch did not return a valid parsable job ID; inspect scheduler before retrying'
job_id="${BASH_REMATCH[1]}"
printf '%s\n' "$job_id" > "$job_id_path"
printf 'TDN_JOB_ID=%s\n' "$job_id"
