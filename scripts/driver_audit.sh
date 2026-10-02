#!/usr/bin/env bash
# Driver observation only, before wheel selection. No Python or CUDA kernels.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tdn_runtime_policy
tdn_allocation
: "${TDN_RUN_DIR:?}"
run_dir="$(tdn_inside "$TDN_RUN_DIR")"
tdn_mkdir "$run_dir"
command -v nvidia-smi >/dev/null || tdn_die 'nvidia-smi unavailable in the allocated task'
csv_path="$(tdn_inside "$run_dir/driver_audit.csv")"
metadata_path="$(tdn_inside "$run_dir/driver_audit_metadata.txt")"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
    > "$csv_path"
{
    printf 'category=newly_measured_result\n'
    printf 'slurm_job_id=%s\nslurm_step_id=%s\naccount=%s\n' \
        "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$SLURM_JOB_ACCOUNT"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'note=nvidia-smi can list physical GPUs beyond the task-visible mapping; no CUDA kernels ran.\n'
} > "$metadata_path"
cat "$csv_path"
printf 'Driver observations saved in %s\n' "$run_dir"
