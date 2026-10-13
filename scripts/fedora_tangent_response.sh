#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# == 0 ]]; then set -- plan; fi
case "$1" in
    doctor|configure|setup) exec bash "$root/scripts/fedora_setup.sh" "$@" ;;
    _worker-batch|_worker-task)
        : "${TDN_TANGENT_WORKFLOW:?}" "${TDN_TANGENT_STAGE:?}" "${TDN_TANGENT_CPUS:?}" "${SLURM_JOB_ID:?}"
        [[ "${TDN_EXECUTION_MODE:-}" == desktop-slurm && "$root" != /home1/aadaniel/projects/TDN ]] || exit 2
        [[ "$TDN_TANGENT_CPUS" =~ ^[1-8]$ && "$TDN_TANGENT_STAGE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
        [[ "${TDN_TANGENT_DEVICE:-}" == cpu || "${TDN_TANGENT_DEVICE:-}" == cuda ]] || exit 2
        cd -- "$root"
        source "$root/scripts/common.sh"
        tdn_prepare_env
        export TDN_EXECUTION_MODE=desktop-slurm TDN_PROJECT_ROOT="$root" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
        if [[ "$1" == _worker-batch ]]; then
            gres=none
            if [[ "$TDN_TANGENT_DEVICE" == cuda ]]; then gres="${TDN_FEDORA_GPU_GRES:?}"; fi
            exec srun --ntasks=1 --cpus-per-task="$TDN_TANGENT_CPUS" --unbuffered --export=ALL --gres="$gres" \
                bash "$root/scripts/fedora_tangent_response.sh" _worker-task
        fi
        [[ "${SLURM_STEP_ID:-}" =~ ^[0-9]+$ ]] || exit 2
        exec "$root/.venv/bin/python" "$root/scripts/tangent_response_workflow.py" worker \
            --workflow "$TDN_TANGENT_WORKFLOW" --stage "$TDN_TANGENT_STAGE" ;;
esac
export PYTHONDONTWRITEBYTECODE=1
python="${TDN_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then python="$root/.venv/bin/python"; else python=python3; fi
fi
exec "$python" "$root/scripts/tangent_response_workflow.py" "$@"
