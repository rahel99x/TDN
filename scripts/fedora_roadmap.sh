#!/usr/bin/env bash
# Ten bounded research stages on the explicit Fedora desktop Slurm profile.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# == 0 ]]; then set -- plan; fi
case "$1" in
    doctor|configure|setup) exec bash "$root/scripts/fedora_setup.sh" "$@" ;;
esac
export PYTHONDONTWRITEBYTECODE=1
# Artifact validation imports the scientific dependencies installed in this
# checkout's venv. An interactive shell need not have activated that venv.
python="${TDN_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then
        python="$root/.venv/bin/python"
    else
        # Planning/bootstrap remains usable before the project venv exists.
        python=python3
    fi
fi
exec "$python" "$root/scripts/roadmap_workflow.py" "$@"
