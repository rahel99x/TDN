#!/usr/bin/env bash
# Three-path portfolio with finite dependency jobs and exact-source recovery.
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
exec "$python" "$root/scripts/portfolio_workflow.py" "$@"
