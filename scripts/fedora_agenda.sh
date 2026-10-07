#!/usr/bin/env bash
# Eight bounded research stages on the explicit Fedora desktop Slurm profile.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# == 0 ]]; then set -- plan; fi
case "$1" in
    doctor|configure|setup) exec bash "$root/scripts/fedora_setup.sh" "$@" ;;
esac
export PYTHONDONTWRITEBYTECODE=1
exec "${TDN_PYTHON:-python3}" "$root/scripts/agenda_workflow.py" "$@"
