#!/usr/bin/env bash
# Three-stage bounded consistency experiment, using the existing Fedora setup.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# == 0 ]]; then set -- plan; fi
case "$1" in
    doctor|configure|setup) exec bash "$root/scripts/fedora_setup.sh" "$@" ;;
esac
export PYTHONDONTWRITEBYTECODE=1
exec "${TDN_PYTHON:-python3}" "$root/scripts/consistency_workflow.py" "$@"
