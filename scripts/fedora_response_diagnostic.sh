#!/usr/bin/env bash
# Separate Response diagnostics; native work remains in checked Slurm jobs.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# == 0 ]]; then set -- plan; fi
case "$1" in
    doctor|configure|setup) exec bash "$root/scripts/fedora_setup.sh" "$@" ;;
esac
export PYTHONDONTWRITEBYTECODE=1
export TDN_RESPONSE_POINTER=.fedora-response-diagnostic-latest.json
python="${TDN_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then python="$root/.venv/bin/python"; else python=python3; fi
fi
if [[ "$1" == compact ]]; then
    shift
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then shift; fi
    if [[ "$run" == latest ]]; then
        run="$("$python" - "$root" <<'PY'
import json,sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'runs/.fedora-response-diagnostic-latest.json').read_text())['run_id'])
PY
)"
    fi
    exec "$python" "$root/scripts/compact_review.py" pack "$run" "$@"
fi
if [[ "$1" == plan || "$1" == run ]]; then
    selected=0
    for arg in "$@"; do
        case "$arg" in
            --diagnostic-smoke|--diagnostic) selected=1 ;;
            --smoke|--pilot|--pilot=*|--development|--confirmation)
                printf 'TDN diagnostic: use --diagnostic-smoke or --diagnostic in this separate workflow.\n' >&2
                exit 2 ;;
        esac
    done
    if [[ "$selected" == 0 ]]; then set -- "$@" --diagnostic-smoke; fi
fi
exec "$python" "$root/scripts/response_workflow.py" "$@"
