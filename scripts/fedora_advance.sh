#!/usr/bin/env bash
# Three-path advance with finite dependency jobs and exact-source recovery.
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
if [[ "$1" == compact ]]; then
    shift
    # Resolve this campaign's pointer before handing an explicit run to the
    # shared exporter. Its historical default pointer belongs to portfolio.
    run="${1:-latest}"
    if [[ $# -gt 0 ]]; then shift; fi
    if [[ "$run" == latest ]]; then
        run="$("$python" - "$root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
print(json.loads((root/'runs/.fedora-advance-latest.json').read_text())['run_id'])
PY
)"
    fi
    exec "$python" "$root/scripts/compact_review.py" pack "$run" "$@"
fi
exec "$python" "$root/scripts/advance_workflow.py" "$@"
