#!/usr/bin/env bash
# Move only one proven-terminal diagnostic campaign; preserve every file.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
python="${TDN_PYTHON:-}"
if [[ -z "$python" ]]; then
    if [[ -x "$root/.venv/bin/python" ]]; then python="$root/.venv/bin/python"; else python=python3; fi
fi
exec "$python" "$root/scripts/response_diagnostic_archive.py" "$@"
