#!/usr/bin/env bash
# Read-only evidence export; no Slurm allocation or scientific imports needed.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
python="${TDN_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
    printf '%s\n' 'TDN: use this checkout’s Python venv or set TDN_PYTHON to its interpreter.' >&2
    exit 2
fi
exec "$python" "$root/scripts/compact_review.py" "$@"
