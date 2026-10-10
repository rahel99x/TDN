#!/usr/bin/env bash
# Verified saved-checkpoint pictures only: no training or Slurm submission.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ ! -x "$root/.venv/bin/python" ]]; then
    echo "TDN resolution images: the project .venv/bin/python is required." >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/.runtime/tmp" TMP="$root/.runtime/tmp" TEMP="$root/.runtime/tmp"
export MPLCONFIGDIR="$root/.cache/matplotlib" XDG_CACHE_HOME="$root/.cache"
exec "$root/.venv/bin/python" "$root/scripts/adjacent_resolution_images.py" "$@"
