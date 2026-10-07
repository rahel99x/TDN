#!/usr/bin/env bash
# Bootstrap is standard-library-only; the selected numerical venv is separate.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
exec "${TDN_PYTHON:-python3}" "$root/scripts/fedora_setup.py" "$@"
