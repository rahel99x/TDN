#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TDN_PYTHON:-$project_root/.venv/bin/python}"
exec "$python_bin" "$project_root/scripts/tangent_response_archive.py" "$@"
