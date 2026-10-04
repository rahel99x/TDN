#!/usr/bin/env bash
# Project report tools only. Never install or modify the Tower application.
set -euo pipefail
TDN_TOWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$TDN_TOWER_ROOT/scripts/tower.py" "$@"
