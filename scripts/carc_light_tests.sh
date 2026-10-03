#!/usr/bin/env bash
# CPU-only bounded scientific screening in the existing project Python venv.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/carc.sh" light "$@"
