#!/usr/bin/env bash
set -euo pipefail
if [[ $# == 0 ]]; then set -- latest; fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/carc.sh" status "$@"
