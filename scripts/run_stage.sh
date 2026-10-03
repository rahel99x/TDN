#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
tdn_runtime_policy
tdn_allocation
tdn_prepare_env
: "${TDN_STAGE:?}" "${TDN_CONFIG:?}" "${TDN_RUN_DIR:?}" "${TDN_DEVICE:?}"
TDN_CONFIG="$(tdn_inside "$TDN_CONFIG")"
TDN_RUN_DIR="$(tdn_inside "$TDN_RUN_DIR")"
TDN_DATASET="$(tdn_inside "${TDN_DATASET:-$TDN_REPO_ROOT/data/pilot}")"
export TDN_CONFIG TDN_RUN_DIR TDN_DATASET
tdn_mkdir "$TDN_RUN_DIR"
cd "$TDN_REPO_ROOT"
if [[ "$TDN_STAGE" == driver-audit ]]; then
    [[ "$TDN_DEVICE" == cuda ]] || tdn_die 'Driver audit requires a GPU task'
    exec bash "$TDN_REPO_ROOT/scripts/driver_audit.sh"
fi
if [[ "$TDN_STAGE" == setup ]]; then
    exec bash "$TDN_REPO_ROOT/scripts/setup_venv.sh"
fi
python="$(tdn_python)"
if [[ "$TDN_STAGE" == light-tests || "$TDN_STAGE" == light-screen ]]; then
    [[ "$TDN_DEVICE" == cpu ]] || tdn_die 'Light screening and tests require a CPU task'
fi
if [[ "$TDN_DEVICE" == cuda ]]; then
    "$python" "$TDN_REPO_ROOT/scripts/gpu_preflight.py" --output "$TDN_RUN_DIR/gpu_preflight.json"
fi
case "$TDN_STAGE" in
    light-tests)
        pytest_work="$(tdn_inside "$TDN_RUN_DIR/pytest-work")"
        exec "$python" -m pytest -q tests/test_light_screen.py tests/test_temporal_oracle_screen.py \
            tests/test_light_screen_cli.py tests/test_analysis.py \
            --basetemp "$pytest_work" ;;
    light-screen)
        exec "$python" -u "$TDN_REPO_ROOT/scripts/light_screen.py" \
            --config "$TDN_CONFIG" --run-dir "$TDN_RUN_DIR" ;;
    cpu-tests)
        pytest_work="$(tdn_inside "$TDN_RUN_DIR/pytest-work")"
        exec "$python" -m pytest -q -m 'not gpu' --basetemp "$pytest_work" ;;
    gpu-tests)
        export TDN_REQUIRE_GPU_TESTS=1
        pytest_work="$(tdn_inside "$TDN_RUN_DIR/pytest-work")"
        exec "$python" -m pytest -q -m gpu --basetemp "$pytest_work" ;;
    audit|generate|calibrate|train|evaluate|benchmark) ;;
    *) tdn_die "Unknown stage $TDN_STAGE" ;;
esac
resume="${TDN_RESUME:-none}"
if [[ "$resume" != none ]]; then resume="$(tdn_inside "$resume")"; fi
if [[ -n "${TDN_GATE_REPORT:-}" ]]; then export TDN_GATE_REPORT="$(tdn_inside "$TDN_GATE_REPORT")"; fi
if [[ -n "${TDN_CALIBRATION_REPORT:-}" ]]; then export TDN_CALIBRATION_REPORT="$(tdn_inside "$TDN_CALIBRATION_REPORT")"; fi
exec "$python" -u -m tdn.cli "$TDN_STAGE" --config "$TDN_CONFIG" \
    --run-dir "$TDN_RUN_DIR" --dataset "$TDN_DATASET" --device "$TDN_DEVICE" --resume "$resume"
