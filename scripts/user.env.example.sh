#!/usr/bin/env bash
# Source this file after the read-only probe. No secrets belong here.
export CARC_ACCOUNT=anakano_81
export TDN_EXPECTED_USER=aadaniel
# Leave unset for automatic discovery among live authorized CPU partitions.
# export TDN_CPU_PARTITION=NAME_VERIFIED_BY_PROBE
# Optional: select an available module providing Python 3.11 or newer.
# export TDN_PYTHON_MODULE=NAME_FROM_MODULE_AVAIL_PYTHON
# Select from the official PyTorch selector after reading the actual driver audit.
# For GPU stages, a CUDA-enabled wheel is required; never modify the driver.
# export TORCH_VERSION=EXPLICIT_TESTED_VERSION
# export TORCH_WHEEL_INDEX=https://download.pytorch.org/whl/cuNNN
export TDN_RESUME=none
