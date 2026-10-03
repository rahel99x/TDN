#!/usr/bin/env bash
# Known CARC choices from the actual A100 driver audit, not a CUDA toolkit guess.
# This sourceable file changes no shell options and performs no installation.
export CARC_ACCOUNT="${CARC_ACCOUNT:-anakano_81}"
export TDN_CPU_PARTITION="${TDN_CPU_PARTITION:-main}"
export TDN_PYTHON_MODULE="${TDN_PYTHON_MODULE:-python/3.11.9}"
export TORCH_VERSION="${TORCH_VERSION:-2.10.0+cu126}"
export TORCH_WHEEL_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu126}"
export TDN_RESUME=none
