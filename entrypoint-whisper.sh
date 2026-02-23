#!/bin/bash
set -e

source /opt/intel/oneapi/setvars.sh

export GGML_SYCL_DEVICE=${GGML_SYCL_DEVICE:-0}
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}

MODEL_PATH=${MODEL_PATH:-/models/ggml-large-v3.bin}

echo "Starting whisper-server with model: ${MODEL_PATH}"
echo "SYCL device: ${GGML_SYCL_DEVICE}"

exec whisper-server \
    --host 0.0.0.0 \
    --port 8081 \
    -m "${MODEL_PATH}" \
    --convert \
    --no-mmap \
    "$@"
