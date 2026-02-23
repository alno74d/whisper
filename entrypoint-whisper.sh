#!/bin/bash
set -e

source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true

export GGML_SYCL_DEVICE=${GGML_SYCL_DEVICE:-0}
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}

MODEL_PATH=${MODEL_PATH:-/models/ggml-large-v3-turbo.bin}
WHISPER_THREADS=${WHISPER_THREADS:-$(nproc)}

echo "Starting whisper-server with model: ${MODEL_PATH}"
echo "SYCL device: ${GGML_SYCL_DEVICE}"
echo "Threads: ${WHISPER_THREADS}"

exec whisper-server \
    --host 0.0.0.0 \
    --port 8081 \
    -m "${MODEL_PATH}" \
    --threads "${WHISPER_THREADS}" \
    --print-progress \
    "$@"
