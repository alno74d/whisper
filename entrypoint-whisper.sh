#!/bin/bash
set -e

source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true

export GGML_SYCL_DEVICE=${GGML_SYCL_DEVICE:-0}
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}

MODEL_PATH=${MODEL_PATH:-/models/ggml-large-v3-turbo.bin}
WHISPER_THREADS=${WHISPER_THREADS:-$(nproc)}

# Download model if not present
if [ ! -f "${MODEL_PATH}" ]; then
  MODEL_NAME=$(basename "${MODEL_PATH}")
  URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MODEL_NAME}"
  echo "Model not found at ${MODEL_PATH}, downloading from ${URL} ..."
  mkdir -p "$(dirname "${MODEL_PATH}")"
  wget -q --show-progress -O "${MODEL_PATH}" "${URL}"
  echo "Download complete."
fi

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
