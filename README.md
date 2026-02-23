# Whisper for Bazarr (Intel Arc GPU)

Dockerized Whisper transcription stack for [Bazarr](https://www.bazarr.media/), accelerated with Intel Arc GPUs via SYCL/oneAPI.

## Architecture

Two containers:

- **whisper-server** — [whisper.cpp](https://github.com/ggerganov/whisper.cpp) compiled with Intel SYCL (oneAPI), serving the `/inference` endpoint on port 8081
- **wrapper** — FastAPI app that translates Bazarr's whisper-asr-webservice API (`/asr`, `/detect-language`) into whisper.cpp's `/inference` format, exposed on port 9000

```
Bazarr  -->  wrapper (:9000)  -->  whisper-server (:8081)
              /asr                   /inference
              /detect-language
              /health
```

## Target hardware

- Intel Arc A770 (16 GB VRAM)
- AMD Threadripper CPU
- Tested with the `large-v3` model (~3 GB)

## Prerequisites

- Docker and Docker Compose v2
- Intel GPU kernel drivers (i915) and compute runtime installed on the host
- `/dev/dri` must be available with `renderD128`:

```bash
ls -la /dev/dri/
# Expected: card0 or card1, renderD128
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/alno74d/whisper.git
cd whisper
```

### 2. Download the Whisper model

```bash
mkdir -p models
wget -O models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

### 3. Build and start

```bash
docker compose build
docker compose up -d
```

The build compiles whisper.cpp from source with oneAPI/SYCL support inside the container. No oneAPI installation needed on the host.

### 4. Verify

```bash
docker compose ps
curl http://localhost:9000/health
```

## Bazarr configuration

In Bazarr settings, set the Whisper provider URL to:

```
http://<host-ip>:9000
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/ggml-large-v3.bin` | Path to the GGML model inside the container |
| `GGML_SYCL_DEVICE` | `0` | SYCL device index for the GPU |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` | oneAPI device selector |
| `WHISPER_SERVER_URL` | `http://whisper-server:8081` | Internal URL the wrapper uses to reach whisper-server |

### GPU device selection

If your system has multiple GPUs (e.g., an iGPU + the Arc A770), you may need to adjust `GGML_SYCL_DEVICE` or `ONEAPI_DEVICE_SELECTOR` in `docker-compose.yml` to target the correct device.
