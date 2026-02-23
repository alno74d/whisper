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
- Tested with `large-v3` and `large-v3-turbo` models

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

### 2. Download a Whisper model

Pick one (or both). Turbo is faster with minimal quality loss; large-v3 is more accurate.

```bash
mkdir -p models

# large-v3-turbo (~1.6 GB) — recommended, much faster
wget -O models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# large-v3 (~3 GB) — best accuracy
wget -O models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

Set `MODEL_PATH` in `docker-compose.yml` to match whichever model you downloaded.

### 3. Start the stack

You can either pull pre-built images from Docker Hub or build locally.

**Pull from Docker Hub (recommended):**

```bash
docker compose pull
docker compose up -d
```

**Build locally:**

```bash
docker compose build
docker compose up -d
```

Building locally compiles whisper.cpp from source with oneAPI/SYCL support inside the container. No oneAPI installation needed on the host.

### 4. Verify

```bash
docker compose ps
curl http://localhost:9000/health
```

## Deploy on another machine

On a target machine that has Docker and the Intel GPU drivers, you only need three things:

1. A `docker-compose.yml` (copy it from this repo or write a minimal one):

```yaml
services:
  whisper-server:
    image: lozachmp/whisper-server:latest
    devices:
      - /dev/dri:/dev/dri
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/ggml-large-v3-turbo.bin
      - GGML_SYCL_DEVICE=0
      - WHISPER_THREADS=16
    networks:
      - whisper-net
    restart: unless-stopped

  wrapper:
    image: lozachmp/whisper-wrapper:latest
    ports:
      - "9000:9000"
    environment:
      - WHISPER_SERVER_URL=http://whisper-server:8081
    depends_on:
      - whisper-server
    networks:
      - whisper-net
    restart: unless-stopped

networks:
  whisper-net:
```

2. A Whisper model file (pick one):

```bash
mkdir -p models

# large-v3-turbo (~1.6 GB) — recommended
wget -O models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# large-v3 (~3 GB) — best accuracy
wget -O models/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

Update `MODEL_PATH` in the compose file to match (e.g., `/models/ggml-large-v3-turbo.bin`).

3. Pull and start:

```bash
docker compose pull
docker compose up -d
```

No build tools, no source code needed on the target machine.

## Publishing images to Docker Hub

Images are built and pushed from the build machine (Threadripper) using `build.sh`.

```bash
# Log in to Docker Hub (one-time)
docker login

# Build and push both images
./build.sh

# Build only (no push)
./build.sh --no-push
```

The script tags each image with `latest` and the current git short SHA (e.g., `lozachmp/whisper-server:ed4c4fe`).

## Bazarr configuration

In Bazarr settings, set the Whisper provider URL to:

```
http://<host-ip>:9000
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/ggml-large-v3-turbo.bin` | Path to the GGML model inside the container |
| `GGML_SYCL_DEVICE` | `0` | SYCL device index for the GPU |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` | oneAPI device selector |
| `WHISPER_THREADS` | all cores (`nproc`) | Number of CPU threads for whisper-server |
| `WHISPER_SERVER_URL` | `http://whisper-server:8081` | Internal URL the wrapper uses to reach whisper-server |

### GPU device selection

If your system has multiple GPUs (e.g., an iGPU + the Arc A770), you may need to adjust `GGML_SYCL_DEVICE` or `ONEAPI_DEVICE_SELECTOR` in `docker-compose.yml` to target the correct device.
