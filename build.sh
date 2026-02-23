#!/usr/bin/env bash
set -euo pipefail

REPO_PREFIX="lozachmp"
IMAGES=("whisper-server:Dockerfile.whisper" "whisper-wrapper:Dockerfile.wrapper")
PUSH=true

for arg in "$@"; do
  case "$arg" in
    --no-push) PUSH=false ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# Check Docker login if pushing
if $PUSH; then
  if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "Error: not logged in to Docker Hub. Run 'docker login' first."
    exit 1
  fi
fi

VERSION=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "Version tag: $VERSION"

for entry in "${IMAGES[@]}"; do
  NAME="${entry%%:*}"
  DOCKERFILE="${entry##*:}"
  FULL_NAME="${REPO_PREFIX}/${NAME}"

  echo ""
  echo "=== Building ${FULL_NAME} ==="

  docker build \
    -t "${FULL_NAME}:latest" \
    -t "${FULL_NAME}:${VERSION}" \
    -f "${DOCKERFILE}" \
    .

  if $PUSH; then
    echo "--- Pushing ${FULL_NAME} ---"
    docker push "${FULL_NAME}:latest"
    docker push "${FULL_NAME}:${VERSION}"
  fi
done

echo ""
echo "Done."
