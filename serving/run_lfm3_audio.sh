#!/usr/bin/env bash
# Startet LFM3-Audio-2B (TTS + ASR) auf dem DGX Spark, Port 8016.
# MODEL_DIR=LiquidAI--LFM3-Omni-2B lädt Omni — dann nur Transkription (Judge-Kalibrierung).
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-lfm3-audio}"
HOST_PORT="${HOST_PORT:-8016}"
IMAGE="${IMAGE:-spark-lfm3-audio:v1}"
MODEL_DIR="${MODEL_DIR:-LiquidAI--LFM3-Audio-2B}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" --gpus all --ipc=host \
  -p "${HOST_PORT}:8016" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e LFM3_AUDIO_PATH="/hf_models/${MODEL_DIR}" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
