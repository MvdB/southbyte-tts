#!/usr/bin/env bash
# Startet den Audio8-TTS-Container auf dem DGX Spark.
#
# Stimmen: 'default' ohne Referenz. Zum Klonen braucht dieses Modell je Stimme
# ZWEI Dateien im Stimmenverzeichnis: <name>.wav und <name>.txt mit dem
# gesprochenen Text der Aufnahme.
set -euo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-audio8-tts}"
HOST_PORT="${HOST_PORT:-8009}"
IMAGE="${IMAGE:-spark-audio8:v1}"
VOICES_DIR="${VOICES_DIR:-$HIER/../voices}"
MODEL_DIR="${MODEL_DIR:-Audio8--Audio8-TTS-Preview-0.6b}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

EXTRA=()
[[ -d "${VOICES_DIR}" ]] && EXTRA+=(-v "$(cd "${VOICES_DIR}" && pwd):/voices:ro")

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8009" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e AUDIO8_PATH="/hf_models/${MODEL_DIR}" \
  "${EXTRA[@]}" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
