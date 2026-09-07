#!/usr/bin/env bash
# Startet Breeze-TTS-2 auf dem DGX Spark.
#
# ACHTUNG: Dieses Modell kann kein Deutsch (Modellkarte: nur en/zh) und steht
# unter einer nicht-kommerziellen Lizenz. Ein Lauf gegen den deutschen
# Testsatz dokumentiert eine Sprachlücke, keine Sprachqualität.
#
# Zum Klonen braucht es je Stimme <name>.wav UND <name>.txt mit dem
# gesprochenen Text der Aufnahme.
set -euo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-breeze-tts}"
HOST_PORT="${HOST_PORT:-8013}"
IMAGE="${IMAGE:-spark-breeze:v1}"
VOICES_DIR="${VOICES_DIR:-$HIER/../voices}"
MODEL_DIR="${MODEL_DIR:-BreezeBlue--Breeze-TTS-2}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

EXTRA=()
[[ -d "${VOICES_DIR}" ]] && EXTRA+=(-v "$(cd "${VOICES_DIR}" && pwd):/voices:ro")

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8013" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e BREEZE_PATH="/hf_models/${MODEL_DIR}" \
  "${EXTRA[@]}" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
