#!/usr/bin/env bash
# Startet Audio8 0.1B ONNX INT8 auf dem DGX Spark.
#
# Läuft auf der CPU — kein --gpus. Stimmen sind hier keine WAV-Dateien, sondern
# registrierte Codesätze: einmalig aus Aufnahme und Transkript erzeugen mit
#
#   curl http://127.0.0.1:8012/api/voices/register \
#     -F audio=@voices/de_f1.wav -F "text=$(cat voices/de_f1.txt)" -F name=de_f1
#
# Der Codesatz liegt danach in VOICES_STORE und übersteht einen Neustart.
set -euo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-audio8-onnx}"
HOST_PORT="${HOST_PORT:-8012}"
IMAGE="${IMAGE:-spark-audio8-onnx:v1}"
VOICES_STORE="${VOICES_STORE:-$HIER/../voices_onnx}"
THREADS="${THREADS:-6}"

mkdir -p "$VOICES_STORE"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:8012" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -v "$(cd "$VOICES_STORE" && pwd):/voices_onnx" \
  -e ARKTTS_THREADS="${THREADS}" \
  -e ARKTTS_REGISTRATION_DIR=/hf_models/Audio8--audio8-TTS-0.1B-ONNX-INT8/registration \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
