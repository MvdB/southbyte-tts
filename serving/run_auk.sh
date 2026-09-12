#!/usr/bin/env bash
# Startet tencent/AuK bzw. AuK-Flash auf dem DGX Spark.
#
#   MODEL_DIR=tencent--AuK-Flash ./run_auk.sh      # destillierte 4-Schritt-Variante
#
# VORAUSSETZUNG: Qwen2.5-Omni-3B (12 GB) muss im Modellspeicher liegen — er ist
# der Text-Encoder, ohne ihn startet nichts.
#
# ACHTUNG: Weder Modellkarte noch Quelltext erwähnen Deutsch (AuK-Flash
# deklariert zh/en). Vor einem vollen Messlauf die Sprachprobe fahren:
#
#   ./sprachprobe_auk.sh
#
# Stimmen brauchen <name>.wav UND <name>.txt — das Transkript geht in die
# Dauerschätzung ein, denn AuK bestimmt seine Länge nicht selbst.
set -euo pipefail

HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
MODEL_DIR="${MODEL_DIR:-tencent--AuK}"
QWEN_DIR="${QWEN_DIR:-Qwen--Qwen2.5-Omni-3B}"
CONTAINER_NAME="${CONTAINER_NAME:-auk-tts}"
HOST_PORT="${HOST_PORT:-8014}"
IMAGE="${IMAGE:-spark-auk:v1}"
VOICES_DIR="${VOICES_DIR:-$HIER/../voices}"

for d in "$MODEL_DIR" "$QWEN_DIR"; do
  [[ -d "${HF_MODELS_DIR}/${d}" ]] || {
    echo "FEHLT im Modellspeicher: ${HF_MODELS_DIR}/${d}" >&2
    echo "  Beide gehoeren in die HF-Collection, sonst zieht der Sync sie nicht." >&2
    exit 1; }
done

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

EXTRA=()
[[ -d "${VOICES_DIR}" ]] && EXTRA+=(-v "$(cd "${VOICES_DIR}" && pwd):/voices:ro")

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_PORT}:8014" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e AUK_MODEL_DIR="/hf_models/${MODEL_DIR}" \
  -e AUK_QWEN_DIR="/hf_models/${QWEN_DIR}" \
  "${EXTRA[@]}" \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
