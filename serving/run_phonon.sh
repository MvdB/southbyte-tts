#!/usr/bin/env bash
# Startet phonon-40m (gradium) auf dem DGX Spark.
#
# Laeuft auf der CPU — kein --gpus — und ohne Netz: ptts laedt von lokalen
# Pfaden nie nach, die Stimmen kommen aus gradium--phonon-voices. --network
# none waere zu streng (der Port muss erreichbar sein); die Gewaehr gibt der
# lokale Pfad.
#
#   ./run_phonon.sh                                        # phonon-40m-de
#   MODEL_DIR=gradium--phonon-40m-multilingual ./run_phonon.sh
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
MODEL_DIR="${MODEL_DIR:-gradium--phonon-40m-de}"
CONTAINER_NAME="${CONTAINER_NAME:-phonon-tts}"
HOST_PORT="${HOST_PORT:-8015}"
IMAGE="${IMAGE:-spark-phonon:v1}"
THREADS="${THREADS:-0}"

[[ -f "${HF_MODELS_DIR}/${MODEL_DIR}/config.json" ]] || { echo "Modell fehlt: ${HF_MODELS_DIR}/${MODEL_DIR}" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:8015" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e PHONON_MODEL="/hf_models/${MODEL_DIR}" \
  -e PHONON_THREADS="${THREADS}" \
  -e HF_HUB_OFFLINE=1 \
  "${IMAGE}"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}  (${MODEL_DIR})"
