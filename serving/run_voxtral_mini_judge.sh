#!/usr/bin/env bash
# Zweit-Judge (Port 8006): Voxtral-Mini-3B-2507.
#
# Er ist NICHT der bessere Judge — auf dem Kalibrier-Audio schreibt er 67 % der
# gesprochenen Zahlwörter in Ziffern zurück, gegenüber 28 % bei Whisper, und
# kommt auf 0.208 statt 0.154 WER (siehe run_voxtral_realtime_judge.sh für die
# vollständige Tabelle). Seine Aufgabe ist eine andere: dieselben Clips durch
# ein zweites, unabhängiges ASR-Modell zu schicken. Die Spanne zwischen beiden
# ist das Unsicherheitsband der Messung.
#
# Er wird gebraucht für eval/rescore_with_judge.py — und ohne dessen
# rescore_judge2.json nimmt southbyte-results/feeds.py einen Lauf gar nicht
# erst auf (load_tts liest nur Verzeichnisse mit dieser Datei). Ein Lauf ohne
# Rescoring erscheint auf der Ergebnisseite also schlicht nicht.
#
# Bis 2026-09-01 gab es dieses Skript nicht und der Judge wurde von Hand
# gestartet; damit war jeder veröffentlichte Lauf an ein nicht festgehaltenes
# Kommando gebunden.
#
# enforce_eager wie bei den übrigen Voxtral-Modellen: auf GB10 haben
# CUDA-Graphen bei dieser Familie schon Audio korrumpiert, und bei einem Judge
# wäre ein solcher Fehler nicht als solcher erkennbar.
set -euo pipefail

HF_MODELS_DIR="${HF_MODELS_DIR:-$HOME/hf_models}"
CONTAINER_NAME="${CONTAINER_NAME:-voxtral-mini-judge}"
HOST_PORT="${HOST_PORT:-8006}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.25.1}"
GPU_UTIL="${GPU_UTIL:-0.18}"
MAX_LEN="${MAX_LEN:-8192}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' existiert -> wird entfernt."
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all --shm-size=4g \
  -p "${HOST_PORT}:${HOST_PORT}" \
  -v "${HF_MODELS_DIR}:/hf_models:ro" \
  -e VLLM_DISABLE_COMPILE_CACHE=1 \
  --entrypoint bash "${IMAGE}" \
  -c "pip install --quiet 'vllm[audio]' && exec vllm serve \
        /hf_models/mistralai--Voxtral-Mini-3B-2507 \
        --served-model-name voxtral-mini-3b --port ${HOST_PORT} \
        --tokenizer-mode mistral \
        --gpu-memory-utilization ${GPU_UTIL} --max-model-len ${MAX_LEN} \
        --max-num-seqs 4 --enforce-eager"

echo "Gestartet: http://0.0.0.0:${HOST_PORT}"
echo "  Logs : docker logs -f ${CONTAINER_NAME}"
echo "  Test : curl -s http://127.0.0.1:${HOST_PORT}/v1/models"
