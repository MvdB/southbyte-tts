#!/usr/bin/env bash
# Komplette Eval-Suite: alle Modell/Stimme-Konfigurationen seriell,
# je N=3 gegen den Whisper-Judge (Port 8007) + Rescoring mit dem
# Zweit-Judge Voxtral-Mini (Port 8006). Voraussetzungen: beide Judges laufen
# bereits — serving/run_whisper_judge.sh und serving/run_voxtral_mini_judge.sh;
# es wird immer nur EIN TTS-Container gleichzeitig gestartet (Unified
# Memory). Einzelne Konfigurationen duerfen fehlschlagen (kein set -e) —
# am Ende steht eine Zusammenfassung in $LOG.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
# SUITE_DATE erlaubt einen abweichenden Ergebnis-Praefix (z. B. das Morgen-
# Datum fuer einen Nachtlauf, damit die heutigen Ergebnisse nicht
# ueberschrieben werden, solange der neue Lauf nicht durch ist).
DATE=${SUITE_DATE:-$(date +%F)}
STT=http://127.0.0.1:8007   # Whisper large-v3 (Haupt-Judge)
STT2=http://127.0.0.1:8006  # Voxtral-Mini-3B (Zweit-Judge)
LOG="results/suite_${DATE}.log"
mkdir -p results

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Preflight: ohne Haupt-Judge ist die ganze Nacht verloren — sofort und
# laut abbrechen. Der Zweit-Judge ist nur fuers Rescoring noetig (Warnung).
if ! curl -sf --max-time 10 "$STT/v1/models" >/dev/null; then
  log "ABBRUCH: Haupt-Judge (Whisper) auf $STT nicht erreichbar"
  exit 1
fi
curl -sf --max-time 10 "$STT2/v1/models" >/dev/null \
  || log "WARNUNG: Zweit-Judge auf $STT2 nicht erreichbar — Rescoring wird fehlschlagen"

wait_ready() { # port
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}

run_config() { # name port voice
  local name=$1 port=$2 voice=$3 out="results/${DATE}_suite_${1}"
  # Stirbt der Haupt-Judge mitten in der Nacht, ist jede weitere
  # Konfiguration wertlos — dann sofort raus statt stundenlang Fehler loggen.
  if ! curl -sf --max-time 10 "$STT/v1/models" >/dev/null; then
    log "ABBRUCH: Haupt-Judge weggebrochen (vor $name)"
    exit 1
  fi
  if ! wait_ready "$port"; then log "FEHLER: $name — Server auf :$port nicht bereit, uebersprungen"; return 1; fi
  log "START $name (voice=$voice)"
  if python3 eval/roundtrip_eval.py --testset testset/german_tts_v1.jsonl \
       --tts "http://127.0.0.1:$port" --stt "$STT" --voice "$voice" \
       --repeats 3 --out "$out" >> "$LOG" 2>&1; then
    python3 eval/rescore_with_judge.py --stt2 "$STT2" "$out" >> "$LOG" 2>&1 \
      || log "WARNUNG: $name — Rescoring fehlgeschlagen"
    log "FERTIG $name: $(python3 -c "import json; s=json.load(open('$out/summary.json')); print(f\"WER {s.get('wer_capped_mean', s['wer_mean'])} (Cap) / {s['wer_mean']} (roh), ASR-Runaways: {s.get('n_asr_runaway', 0)}\")" 2>/dev/null)"
  else
    log "FEHLER: $name — Eval fehlgeschlagen"
  fi
}

# Ausgangszustand: KEIN TTS-Container darf laufen (Unified Memory —
# beide Judges + ein TTS passen, mehr nicht).
log "stoppe evtl. laufende TTS-Container"
docker stop qwen3-tts chatterbox-tts voxcpm2 voxtral-tts magpie-tts audio8-tts audio8-onnx >> "$LOG" 2>&1 || true

# Magpie-Basisimage aktualisieren (server.py ist eingebacken; bei
# unveraendertem Code ist das ein Cache-Hit in Sekunden). Danach das
# TN-Image im Hintergrund darauf aufbauen (Quell-Build, ~30-40 min —
# wird erst fuer die letzte Konfiguration gebraucht).
log "Magpie-Basisimage-Build (Cache-Hit, falls unveraendert)"
( cd serving && docker build -t spark-magpie-tts:v1 . ) >> "$LOG" 2>&1 \
  || log "WARNUNG: Magpie-Basisimage-Build fehlgeschlagen — laufe mit dem alten Image weiter"
log "TN-Image-Build startet im Hintergrund"
( cd serving && docker build -t spark-magpie-tts:v1-tn -f Dockerfile.tn . \
    > "$REPO/results/tn_build_${DATE}.log" 2>&1 ) &
TN_BUILD_PID=$!

# ── Qwen3-TTS CustomVoice: 3 Stimmen ueber einen Server ─────────────────────
./serving/run_qwen3tts.sh >> "$LOG" 2>&1
run_config qwen-cv-serena  8002 serena
run_config qwen-cv-aiden   8002 aiden
run_config qwen-cv-unclefu 8002 uncle_fu

# ── Qwen3-TTS VoiceDesign: vier Stimmbeschreibungen ueber einen Server ──────
#    (Presets werden je Request gewaehlt, der instruct-Text landet in der
#     summary.json — ohne ihn waere ein VoiceDesign-Lauf nicht reproduzierbar.)
MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./serving/run_qwen3tts.sh >> "$LOG" 2>&1
run_config qwen-vd-de-female-news 8002 de_female_news
run_config qwen-vd-de-male-news   8002 de_male_news
run_config qwen-vd-de-female-calm 8002 de_female_calm
run_config qwen-vd-de-male-young  8002 de_male_young
run_config qwen-vd-de-male-coach  8002 de_male_coach
run_config qwen-vd-de-female-coach 8002 de_female_coach
docker stop qwen3-tts >> "$LOG" 2>&1

# ── Chatterbox: default + de_f1 ─────────────────────────────────────────────
VOICES_DIR="$REPO/voices" ./serving/run_chatterbox.sh >> "$LOG" 2>&1
run_config chatterbox-de-f1   8003 de_f1
run_config chatterbox-default 8003 default
docker stop chatterbox-tts >> "$LOG" 2>&1

# ── Audio8-TTS 0.6b: default + de_f1 ────────────────────────────────────────
#    Zum Klonen braucht dieses Modell neben voices/de_f1.wav auch das
#    Transkript voices/de_f1.txt; ohne das lehnt der Adapter die Stimme ab.
VOICES_DIR="$REPO/voices" ./serving/run_audio8.sh >> "$LOG" 2>&1
run_config audio8-de-f1   8009 de_f1
run_config audio8-default 8009 default
docker stop audio8-tts >> "$LOG" 2>&1

# ── Audio8 0.1B ONNX INT8 (CPU, kein --gpus) ────────────────────────────────
#    Stimmen sind hier registrierte Codesaetze, keine Dateien. Die
#    Registrierung ist idempotent (overwrite) und kostet wenige Sekunden.
./serving/run_audio8_onnx.sh >> "$LOG" 2>&1
if wait_ready 8012; then
  curl -sf --max-time 300 http://127.0.0.1:8012/api/voices/register \
    -F "audio=@$REPO/voices/de_f1.wav" -F "text=$(cat "$REPO/voices/de_f1.txt")" \
    -F 'name=de_f1' -F 'overwrite=true' >> "$LOG" 2>&1 \
    || log "WARNUNG: audio8onnx — Registrierung von de_f1 fehlgeschlagen"
  docker exec -e PYTHONPATH=/opt/audio8onnx audio8-onnx python3 \
    /opt/audio8onnx/scripts/register_default_voice.py \
    --model-dir /hf_models/Audio8--audio8-TTS-0.1B-ONNX-INT8 \
    --voices-dir /voices_onnx --overwrite >> "$LOG" 2>&1 \
    || log "WARNUNG: audio8onnx — Registrierung der Standardstimme fehlgeschlagen"
  curl -sf --max-time 120 -X POST http://127.0.0.1:8012/api/runtime/reload >> "$LOG" 2>&1 || true
  run_config audio8onnx-de-f1   8012 de_f1
  run_config audio8onnx-default 8012 default
else
  log "FEHLER: audio8onnx — Dienst auf :8012 nicht bereit, uebersprungen"
fi
docker stop audio8-onnx >> "$LOG" 2>&1

# ── VoxCPM2 ─────────────────────────────────────────────────────────────────
./serving/run_voxcpm.sh >> "$LOG" 2>&1
run_config voxcpm2-design 8004 design
docker stop voxcpm2 >> "$LOG" 2>&1

# ── Voxtral: beide deutschen Preset-Stimmen ─────────────────────────────────
./serving/run_voxtral_tts.sh >> "$LOG" 2>&1
run_config voxtral-de-female 8005 de_female
run_config voxtral-de-male   8005 de_male
docker stop voxtral-tts >> "$LOG" 2>&1

# ── Magpie ohne TN ──────────────────────────────────────────────────────────
./serving/run_server.sh >> "$LOG" 2>&1
run_config magpie-sofia 8001 sofia
docker stop magpie-tts >> "$LOG" 2>&1

# ── Magpie mit TN (wartet ggf. auf den Image-Build) ─────────────────────────
log "warte auf TN-Image-Build (PID $TN_BUILD_PID)"
if wait "$TN_BUILD_PID"; then
  IMAGE=spark-magpie-tts:v1-tn ./serving/run_server.sh >> "$LOG" 2>&1
  # Gegenprobe: ohne nemo_text_processing waere apply_TN ein stiller No-op und
  # der Lauf eine blosse Kopie des Laufs ohne TN.
  if wait_ready 8001 && [ "$(curl -s http://127.0.0.1:8001/health | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tn"))')" = "True" ]; then
    run_config magpie-tn-sofia 8001 sofia
  else
    log "FEHLER: magpie-tn — /health meldet kein TN, Lauf waere wertlos"
  fi
  docker stop magpie-tts >> "$LOG" 2>&1
else
  log "FEHLER: TN-Image-Build fehlgeschlagen (s. results/tn_build_${DATE}.log) — magpie-tn uebersprungen"
fi

log "SUITE KOMPLETT"
