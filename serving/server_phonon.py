#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für gradium/phonon-40m (de und multilingual).
Gleiche API wie die übrigen Adapter:
  POST /v1/audio/speech   – {input, voice, language}
  GET  /v1/voices
  GET  /health

Läuft über ptts, die Laufzeit des Herstellers (Rust, GGUF-Gewichte, CPU) —
nicht über transformers. Der Unterschied ist für diesen Testsatz tragend: ptts
normalisiert den Text vor der Tokenisierung (Zahlen, Beträge, Abkürzungen,
E-Mail, URL), der transformers-Weg laut Modellkarte nicht. Gemessen wird also
das Produkt, wie es ausgeliefert wird.

Alles kommt aus dem Modellspeicher, nichts aus dem Netz: ptts lädt von einem
lokalen Pfad nie nach, und die Stimmen kommen aus dem lokalen Katalog
gradium--phonon-voices (voices.jsonl). Neben dem Katalog gibt es die
eingebaute Stimme 'default' (default-voice.safetensors im Modellordner).
Eine Stimme, die weder dort noch im Katalog steht, ist ein Fehler (HTTP 400),
kein stiller Rückfall auf die Standardstimme. Klonen kann phonon-40m nicht.

Abtastung wie in der Laufzeit voreingestellt: Temperatur 0,3 und ihr fester
Seed. Der legt nur den Startzustand beim Laden fest — derselbe Text dreimal
angefragt ergibt drei verschiedene Aufnahmen (geprueft 29.10.2026), die
Wiederholungen im Rundlauf messen also echte Streuung.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("phonon-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("PHONON_MODEL", "/hf_models/gradium--phonon-40m-de")
VOICE_CATALOG = Path(os.environ.get("PHONON_VOICES", "/hf_models/gradium--phonon-voices/voices.jsonl"))
QUANT = os.environ.get("PHONON_QUANT", "q8")              # Voreinstellung der Modellkarte
DEFAULT_VOICE = os.environ.get("PHONON_VOICE", "default")  # die eingebaute Stimme des Checkpoints
THREADS = int(os.environ.get("PHONON_THREADS", "0"))      # 0: Voreinstellung der Laufzeit
# Steuert die Textnormalisierung und wird beim Laden festgelegt; ein Server spricht eine Sprache.
LANG = os.environ.get("PHONON_LANG", "de")

app = FastAPI(title="phonon-tts", version="0.1.0")
tts = None
katalog: dict[str, dict] = {}


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = ""
    language: str = "de"
    response_format: str = "wav"
    model: str | None = None
    speed: float | None = None


@app.on_event("startup")
def load_model() -> None:
    global tts, katalog
    import ptts

    if THREADS > 0:
        ptts.set_num_threads(THREADS)
    t0 = time.time()
    log.info("Lade phonon aus %s (quant=%s, ptts %s) ...", MODEL_PATH, QUANT, ptts.__version__)
    tts = ptts.TTS(MODEL_PATH, quant=QUANT, lang=LANG, voice_catalog=str(VOICE_CATALOG))
    katalog = {}
    for zeile in VOICE_CATALOG.read_text(encoding="utf-8").splitlines():
        if zeile.strip():
            v = json.loads(zeile)
            katalog[v["name"].lower()] = v
    log.info("Geladen in %.1fs (sr=%d, quant=%s, device=%s, Sprachen=%s, %d Katalogstimmen)",
             time.time() - t0, tts.sample_rate, tts.quant, tts.device, tts.languages, len(katalog))


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if tts is not None else "loading", "model": MODEL_PATH,
            "quant": tts.quant if tts else None, "lang": LANG}


@app.get("/v1/voices")
def voices() -> dict:
    """Eingebaute Stimmen, dann die Katalogstimmen in der Sprache dieses Servers."""
    namen = sorted(v["name"] for v in katalog.values() if v.get("language") == LANG)
    return {"voices": [*(tts.voices if tts else []), *namen], "languages": [LANG]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if tts is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")
    sprache = req.language.split("-")[0].lower()
    if sprache != LANG:
        raise HTTPException(400, f"Dieser Server normalisiert '{LANG}', nicht '{req.language}' (PHONON_LANG)")
    stimme = req.voice or DEFAULT_VOICE
    if stimme not in tts.voices and stimme.lower() not in katalog:
        raise HTTPException(400, f"Stimme '{stimme}' ist weder eingebaut ({tts.voices}) noch im "
                                 f"lokalen Katalog {VOICE_CATALOG}")
    t0 = time.time()
    pcm = np.clip(tts.synth(req.input, voice=stimme), -1.0, 1.0)
    wall = time.time() - t0
    duration = len(pcm) / tts.sample_rate
    log.info("synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s, lang=%s)",
             len(req.input), duration, wall, wall / max(duration, 1e-6), stimme, sprache)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts.sample_rate)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"X-Audio-Duration": f"{duration:.3f}", "X-Synthesis-Time": f"{wall:.3f}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8015")))
