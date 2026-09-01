#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für Audio8-TTS-Preview-0.6b (Architektur arktts).

Gleiche API wie die übrigen Adapter (server.py, server_qwen3tts.py,
server_chatterbox.py):
  POST /v1/audio/speech   – {input, voice, language}
  GET  /v1/voices
  GET  /health

Stimmen: 'default' erzeugt ohne Referenz mit der modelleigenen Stimme. Jeder
andere Name klont per Zero-Shot aus /voices/<name>.wav.

Anders als Chatterbox verlangt dieses Modell zur Referenz **zusätzlich das
Transkript** der Aufnahme (siehe Modellkarte: „The reference transcript must
match the spoken content in the reference audio."). Es wird aus
/voices/<name>.txt gelesen; fehlt die Datei, wird die Stimme abgelehnt statt
still ohne Referenz zu erzeugen — sonst misst man später eine andere Stimme,
als der Lauf behauptet.

Das Modell kennt **keinen Sprachparameter**: der Systemprompt lautet fest
„convert the provided text to speech" (processing_arktts.py). Das Feld
'language' wird daher entgegengenommen, aber nicht ausgewertet; die Sprache
ergibt sich aus dem Text und der Referenz. Der Antwort-Header
X-Language-Ignored macht das für Auswertungen sichtbar.
"""

from __future__ import annotations

import io
import logging
import os
import time
import wave
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("audio8-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("AUDIO8_PATH", "/hf_models/Audio8--Audio8-TTS-Preview-0.6b")
VOICES_DIR = Path(os.environ.get("AUDIO8_VOICES_DIR", "/voices"))

# Abtastwerte der Modellkarte. Über die Umgebung überschreibbar, damit ein
# Lauf reproduzierbar dokumentiert werden kann.
TEMPERATURE = float(os.environ.get("AUDIO8_TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("AUDIO8_TOP_P", "0.95"))
TOP_K = int(os.environ.get("AUDIO8_TOP_K", "50"))

# Kontextgrenze aus config.json (max_seq_len 2048, Text und Audio geteilt).
# Der Abstand ist derselbe Gedanke wie im vLLM-Testplan: ein fester Sockel
# plus ein Anteil der Prompt-Länge, damit lange Eingaben nicht auf der
# letzten Position auflaufen.
CONTEXT_LIMIT = int(os.environ.get("AUDIO8_MAX_SEQ_LEN", "2048"))
CONTEXT_MARGIN = 32
CONTEXT_ANTEIL = 0.10

app = FastAPI(title="audio8-tts", version="0.1.0")
model = None
processor = None
sample_rate = 44100


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "default"
    language: str = "de"
    response_format: str = "wav"
    model: str | None = None
    speed: float | None = None


def _referenz(name: str) -> tuple[str, str]:
    """Pfad und Transkript einer Referenzstimme; wirft HTTPException bei Lücken."""
    wav = VOICES_DIR / f"{name}.wav"
    txt = VOICES_DIR / f"{name}.txt"
    if not wav.exists():
        raise HTTPException(400, f"Referenz-Audio fehlt: {wav}")
    if not txt.exists():
        raise HTTPException(
            400,
            f"Referenz-Transkript fehlt: {txt} — dieses Modell braucht zur "
            f"Aufnahme den gesprochenen Text.",
        )
    text = txt.read_text(encoding="utf-8").strip()
    if not text:
        raise HTTPException(400, f"Referenz-Transkript ist leer: {txt}")
    return str(wav), text


@app.on_event("startup")
def load_model() -> None:
    global model, processor, sample_rate
    from transformers import AutoModel, AutoProcessor

    t0 = time.time()
    log.info("Lade Audio8-TTS aus %s ...", MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, dtype=torch.bfloat16)
        .eval()
        .to("cuda")
    )
    sample_rate = int(model.config.codec_sample_rate)
    log.info("Geladen in %.1fs (sr=%d, codebooks=%d)",
             time.time() - t0, sample_rate, model.config.num_codebooks)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading", "model": MODEL_PATH}


@app.get("/v1/voices")
def voices() -> dict:
    """Nur Stimmen melden, die auch benutzbar sind — also mit Transkript."""
    refs = []
    if VOICES_DIR.is_dir():
        refs = sorted(p.stem for p in VOICES_DIR.glob("*.wav")
                      if (VOICES_DIR / f"{p.stem}.txt").exists())
    return {"voices": ["default", *refs], "languages": []}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None or processor is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")

    kwargs: dict = {}
    if req.voice != "default":
        wav_pfad, ref_text = _referenz(req.voice)
        kwargs["reference_audio"] = [wav_pfad]
        kwargs["reference_text"] = [ref_text]

    inputs = processor(text=[req.input], return_tensors="pt", **kwargs)
    inputs = {name: wert.to("cuda") for name, wert in inputs.items()}

    # Belegte Positionen zählen, damit die Generierung nicht in die
    # Kontextgrenze läuft. reference_audio_values ist eine Wellenform, keine
    # Token — sie zählt über die Codec-Bildrate.
    belegt = int(inputs["prefix_input_ids"].shape[1] + inputs["suffix_input_ids"].shape[1])
    if "reference_audio_lengths" in inputs:
        rahmen = model.config.codec_frame_size
        belegt += int(inputs["reference_audio_lengths"].max().item()) // rahmen + 1
    abstand = CONTEXT_MARGIN + int(belegt * CONTEXT_ANTEIL)
    budget = CONTEXT_LIMIT - belegt - abstand
    if budget <= 0:
        raise HTTPException(
            400, f"Eingabe zu lang: {belegt} Positionen belegt von {CONTEXT_LIMIT}")

    t0 = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            return_dict_in_generate=True,
        )
        waveforms, laengen = model.decode_audio(output.codes)
    wall = time.time() - t0

    pcm = waveforms[0, : int(laengen[0])].float().cpu().numpy()
    pcm = np.clip(pcm, -1.0, 1.0)
    duration = len(pcm) / sample_rate
    log.info("synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s, budget=%d)",
             len(req.input), duration, wall, wall / max(duration, 1e-6), req.voice, budget)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Synthesis-Time": f"{wall:.3f}",
            "X-Language-Ignored": "1",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8009")))
