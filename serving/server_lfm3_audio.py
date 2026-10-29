#!/usr/bin/env python3
"""OpenAI-kompatibler Sprachserver für LiquidAI/LFM3-Audio-2B (TTS und ASR in einem Modell).
Gleiche API wie die übrigen Adapter, dazu der ASR-Endpunkt der Judge-Kalibrierung:
  POST /v1/audio/speech          – {input, voice, language}  → WAV mono 16-bit, 24 kHz
  POST /v1/audio/transcriptions  – multipart file, language  → {"text": …}
  GET  /v1/voices, /v1/models
  GET  /health

Läuft über transformers mit dem Modellcode des Repos (trust_remote_code). vLLM
kennt LFM3 nicht (Stand 29.10.2026, auch nicht im Hauptzweig), die Karte
verlangt transformers >= 5.20.2.

Stimmen: Das Modell kennt keine benannten Stimmen und klont nicht. Die Karte
steuert Sprache und Geschlecht über den Systemprompt „Perform TTS in <Sprache>
using a <female|male> voice.“ — das sind die beiden Stimmen hier. Gesampelt
wird mit den Kartenwerten (audio_temperature 1,0, top_k 128,
repetition_penalty 1,1, Fenster 4096); Wiederholungen streuen also.

ASR: Systemprompt „Perform ASR in <Sprache>.“, Textausgabe erzwungen
(suppress_mixed, suppress_audio_out). Einen Initial-Prompt wie Whisper kennt
das Modell nicht; ein mitgeschicktes `prompt` wird angenommen, aber nicht
ausgewertet, und der Antwort-Header X-Prompt-Ignored macht das sichtbar.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import time
import wave

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

log = logging.getLogger("lfm3-audio-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("LFM3_AUDIO_PATH", "/hf_models/LiquidAI--LFM3-Audio-2B")
# Jeder Sprach-Frame ist ein Generierungsschritt von 80 ms; 1024 deckt ~80 s.
TTS_MAX_FRAMES = int(os.environ.get("LFM3_TTS_MAX_FRAMES", "1024"))
ASR_MAX_TOKENS = int(os.environ.get("LFM3_ASR_MAX_TOKENS", "1024"))
SPRACHEN = {"de": "German", "en": "English", "fr": "French", "es": "Spanish", "it": "Italian",
            "pt": "Portuguese", "nl": "Dutch", "ja": "Japanese", "zh": "Chinese", "ar": "Arabic"}
STIMMEN = ("female", "male")
RATE = 24_000

app = FastAPI(title="lfm3-audio", version="0.1.0")
model = processor = codec = None


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "female"
    language: str = "de"
    response_format: str = "wav"
    model: str | None = None
    speed: float | None = None


def _sprache(code: str) -> str:
    kurz = (code or "de").split("-")[0].lower()
    if kurz not in SPRACHEN:
        raise HTTPException(400, f"Sprache '{code}' nicht abgebildet ({sorted(SPRACHEN)})")
    return SPRACHEN[kurz]


@app.on_event("startup")
def laden() -> None:
    global model, processor, codec
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    codec = processor.audio_tokenizer.to(device="cuda").eval()
    log.info("LFM3-Audio geladen in %.1fs aus %s", time.time() - t0, MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if model is not None else "loading", "model": MODEL_PATH}


@app.get("/v1/models")
def models() -> dict:
    """Wie ein vLLM-Judge: judge_bench.py und roundtrip_eval.py lesen die Modell-ID hier."""
    return {"object": "list", "data": [{"id": "lfm3-audio-2b", "object": "model"}]}


@app.get("/v1/voices")
def voices() -> dict:
    return {"voices": list(STIMMEN), "languages": sorted(SPRACHEN)}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")
    if req.voice not in STIMMEN:
        raise HTTPException(400, f"Stimme '{req.voice}' gibt es nicht; das Modell kennt {list(STIMMEN)}")
    sprache = _sprache(req.language)
    gespraech = [
        {"role": "system", "content": f"Perform TTS in {sprache} using a {req.voice} voice."},
        {"role": "user", "content": req.input},
        {"role": "assistant", "content": "<|audio_start|>"},
    ]
    inputs = processor.apply_chat_template(gespraech, continue_final_message=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=TTS_MAX_FRAMES, audio_temperature=1.0,
                             audio_top_k=128, audio_repetition_penalty=1.1, audio_repetition_window=4096,
                             return_dict_in_generate=True)
        welle = codec.decode(out.audio_codes[:1, :, :out.audio_lengths[0]]).audio_values[0]
    wall = time.time() - t0
    pcm = np.clip(welle.float().cpu().numpy().reshape(-1), -1.0, 1.0)
    dauer = len(pcm) / RATE
    log.info("tts: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, %s, %s, %d Frames)", len(req.input), dauer,
             wall, wall / max(dauer, 1e-6), req.voice, sprache, int(out.audio_lengths[0]))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Duration": f"{dauer:.3f}", "X-Synthesis-Time": f"{wall:.3f}",
                             "X-Frames-Max-Hit": str(int(out.audio_lengths[0]) >= TTS_MAX_FRAMES)})


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model_name: str | None = Form(None, alias="model"),
                         language: str = Form("de"), prompt: str | None = Form(None),
                         temperature: float = Form(0.0),
                         max_completion_tokens: int | None = Form(None)) -> JSONResponse:
    if model is None:
        raise HTTPException(503, "Modell lädt noch")
    sprache = _sprache(language)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        gespraech = [{"role": "system", "content": f"Perform ASR in {sprache}."},
                     {"role": "user", "content": [{"type": "audio", "audio": tmp.name}]}]
        inputs = processor.apply_chat_template(gespraech, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to("cuda")
        grenze = min(ASR_MAX_TOKENS, max_completion_tokens or ASR_MAX_TOKENS)
        kw = {"do_sample": temperature > 0}
        if temperature > 0:
            kw["temperature"] = float(temperature)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=grenze, suppress_mixed=True, suppress_audio_out=True,
                                 return_dict_in_generate=True, **kw)
    text = processor.decode(out.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return JSONResponse({"text": text}, headers={"X-Prompt-Ignored": "1" if prompt else "0"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8016")))
