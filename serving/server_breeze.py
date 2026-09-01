#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für BreezeBlue/Breeze-TTS-2.

Gleiche API wie die übrigen Adapter:
  POST /v1/audio/speech   – {input, voice, language}
  GET  /v1/voices
  GET  /health

**Dieses Modell kann kein Deutsch.** Die Modellkarte nennt ausschließlich
Englisch und Chinesisch; „German" kommt darin nicht vor. Ein Lauf gegen den
deutschen Testsatz misst deshalb nicht die Sprachqualität des Modells, sondern
was passiert, wenn man ihm eine Sprache vorlegt, die es nicht gelernt hat. Der
Antwort-Header X-Language-Unsupported hält das an jeder einzelnen Antwort fest,
damit die Einschränkung nicht auf dem Weg in eine Tabelle verlorengeht.

Lizenz: BreezeBlue Research and Non-Commercial License — der Quelltext ist
Apache-2.0, die Gewichte sind es nicht.

Warum ein eigener Adapter und kein Aufsatz auf ihre App (wie bei
server_audio8_onnx.py): Breeze bedient `/v1/audio/speech` zwar selbst, aber mit
Multipart-Formulardaten und einer Referenz als Datei-Upload je Anfrage — nicht
mit unserem JSON. Und sein `/health` meldet kein Modell, das der Evaluator in
die summary.json schreiben könnte.

Stimmen: 'default' erzeugt ohne Referenz aus der Stimmbeschreibung
(BREEZE_INSTRUCTION). Jeder andere Name klont aus /voices/<name>.wav und
verlangt — wie Audio8 — das Transkript in /voices/<name>.txt.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("breeze-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("BREEZE_PATH", "/hf_models/BreezeBlue--Breeze-TTS-2")
VOICES_DIR = Path(os.environ.get("BREEZE_VOICES_DIR", "/voices"))
CODE_DIR = Path(os.environ.get("BREEZE_CODE_DIR", "/opt/breeze"))

# Werte aus infer.py des Herstellers — nicht geraten, sondern von dort
# übernommen, damit unser Lauf ihrem Referenzpfad entspricht.
CFG_SCALE = float(os.environ.get("BREEZE_CFG_SCALE", "1.0"))
MAX_NEW_TOKENS = int(os.environ.get("BREEZE_MAX_NEW_TOKENS", "1500"))
MAX_SEQ_LEN = int(os.environ.get("BREEZE_MAX_SEQ_LEN", "2048"))
REPETITION_PENALTY = float(os.environ.get("BREEZE_REPETITION_PENALTY", "1.1"))
SEED = int(os.environ.get("BREEZE_SEED", "42"))
INSTRUCTION = os.environ.get("BREEZE_INSTRUCTION", "Speak clearly and naturally.")

app = FastAPI(title="breeze-tts", version="0.1.0")

_runtime = None
_model = None
_tokenizer = None
_audio_tokenizer = None
_prepare_inputs = None
_get_template = None
_set_all_seeds = None
# Der Hersteller serialisiert Anfragen ebenso (breeze_infer/api.py): eine
# Laufzeit, eine Anfrage. Der Streaming-Zustand ist nicht wiedereintrittsfähig.
_lock = threading.Lock()


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
        raise HTTPException(400, f"Referenz-Transkript fehlt: {txt}")
    text = txt.read_text(encoding="utf-8").strip()
    if not text:
        raise HTTPException(400, f"Referenz-Transkript ist leer: {txt}")
    return str(wav), text


@app.on_event("startup")
def load_model() -> None:
    global _runtime, _model, _tokenizer, _audio_tokenizer
    global _prepare_inputs, _get_template, _set_all_seeds

    sys.path.insert(0, str(CODE_DIR))
    from breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        set_all_seeds,
        update_generation_config_for_breeze,
    )
    from breeze_infer.templates import get_template, prepare_inputs
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    t0 = time.time()
    log.info("Lade Breeze-TTS-2 aus %s ...", MODEL_PATH)
    # 'eager' statt 'flash_attention_2': flash-attn liegt beim Hersteller nur
    # für sm90 vorgebaut vor, wir fahren sm120. Ihre eigene infer.py nutzt
    # ebenfalls 'eager' — es ist kein Notbehelf, sondern ihr Referenzpfad.
    _tokenizer, _model, _audio_tokenizer = load_runtime(
        Path(MODEL_PATH), device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(_model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        # None statt False: der Hauptschalter ueberstimmt bei False alle
        # Einzelstufen, bei None greifen sie. Aufgeloest ist beides hier
        # identisch (alle Stufen aus) — None ist aber der Vorgabewert von
        # infer.py, und damit bleibt unser Pfad ihrem Referenzpfad gleich,
        # auch wenn sich deren Einzelvorgaben einmal aendern.
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        repetition_penalty=REPETITION_PENALTY,
    )
    _runtime = FastBreezeStreamingRuntime(
        _model, _audio_tokenizer, config, tokenizer=_tokenizer
    )
    _prepare_inputs, _get_template, _set_all_seeds = prepare_inputs, get_template, set_all_seeds
    log.info("Geladen in %.1fs (sr=%d)", time.time() - t0, _runtime.sample_rate)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _runtime is not None else "loading", "model": MODEL_PATH}


@app.get("/v1/voices")
def voices() -> dict:
    refs = []
    if VOICES_DIR.is_dir():
        refs = sorted(p.stem for p in VOICES_DIR.glob("*.wav")
                      if (VOICES_DIR / f"{p.stem}.txt").exists())
    return {"voices": ["default", *refs], "languages": ["en", "zh"],
            "instructs": {"default": INSTRUCTION}}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if _runtime is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")

    anfrage: dict = {"id": "sb-request", "text": req.input,
                     "instruction": INSTRUCTION, "speaker": "S0"}
    vorlage = "tts_instruction"
    if req.voice != "default":
        wav_pfad, ref_text = _referenz(req.voice)
        anfrage["ref_audio_path"] = wav_pfad
        anfrage["ref_text"] = ref_text
        vorlage = "ref_edit_tata"

    t0 = time.time()
    with _lock:
        _set_all_seeds(SEED)
        inputs = _prepare_inputs(
            _tokenizer, _audio_tokenizer, _model, [anfrage], _get_template(vorlage),
            guidance_scale=CFG_SCALE, guidance_scale_ref=None, guidance_scale_ins=None,
        )
        stuecke = [c.audio for c in _runtime.iter_audio_chunks(inputs, request_id="sb-request")]
    wall = time.time() - t0

    if not stuecke:
        raise HTTPException(500, "Modell lieferte kein Audio")
    pcm = np.clip(np.concatenate(stuecke).astype(np.float32), -1.0, 1.0)
    sr = int(_runtime.sample_rate)
    duration = len(pcm) / sr
    log.info("synthesize: %d Zeichen -> %.2fs Audio in %.2fs (RTF %.2f, voice=%s)",
             len(req.input), duration, wall, wall / max(duration, 1e-6), req.voice)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(
        content=buf.getvalue(), media_type="audio/wav",
        headers={"X-Audio-Duration": f"{duration:.3f}",
                 "X-Synthesis-Time": f"{wall:.3f}",
                 "X-Language-Unsupported": "de"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8013")))
