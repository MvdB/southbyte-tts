#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für tencent/AuK und tencent/AuK-Flash.

Gleiche API wie die übrigen Adapter:
  POST /v1/audio/speech   – {input, voice, language}
  GET  /v1/voices
  GET  /health

Beide Varianten teilen denselben Code und unterscheiden sich nur in Checkpoint
und Konfiguration; die Variante wird über AUK_MODEL_DIR gewählt (wie MODEL_DIR
bei server_qwen3tts.py). AuK-Flash erkennt die Laufzeit selbst am Konfignamen
und schaltet dann auf ihr 4-Schritt-Rezept mit abgeschaltetem CFG um.

**Dieses Modell nennt Deutsch nirgends.** Weder die Modellkarten noch der
Quelltext erwähnen es; AuK-Flash deklariert `zh, en`. Ein Lauf gegen den
deutschen Testsatz kann daher eine Sprachlücke messen statt Sprachqualität.
Vor einem vollen Lauf gehört die Sprachprobe gefahren (siehe run_auk.sh).
Der Header X-Language-Undeclared hält den Vorbehalt an jeder Antwort fest.

Zwei Eigenheiten gegenüber allen anderen Adaptern:

1. **Die Zieldauer muss vorgegeben werden.** AuK bestimmt seine Länge nicht
   selbst. Ihr eigener Schätzer (get_gen_duration) leitet sie aus dem
   Byte-Verhältnis von Zieltext zu Referenztranskript ab, skaliert mit der
   Länge der Referenzaufnahme. Deshalb braucht eine Stimme hier zwingend
   <name>.txt — dasselbe Paar wie bei Audio8, aus einem anderen Grund.

   In UTF-8 sind Umlaute und ß zwei Bytes. Am Testsatz gemessen (09.09.2026)
   überschätzt das die Zieldauer je Kategorie um 0,5 bis 3,5 %, im härtesten
   Einzelfall (long-004) um 14 %. Wir übernehmen den Schätzer des Herstellers
   trotzdem: unsere Zahlen sollen auf seinem Referenzpfad stehen, nicht auf
   einer eigenen Korrektur. AUK_DAUER_MODUS='zeichen' rechnet stattdessen über
   Zeichen statt Bytes und macht die Abweichung messbar.

2. **Ein drittes Modell ist Pflicht.** Der Text-Encoder ist Qwen2.5-Omni-3B
   (12 GB); ohne ihn startet nichts. Pfad über AUK_QWEN_DIR.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

log = logging.getLogger("auk-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = Path(os.environ.get("AUK_MODEL_DIR", "/hf_models/tencent--AuK"))
QWEN_DIR = os.environ.get("AUK_QWEN_DIR", "/hf_models/Qwen--Qwen2.5-Omni-3B")
VOICES_DIR = Path(os.environ.get("AUK_VOICES_DIR", "/voices"))
CODE_DIR = os.environ.get("AUK_CODE_DIR", "/opt/auk/src")

# Abtastwerte aus der CLI des Herstellers (src/auk/infer): nfe 32, cfg 2.0,
# sway -1.0. Bei AuK-Flash überstimmt die Laufzeit sie selbst.
NFE = int(os.environ.get("AUK_NFE", "32"))
CFG = float(os.environ.get("AUK_CFG", "2.0"))
SWAY = float(os.environ.get("AUK_SWAY", "-1.0"))
SEED = int(os.environ.get("AUK_SEED", "42"))
DAUER_MODUS = os.environ.get("AUK_DAUER_MODUS", "bytes")  # bytes | zeichen

# Vorlage für Zero-Shot-TTS, wörtlich aus docs/COOKBOOK.md 1.1.
VORLAGE = 'Say the following with the same voice: "{text}"'

app = FastAPI(title="auk-tts", version="0.1.0")
_engine = None
_sr = 24000
# Der Streaming-/Diffusionszustand ist nicht wiedereintrittsfähig; der
# Hersteller serialisiert in seiner Gradio-Oberfläche ebenso.
_lock = threading.Lock()


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4096)
    voice: str = "de_f1"
    language: str = "de"
    response_format: str = "wav"
    model: str | None = None
    speed: float | None = None


def _referenz(name: str) -> tuple[str, str]:
    """WAV und Transkript einer Stimme. Beides Pflicht — das Transkript geht
    als ref_text in die Dauerschätzung ein, ohne es wäre die Länge geraten."""
    wav = VOICES_DIR / f"{name}.wav"
    txt = VOICES_DIR / f"{name}.txt"
    if not wav.exists():
        raise HTTPException(400, f"Referenz-Audio fehlt: {wav}")
    if not txt.exists():
        raise HTTPException(
            400, f"Referenz-Transkript fehlt: {txt} — AuK schätzt daraus die Zieldauer.")
    text = txt.read_text(encoding="utf-8").strip()
    if not text:
        raise HTTPException(400, f"Referenz-Transkript ist leer: {txt}")
    return str(wav), text


def _ref_sekunden(wav_pfad: str) -> float:
    """Laenge der Referenzaufnahme.

    Der Hersteller nimmt dafuer torchaudio.info(). Das gibt es in unserem
    torchaudio 2.11 nicht mehr — eine Folge davon, dass wir ihren Pin
    torch>=2.7,<2.8 umgehen muessen, weil sm120 das NGC-torch 2.13 braucht.
    soundfile liefert dieselbe Angabe und ist im Basis-Image vorhanden.
    """
    import soundfile as sf

    info = sf.info(wav_pfad)
    return info.frames / info.samplerate


def _zieldauer(wav_pfad: str, ref_text: str, ziel_text: str) -> float:
    """Zieldauer in Sekunden.

    Die Formel ist woertlich die des Herstellers (get_gen_duration in
    infer_auk.py): Laenge der Referenz, skaliert mit dem Verhaeltnis der
    UTF-8-Byte-Laengen. Nachgebaut statt aufgerufen, weil ihre Fassung an
    torchaudio.info haengt (siehe _ref_sekunden) — die Rechnung selbst ist
    unveraendert, damit unsere Zahlen auf ihrem Referenzpfad stehen.

    DAUER_MODUS='zeichen' rechnet ueber Zeichen statt Bytes und macht den
    Umlaut-Bias messbar (am Testsatz +0,5 bis +3,5 % je Kategorie, Spitze
    long-004 mit +14 %).
    """
    ref_s = _ref_sekunden(wav_pfad)
    if DAUER_MODUS == "zeichen":
        return ref_s * len(ziel_text) / max(1, len(ref_text))
    return ref_s * len(ziel_text.encode("utf-8")) / max(1, len(ref_text.encode("utf-8")))


def _torchaudio_load_ersetzen() -> None:
    """torchaudio.load auf soundfile umlegen.

    In torchaudio 2.11 delegiert load() an torchcodec. Das ist im Image nicht
    vorhanden, und nachinstallieren hilft nicht: torchcodec 0.16 verlangt die
    FFmpeg-4-Bibliotheken (libavutil.so.56), das Image hat neuere. Beides ist
    Folge davon, dass wir den Pin torch>=2.7,<2.8 des Herstellers umgehen
    muessen — sm120 braucht das NGC-torch 2.13.

    Betroffen ist genau eine Stelle im Fremdcode, die wir benutzen:
    AukInfer._load_audio laedt damit die Referenzaufnahme. Nachgebaut wird die
    alte Signatur von torchaudio.load: Tensor [Kanaele, Rahmen] in float32
    plus Abtastrate.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def laden(pfad, *args, **kwargs):
        daten, sr = sf.read(str(pfad), dtype="float32", always_2d=True)
        return torch.from_numpy(daten.T).contiguous(), int(sr)

    torchaudio.load = laden
    log.info("torchaudio.load auf soundfile umgelegt (torchcodec fehlt)")


@app.on_event("startup")
def load_model() -> None:
    global _engine, _sr

    sys.path.insert(0, CODE_DIR)
    _torchaudio_load_ersetzen()
    from auk.infer.infer_auk import AukInfer

    ckpt = next((MODEL_DIR / n for n in ("auk_base.safetensors", "auk_flash.safetensors")
                 if (MODEL_DIR / n).is_file()), None)
    if ckpt is None:
        raise RuntimeError(f"Kein auk_*.safetensors in {MODEL_DIR}")
    cfg = MODEL_DIR / "config.yaml"
    if not cfg.is_file():
        raise RuntimeError(f"config.yaml fehlt in {MODEL_DIR}")

    t0 = time.time()
    log.info("Lade AuK aus %s (Encoder: %s) ...", ckpt.name, QWEN_DIR)
    _engine = AukInfer(str(cfg), str(ckpt), dtype="bf16", qwen_path=QWEN_DIR)
    _sr = int(getattr(_engine, "target_sample_rate", 24000))
    log.info("Geladen in %.1fs (sr=%d, flash=%s)",
             time.time() - t0, _sr, getattr(_engine, "is_flash", False))


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _engine is not None else "loading",
            "model": str(MODEL_DIR),
            "flash": bool(getattr(_engine, "is_flash", False)) if _engine else None,
            "dauer_modus": DAUER_MODUS}


@app.get("/v1/voices")
def voices() -> dict:
    refs = []
    if VOICES_DIR.is_dir():
        refs = sorted(p.stem for p in VOICES_DIR.glob("*.wav")
                      if (VOICES_DIR / f"{p.stem}.txt").exists())
    return {"voices": refs, "languages": ["en", "zh"]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if _engine is None:
        raise HTTPException(503, "Modell lädt noch")
    if req.response_format != "wav":
        raise HTTPException(400, f"Nur 'wav' unterstützt, nicht '{req.response_format}'")

    wav_pfad, ref_text = _referenz(req.voice)
    sekunden = _zieldauer(wav_pfad, ref_text, req.input)
    nachricht = [{"role": "user", "content": [
        {"type": "text", "text": VORLAGE.format(text=req.input)},
        {"type": "audio", "audio": wav_pfad},
    ]}]

    t0 = time.time()
    with _lock:
        audio, sr = _engine.generate(nachricht, gen_seconds=sekunden,
                                     nfe=NFE, cfg_strength=CFG,
                                     sway_sampling_coef=SWAY, seed=SEED)
    wall = time.time() - t0

    pcm = audio.squeeze().to("cpu").float().numpy()
    pcm = np.clip(pcm, -1.0, 1.0)
    dauer = len(pcm) / sr
    log.info("synthesize: %d Zeichen -> Ziel %.2fs, erhalten %.2fs in %.2fs (RTF %.2f, voice=%s)",
             len(req.input), sekunden, dauer, wall, wall / max(dauer, 1e-6), req.voice)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return Response(content=buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Duration": f"{dauer:.3f}",
                             "X-Synthesis-Time": f"{wall:.3f}",
                             "X-Target-Duration": f"{sekunden:.3f}",
                             "X-Language-Undeclared": "de"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8014")))
