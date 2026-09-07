#!/usr/bin/env python3
"""OpenAI-kompatibler TTS-Server für Audio8 0.1B ONNX INT8.

Dieses Modell bringt — anders als alle übrigen — seine **eigene Laufzeit** mit
(Apache-2.0, github.com/Audio8-AI/Audio8_TTS, Verzeichnis
`onnx_runtime_0_1b_int8`). Sie läuft auf ONNX Runtime mit dem
CPUExecutionProvider, nicht auf torch, und bedient `/v1/audio/speech` bereits
selbst.

Nachgebaut wird deshalb nichts. Dieser Adapter lädt die fremde FastAPI-App und
hängt nur die zwei Endpunkte an, die unser Vertrag zusätzlich verlangt:

  GET /health      – der Evaluator liest daraus 'model'; fehlt er, landet "?"
                     in der summary.json und make_docs.py benennt die Seite falsch
  GET /v1/voices   – Stimmenliste in unserer Form

Der Synthesepfad bleibt unangetastet der ihre. Sein Stimmenbegriff ist ein
anderer als bei server_audio8.py: eine Stimme ist hier kein WAV im Dateisystem,
sondern ein **registrierter Codesatz** unter ARKTTS_VOICES_DIR, den man vorher
über POST /api/voices/register aus Aufnahme und Transkript erzeugt.

Das Modell kennt keinen Sprachparameter (siehe server_audio8.py); 'language'
wird von der fremden Anfrageklasse schlicht ignoriert.
"""

from __future__ import annotations

import os

from arktts_runtime.service import app, require_runtime


@app.get("/health")
def health_kompatibel() -> dict:
    """Unsere Form von /api/health — 'model' ist das Feld, das der Evaluator liest.

    Gemeldet wird der **Modellpfad**, nicht die model_id aus dem Manifest. Zwei
    Gruende: erstens ist der Pfad die Konvention aller lokalen Adapter, aus der
    make_docs.py mit `_hf()` die HF-Adresse ableitet. Zweitens weicht die
    Manifest-Kennung vom tatsaechlichen Repository ab
    (`Audio8/Audio8-TTS-Preview-0.1B-ONNX-INT8` gegen `Audio8/audio8-TTS-0.1B-ONNX-INT8`);
    wuerde sie durchgereicht, verlinkte die Ergebnisseite auf ein Repository,
    das es nicht gibt.
    """
    obj = require_runtime()
    return {
        "status": "ok",
        "model": os.environ.get("ARKTTS_MODEL_DIR", str(obj.model_dir)),
        "model_id_manifest": obj.manifest.get("model_id"),
        "precision": obj.precision,
        "codec_precision": obj.codec_precision,
        "provider": obj.slow.get_providers(),
    }


@app.get("/v1/voices")
def voices_kompatibel() -> dict:
    """Nur die Namen — 'instructs' bleibt leer, das Modell hat keine Prompt-Stimmen."""
    return {
        "voices": [v.get("name") for v in require_runtime().voices.list() if v.get("name")],
        "languages": [],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8012")))
