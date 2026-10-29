#!/usr/bin/env python3
"""ASR auf den Besprechungsszenen der Diarisierung — WER je Szene gegen den bekannten Text.

Die vier Szenen aus eval/diarisierung/baue_testaudio.py (testset/diarisierung/*.wav)
sind aus TTS-Stimmen gebaut, ihr Inhalt ist also bekannt: die Beiträge in Reihenfolge.
Anders als die Judge-Kalibrierung (Einzelsätze, judge_bench.py) misst das hier
mehrere Sprecher, Wechsel, kurze Einwürfe und Überlappung in einer Aufnahme — näher
an dem, was southbyte-transkript transkribiert.

Die Referenzen sind mit ae/oe/ue/ss geschrieben; verglichen wird deshalb nach
normalize() aus roundtrip_eval mit Umlauten auf ae/oe/ue und ß auf ss gefaltet.
In der Überlappungsszene ist die Wortfolge nicht eindeutig (zwei Sprecher zugleich);
ihre WER steht daneben, zählt aber nicht in den Mittelwert ohne Überlappung.

Aufruf:
  python3 eval/asr_szenen.py --stt http://127.0.0.1:8007 --label whisper-large-v3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
sys.path.insert(0, str(HIER / "diarisierung"))
from baue_testaudio import SZENEN  # noqa: E402
from roundtrip_eval import ASR_VERBATIM_PROMPT, normalize, stt_model_id, wer  # noqa: E402

TESTSET = HIER.parent / "testset" / "diarisierung"
UEBERLAPPUNG = {"2sprecher_ueberlappung"}
_FALTEN = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def vergleichbar(text: str) -> str:
    return normalize(text).translate(_FALTEN)


def transkribieren(url: str, modell: str, wav: bytes) -> str:
    r = requests.post(f"{url}/v1/audio/transcriptions",
                      files={"file": ("szene.wav", wav, "audio/wav")},
                      data={"model": modell, "language": "de", "temperature": 0.0,
                            "prompt": ASR_VERBATIM_PROMPT},
                      timeout=600)
    r.raise_for_status()
    return r.json()["text"].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=str(HIER.parent / "results" / "asr_szenen"))
    args = ap.parse_args()
    modell = stt_model_id(args.stt)
    zeilen = []
    for name, szene in SZENEN.items():
        wav = (TESTSET / f"{name}.wav").read_bytes()
        ref = " ".join(t[1] for t in szene["turns"])
        t0 = time.perf_counter()
        hyp = transkribieren(args.stt, modell, wav)
        dt = time.perf_counter() - t0
        w = wer(vergleichbar(hyp), vergleichbar(ref))
        zeilen.append({"szene": name, "wer": round(w, 4), "sekunden": round(dt, 2),
                       "ueberlappung": name in UEBERLAPPUNG, "transkript": hyp, "referenz": ref})
        print(f"  {name:24} WER {w:.3f}  ({dt:.1f}s)")
    ohne = [z["wer"] for z in zeilen if not z["ueberlappung"]]
    erg = {"label": args.label, "modell": modell, "stt_url": args.stt,
           "wer_mittel_ohne_ueberlappung": round(sum(ohne) / len(ohne), 4),
           "wer_ueberlappung": next(z["wer"] for z in zeilen if z["ueberlappung"]),
           "szenen": zeilen}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(erg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.label}: WER ohne Überlappung {erg['wer_mittel_ohne_ueberlappung']:.3f}, "
          f"Überlappung {erg['wer_ueberlappung']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
