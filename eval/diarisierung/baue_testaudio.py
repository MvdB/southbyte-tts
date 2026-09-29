#!/usr/bin/env python3
"""Baut synthetische Mehrsprecher-Dialoge samt exakter Wahrheit (RTTM).

Warum synthetisch: Diarisierung beantwortet "wer spricht wann", und das laesst
sich nur gegen eine bekannte Wahrheit messen. Bei einer echten Aufnahme muesste
jemand jede Sprecherwechsel-Sekunde von Hand labeln (rund eine Stunde Arbeit je
zehn Minuten Audio), und echte Stimmen von Kollegen waeren personenbezogene
Daten. Hier entsteht die Wahrheit beim Bauen: wir wissen auf die Millisekunde,
welcher Sprecher wann eingesetzt wird, weil wir die Segmente selbst aneinander
legen.

DER PREIS, und er gehoert in jeden Bericht: synthetisches Audio ist sauberer
als jede echte Besprechung — ein Raum, keine Nebengeraeusche, kein Husten, kein
Stuhlruecken, und Ueberlappung nur dort, wo wir sie absichtlich bauen. Eine DER
auf diesem Material ist eine Untergrenze des Fehlers, keine Feldmessung.

Alle Sprecher kommen aus DEMSELBEN Adapter (Qwen3-TTS, Port 8002) mit
verschiedenen Voice-Design-Stimmen. Das haelt Aufnahmebedingung, Abtastrate und
Codec konstant — der Unterschied zwischen den Spuren ist dann die Stimme und
nicht das Modell.

Ausgabe je Szene:
  <ziel>/<szene>.wav    16 kHz mono (Standard fuer Diarisierung)
  <ziel>/<szene>.rttm   Wahrheit im NIST-Format, das die NeMo-Bewertung liest
  <ziel>/manifest.jsonl eine Zeile je Szene fuer e2e_diarize_speech.py

Aufruf (Adapter muss laufen: serving/run_qwen3tts.sh):
    python3 eval/diarisierung/baue_testaudio.py
    python3 eval/diarisierung/baue_testaudio.py --szenen 2sprecher_klar --tts http://localhost:8002
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

HIER = Path(__file__).resolve().parent
REPO = HIER.parent.parent
ZIEL = REPO / "testset" / "diarisierung"
RATE = 16000          # Diarisierungsmodelle arbeiten auf 16 kHz
STILLE_KURZ = 0.25    # Sekunden zwischen zwei Beitraegen desselben Sprechers
STILLE_WECHSEL = 0.6  # Sekunden beim Sprecherwechsel — realistische Gespraechspause

# Die Sprecher sind die BENANNTEN Stimmen des Qwen3-TTS-CustomVoice-Adapters
# (GET /v1/voices), nicht Voice-Design-Beschreibungen: feste Stimmen sind fuer
# Diarisierung der sauberere Weg, weil derselbe Sprecher ueber alle Szenen
# derselbe bleibt. Vier ausgewaehlt, abwechselnd weiblich und maennlich; am
# selben Satz gemessen sprechen sie unterschiedlich schnell (3,4 bis 5,8 s),
# sind also auch prosodisch auseinanderzuhalten.
SPRECHER = {
    "A": "serena",
    "B": "dylan",
    "C": "vivian",
    "D": "ryan",
}

# Szenen bewusst gestuft: erst der leichte Fall, dann die harten. Was eine
# Diarisierung schwierig macht, sind kurze Beitraege, viele Sprecher und
# Ueberlappung — genau in dieser Reihenfolge.
SZENEN: dict[str, dict] = {
    "2sprecher_klar": {
        "beschreibung": "Zwei Sprecher, lange Beitraege, saubere Pausen. Der leichte Fall.",
        "turns": [
            ("A", "Guten Morgen, ich fasse kurz den Stand aus der letzten Woche zusammen."),
            ("B", "Sehr gerne. Mich interessiert vor allem, ob der Zeitplan noch traegt."),
            ("A", "Der Zeitplan traegt. Wir sind bei den ersten beiden Punkten sogar frueher fertig geworden als geplant."),
            ("B", "Das freut mich zu hoeren. Dann koennen wir den dritten Punkt vorziehen."),
        ],
    },
    "2sprecher_kurz": {
        "beschreibung": "Zwei Sprecher, kurze Wechsel. Kurze Beitraege sind der haeufigste Fehlerfall.",
        "turns": [
            ("A", "Passt das so?"), ("B", "Ja."), ("A", "Sicher?"), ("B", "Ganz sicher."),
            ("A", "Gut."), ("B", "Dann machen wir das."), ("A", "Einverstanden."),
            ("B", "Bis morgen."),
        ],
    },
    "4sprecher_besprechung": {
        "beschreibung": "Vier Sprecher, gemischte Laengen. Naeher an einer echten Besprechung.",
        "turns": [
            ("A", "Ich eroeffne die Runde. Wir haben heute drei Themen auf der Liste."),
            ("B", "Zum ersten Thema habe ich Zahlen mitgebracht."),
            ("C", "Bevor wir einsteigen: fehlt uns nicht noch die Rueckmeldung aus dem Betrieb?"),
            ("D", "Die kam gestern Abend. Ich trage sie gleich nach."),
            ("B", "Dann fange ich an."),
            ("A", "Bitte."),
            ("C", "Ich haenge mich an die Zahlen an, sobald sie stehen."),
        ],
    },
    "2sprecher_ueberlappung": {
        "beschreibung": "Zwei Sprecher, drei gebaute Ueberlappungen. Der harte Fall — "
                        "hier faellt die DER bei den meisten Modellen auseinander.",
        "turns": [
            ("A", "Ich wuerde vorschlagen, dass wir den Punkt heute abschliessen."),
            ("B", "Da bin ich anderer Meinung.", {"ueberlappt_vorherigen": 0.8}),
            ("A", "Moment, lass mich kurz ausreden."),
            ("B", "Entschuldige, bitte.", {"ueberlappt_vorherigen": 0.5}),
            ("A", "Danke. Also: der Punkt ist inhaltlich fertig."),
            ("B", "Das sehe ich genauso, nur die Formulierung stoert mich.",
             {"ueberlappt_vorherigen": 1.1}),
        ],
    },
}


def sprich(tts: str, text: str, stimme: str) -> np.ndarray:  # noqa: D401
    """Einen Beitrag synthetisieren, als float32-Mono bei RATE."""
    anfrage = json.dumps({
        "input": text, "voice": stimme,
        "response_format": "wav", "language": "german",
    }).encode()
    req = urllib.request.Request(f"{tts}/v1/audio/speech", data=anfrage,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        roh = r.read()
    import io
    daten, rate = sf.read(io.BytesIO(roh), dtype="float32", always_2d=True)
    daten = daten.mean(axis=1)                      # mono
    if rate != RATE:                                 # lineares Resampling reicht hier
        n = int(round(len(daten) * RATE / rate))
        daten = np.interp(np.linspace(0, len(daten) - 1, n),
                          np.arange(len(daten)), daten).astype(np.float32)
    return daten


def baue_szene(name: str, szene: dict, tts: str, ziel: Path) -> dict:
    spur = np.zeros(0, dtype=np.float32)
    segmente: list[tuple[str, float, float]] = []
    vorher: str | None = None
    for eintrag in szene["turns"]:
        wer, text = eintrag[0], eintrag[1]
        opt = eintrag[2] if len(eintrag) > 2 else {}
        ton = sprich(tts, text, SPRECHER[wer])
        ueberlappung = float(opt.get("ueberlappt_vorherigen", 0.0))
        if ueberlappung > 0 and len(spur) > 0:
            # Der Beitrag setzt VOR dem Ende des vorherigen ein: beide Spuren
            # ueberlagern sich, und beide stehen in der Wahrheit. Genau das
            # unterscheidet eine echte Besprechung von einer Vorlesung.
            start = max(0.0, len(spur) / RATE - ueberlappung)
            i = int(start * RATE)
            noetig = i + len(ton)
            if noetig > len(spur):
                spur = np.concatenate([spur, np.zeros(noetig - len(spur), dtype=np.float32)])
            spur[i:i + len(ton)] += ton
        else:
            pause = STILLE_KURZ if wer == vorher else STILLE_WECHSEL
            if len(spur):
                spur = np.concatenate([spur, np.zeros(int(pause * RATE), dtype=np.float32)])
            start = len(spur) / RATE
            spur = np.concatenate([spur, ton])
        segmente.append((wer, start, len(ton) / RATE))
        vorher = wer

    spitze = float(np.max(np.abs(spur))) or 1.0
    spur = (spur / spitze * 0.89).astype(np.float32)   # Uebersteuerung vermeiden

    wav = ziel / f"{name}.wav"
    rttm = ziel / f"{name}.rttm"
    sf.write(wav, spur, RATE, subtype="PCM_16")
    with rttm.open("w", encoding="utf-8") as f:
        for wer, start, dauer in segmente:
            f.write(f"SPEAKER {name} 1 {start:.3f} {dauer:.3f} <NA> <NA> {wer} <NA> <NA>\n")
    return {"audio_filepath": str(wav), "offset": 0, "duration": len(spur) / RATE,
            "label": "infer", "text": "-", "num_speakers": len({s for s, _, _ in segmente}),
            "rttm_filepath": str(rttm), "uem_filepath": None, "szene": name}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts", default="http://localhost:8002", help="Qwen3-TTS-Adapter")
    ap.add_argument("--ziel", type=Path, default=ZIEL)
    ap.add_argument("--szenen", help="Kommagetrennt; ohne Angabe alle")
    a = ap.parse_args()

    a.ziel.mkdir(parents=True, exist_ok=True)
    wunsch = {s.strip() for s in a.szenen.split(",")} if a.szenen else set(SZENEN)
    zeilen = []
    for name, szene in SZENEN.items():
        if name not in wunsch:
            continue
        eintrag = baue_szene(name, szene, a.tts, a.ziel)
        zeilen.append(eintrag)
        print(f"{name:26} {eintrag['duration']:6.1f}s  {eintrag['num_speakers']} Sprecher  "
              f"{len(szene['turns'])} Beitraege — {szene['beschreibung'][:44]}")
    manifest = a.ziel / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(z, ensure_ascii=False) + "\n" for z in zeilen),
                        encoding="utf-8")
    print(f"\n{len(zeilen)} Szenen in {a.ziel}, Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
