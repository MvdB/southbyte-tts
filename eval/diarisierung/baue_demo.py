#!/usr/bin/env python3
"""Baut die Demo-Seite: Wahrheit gegen Vorhersage auf einer Zeitachse.

Eine einzelne HTML-Datei mit eingebettetem Audio — nichts nachzuladen, laeuft
auf jedem Rechner im Besprechungsraum. Je Szene zwei Spurenbaender uebereinander:
oben was wirklich gesprochen wurde, unten was das Modell erkannt hat.

    python3 eval/diarisierung/baue_demo.py --ergebnis results/diarisierung_<datum>
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

FARBEN = ["#00e676", "#29b6f6", "#ffa726", "#ef5350", "#ab47bc", "#26a69a"]


def spur(segmente: list[tuple[str, float, float]], dauer: float, namen: list[str]) -> str:
    stuecke = []
    for wer, start, ende in segmente:
        links = start / dauer * 100
        breite = max((ende - start) / dauer * 100, 0.4)
        farbe = FARBEN[namen.index(wer) % len(FARBEN)]
        stuecke.append(
            f'<div class="seg" style="left:{links:.2f}%;width:{breite:.2f}%;'
            f'background:{farbe}" title="{html.escape(wer)} {start:.2f}–{ende:.2f}s">'
            f'<span>{html.escape(wer)}</span></div>')
    return f'<div class="spur">{"".join(stuecke)}</div>'


def lies_rttm(p: Path) -> list[tuple[str, float, float]]:
    raus = []
    for z in p.read_text().splitlines():
        t = z.split()
        if len(t) > 7:
            raus.append((t[7], float(t[3]), float(t[3]) + float(t[4])))
    return sorted(raus, key=lambda x: x[1])


def lies_labels(p: Path) -> list[tuple[str, float, float]]:
    raus = []
    for z in p.read_text().splitlines():
        t = z.split()
        if len(t) == 3:
            raus.append((t[2], float(t[0]), float(t[1])))
    return sorted(raus, key=lambda x: x[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ergebnis", required=True, type=Path)
    ap.add_argument("--testset", type=Path,
                    default=Path(__file__).resolve().parents[2] / "testset" / "diarisierung")
    ap.add_argument("--modus", default="offline")
    a = ap.parse_args()

    messungen = json.loads((a.ergebnis / "ergebnisse.json").read_text())
    teile = []
    for m in [x for x in messungen if x["modus"] == a.modus]:
        name = m["szene"]
        wav = a.testset / f"{name}.wav"
        wahrheit = lies_rttm(a.testset / f"{name}.rttm")
        vorhersage = lies_labels(a.ergebnis / f"{name}.{a.modus}.labels")
        dauer = max(e for _, _, e in wahrheit + vorhersage)
        n_w = sorted({w for w, _, _ in wahrheit})
        n_v = sorted({w for w, _, _ in vorhersage})
        b64 = base64.b64encode(wav.read_bytes()).decode()
        d = m["streng"]
        teile.append(f"""
<section>
  <h2>{html.escape(name)}</h2>
  <p class="zahlen">DER <b>{d['DER']:.1%}</b> · Fehlalarm {d['FA']:.1%} ·
     Verpasst {d['MISS']:.1%} · Verwechslung {d['VERWECHSLUNG']:.1%} ·
     Sprecher erkannt {m['sprecher_erkannt']} von {m['sprecher_wahrheit']} ·
     {dauer:.1f}s in {m['dauer_s']}s gerechnet</p>
  <audio controls preload="none" src="data:audio/wav;base64,{b64}"></audio>
  <div class="zeile"><span class="lbl">Wahrheit</span>{spur(wahrheit, dauer, n_w)}</div>
  <div class="zeile"><span class="lbl">Modell</span>{spur(vorhersage, dauer, n_v)}</div>
</section>""")

    doc = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>Diarisierung — Nemotron-3-Diarization</title><style>
:root{{color-scheme:dark}}
body{{background:#0d1117;color:#e6edf3;font:15px/1.6 system-ui,sans-serif;margin:0;padding:2rem;max-width:1100px}}
h1{{font-size:1.5rem;margin:0 0 .3rem}}
.hinweis{{color:#9198a1;border-left:3px solid #ffa726;padding:.6rem 1rem;background:#161b22;margin:1.2rem 0}}
section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem 1.2rem;margin:1.2rem 0}}
h2{{font-size:1.05rem;margin:0 0 .4rem;font-family:ui-monospace,monospace}}
.zahlen{{color:#9198a1;font-size:.86rem;margin:.2rem 0 .8rem}}
audio{{width:100%;height:32px;margin-bottom:.8rem}}
.zeile{{display:flex;align-items:center;gap:.7rem;margin:.35rem 0}}
.lbl{{width:4.5rem;font-size:.8rem;color:#9198a1;text-align:right;flex:none}}
.spur{{position:relative;height:30px;background:#0d1117;border:1px solid #30363d;border-radius:4px;flex:1}}
.seg{{position:absolute;top:0;height:100%;border-radius:3px;display:flex;align-items:center;
     justify-content:center;font-size:.72rem;color:#0d1117;font-weight:700;overflow:hidden}}
footer{{color:#9198a1;font-size:.82rem;margin-top:2rem;border-top:1px solid #30363d;padding-top:1rem}}
</style></head><body>
<h1>Diarisierung — wer spricht wann</h1>
<p class="zahlen">nvidia/Nemotron-3-Diarization · DGX Spark (GB10) · Betriebsart {a.modus}</p>
<div class="hinweis"><b>Synthetisches Testmaterial.</b> Die Dialoge sind aus TTS-Stimmen
gebaut, damit die Wahrheit auf die Millisekunde bekannt ist. Das Audio ist dadurch sauberer
als jede echte Besprechung: ein Raum, keine Nebengeräusche, Überlappung nur dort, wo sie
absichtlich gebaut wurde. Die DER ist deshalb eine <b>Untergrenze des Fehlers</b> und
keine Feldmessung.</div>
{''.join(teile)}
<footer>DER nach NIST md-eval, gerechnet von NeMo selbst — collar 0, Überlappung zählt mit.
Farben stehen je Zeile für Sprecher; oben die gebaute Wahrheit, unten die Vorhersage.</footer>
</body></html>"""
    ziel = a.ergebnis / "demo.html"
    ziel.write_text(doc, encoding="utf-8")
    print(f"Demo: {ziel} ({ziel.stat().st_size/1024:.0f} kB, {len(teile)} Szenen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
