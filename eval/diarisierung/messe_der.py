#!/usr/bin/env python3
"""Misst die Diarisierungsguete (DER) von Nemotron-3-Diarization je Szene.

Die DER-Rechnung kommt aus NeMo selbst (`score_labels_from_rttm_labels`, intern
md-eval — der NIST-Standard). Eine eigene DER-Implementierung wuerde am Ende
sich selbst messen.

Zwei Betriebsarten, beide aus der Modellkarte:
  offline    Modellvorgaben, grosser Puffer — der guenstigste Fall
  streaming  chunk_len=340, right_context=40, fifo_len=40, update_period=300
             (die Werte der Schnellstart-Anleitung)

Und zwei Strengegrade, weil die Zahl ohne diese Angabe nicht lesbar ist:
  streng   collar=0.0, Ueberlappung zaehlt mit — so bewertet der Hersteller in
           diarization_evaluation.md
  ueblich  collar=0.25, Ueberlappung ignoriert — die in der Literatur
           verbreitete Nachsicht; dieselbe Vorhersage sieht damit besser aus

Aufruf im Container (siehe serving/Dockerfile.diarisierung):
    python3 messe_der.py --manifest /daten/manifest.jsonl --ziel /ergebnis
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELL = "/hf_models/nvidia--Nemotron-3-Diarization/Nemotron-3-Diarization.nemo"

# Aus der Schnellstart-Anleitung der Modellkarte.
STREAMING = {"chunk_len": 340, "chunk_right_context": 40, "fifo_len": 40,
             "spkcache_update_period": 300}


def lade(modellpfad: str):
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    m = SortformerEncLabelModel.restore_from(modellpfad, map_location="cuda")
    m.eval()
    if torch.cuda.is_available():
        m = m.to(torch.device("cuda"))
    return m


def als_streaming(modell) -> None:
    for k, v in STREAMING.items():
        setattr(modell.sortformer_modules, k, v)
    modell._check_streaming_parameters()


def vorhersage_zu_labels(vorhersage) -> list[str]:
    """NeMos diarize()-Ausgabe in RTTM-Labelzeilen 'start ende sprecher'.

    Die Ausgabeform haengt an der NeMo-Fassung: mal Strings der Form
    "start end speaker", mal Objekte mit .start/.end/.label. Beides annehmen,
    statt auf eine Form zu wetten.
    """
    labels = []
    for eintrag in vorhersage:
        if isinstance(eintrag, str):
            labels.append(eintrag.strip())
            continue
        start = getattr(eintrag, "start", None)
        ende = getattr(eintrag, "end", None)
        wer = getattr(eintrag, "label", None) or getattr(eintrag, "speaker", None)
        if start is None:
            raise TypeError(f"unbekannte Segmentform: {type(eintrag)} {eintrag!r}")
        labels.append(f"{float(start):.3f} {float(ende):.3f} {wer}")
    return labels


def bewerte(ref_labels, hyp_labels, name, collar, ignore_overlap):
    from nemo.collections.asr.metrics.der import score_labels_from_rttm_labels
    ergebnis = score_labels_from_rttm_labels(
        ref_labels_list=[(name, ref_labels)],
        hyp_labels_list=[(name, hyp_labels)],
        collar=collar, ignore_overlap=ignore_overlap, verbose=False)
    if ergebnis is None:
        return None
    # score_labels_from_rttm_labels gibt (Ergebnisobjekt, Sprecherzuordnung,
    # (DER, FA, MISS, Verwechslung)) zurueck. Die Zahlen stehen im dritten
    # Element; am Beispiel geprueft: DER = FA + MISS + Verwechslung.
    der, fa, miss, verwechslung = (float(x) for x in ergebnis[2])
    return {"DER": round(der, 4), "FA": round(fa, 4),
            "MISS": round(miss, 4), "VERWECHSLUNG": round(verwechslung, 4),
            "zuordnung": ergebnis[1].get(name, {})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/daten/manifest.jsonl")
    ap.add_argument("--ziel", default="/ergebnis")
    ap.add_argument("--modell", default=MODELL)
    a = ap.parse_args()

    from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

    szenen = [json.loads(z) for z in Path(a.manifest).read_text().splitlines() if z.strip()]
    ziel = Path(a.ziel); ziel.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    modell = lade(a.modell)
    print(f"[laden] {time.time() - t0:.0f}s", flush=True)

    ergebnisse = []
    for modus in ("offline", "streaming"):
        if modus == "streaming":
            als_streaming(modell)
        for s in szenen:
            name = s["szene"]
            t0 = time.time()
            roh = modell.diarize(audio=[s["audio_filepath"]], batch_size=1)
            dauer = time.time() - t0
            vorhersage = roh[0] if roh and isinstance(roh, list) else roh
            hyp = vorhersage_zu_labels(vorhersage)
            ref = rttm_to_labels(s["rttm_filepath"])
            eintrag = {
                "szene": name, "modus": modus,
                "dauer_s": round(dauer, 2),
                "echtzeitfaktor": round(dauer / s["duration"], 3),
                "sprecher_wahrheit": s["num_speakers"],
                "sprecher_erkannt": len({z.split()[-1] for z in hyp}),
                "segmente_wahrheit": len(ref), "segmente_erkannt": len(hyp),
                "streng": bewerte(ref, hyp, name, 0.0, False),
                "ueblich": bewerte(ref, hyp, name, 0.25, True),
            }
            ergebnisse.append(eintrag)
            (ziel / f"{name}.{modus}.labels").write_text("\n".join(hyp) + "\n")
            print(f"{modus:9} {name:26} DER streng {eintrag['streng']['DER']:.3f} "
                  f"(FA {eintrag['streng']['FA']:.3f} MISS {eintrag['streng']['MISS']:.3f} "
                  f"VERW {eintrag['streng']['VERWECHSLUNG']:.3f}) | "
                  f"ueblich {eintrag['ueblich']['DER']:.3f} | Sprecher "
                  f"{eintrag['sprecher_erkannt']}/{eintrag['sprecher_wahrheit']} | "
                  f"RTF {eintrag['echtzeitfaktor']}", flush=True)

    (ziel / "ergebnisse.json").write_text(
        json.dumps(ergebnisse, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(ergebnisse)} Messungen → {ziel}/ergebnisse.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
