# Diarisierung — Basis-Test für Nemotron-3-Diarization

Stand 2026-09-28. **Vorbereitet, noch nicht gemessen.**

„Wer spricht wann" — das Modell ordnet jedem Zeitabschnitt einen Sprecher zu,
bis zu acht, offline oder im Streaming. Es ist das erste Modell in diesem Repo,
das Audio **analysiert** statt es zu erzeugen; die TTS-Adapter liefern hier nur
das Testmaterial.

## Warum hier und nicht in einem eigenen Repo

Das Testmaterial entsteht aus den TTS-Stimmen dieses Repos — eine harte
Abhängigkeit. Dazu liegen die STT-Judges (Whisper, Voxtral) längst hier, das
Repo ist faktisch das Audio-Repo. Kommt ein drittes Analyse-Modell dazu
(Sprecher-Identifikation, VAD), ist eine Umbenennung auf `southbyte-speech`
ehrlicher, als den Namen weiter zu dehnen.

## Das Modell

`nvidia/Nemotron-3-Diarization`, 678 MB, **openmdw-1.1 — ausdrücklich auch für
den kommerziellen Einsatz freigegeben** (anders als Qwen-Image-2.1 auf der
Bildseite). Erschienen am 23.09.2026, Streaming-Sortformer-Architektur mit
Arrival-Order Speaker Cache. Latenz ab 0,32 s empfohlen, offline mit 30,4 s
Puffer. Liegt dreifach im Ordner: `.safetensors`, `.nemo` und als `q8_0.gguf`.

**Läuft nicht über vLLM.** Zwei Wege bietet der Hersteller an:

| Weg | Womit | Wofür |
|---|---|---|
| `NeMo-Speech.cpp` | C++-Binär, `nemo-speech diarize meeting.wav` | schlank, gut für die Demo |
| NVIDIA NeMo Speech | `nemo-toolkit[asr]`, Python | die Bewertung — `e2e_diarize_speech.py` rechnet die DER |

Für Zahlen brauchen wir den zweiten Weg: das Bewertungsskript des Herstellers
ist die Quelle der Wahrheit für die DER, und wer eine eigene DER-Rechnung baut,
misst am Ende seine eigene Implementierung.

## Das Testmaterial: synthetisch, mit exakter Wahrheit

`baue_testaudio.py` setzt deutsche Dialoge aus den Voice-Design-Stimmen des
Qwen3-TTS-Adapters zusammen (alle Sprecher aus **demselben** Adapter, damit
Aufnahmebedingung und Codec konstant bleiben). Die Wahrheit entsteht beim
Bauen: wir legen die Segmente selbst aneinander und kennen jeden Einsatz auf
die Millisekunde. Ausgabe ist WAV plus **RTTM**, das Format, das die
NeMo-Bewertung liest.

Vier Szenen, bewusst gestuft:

| Szene | Sprecher | Was sie prüft |
|---|--:|---|
| `2sprecher_klar` | 2 | der leichte Fall: lange Beiträge, saubere Pausen |
| `2sprecher_kurz` | 2 | kurze Wechsel — der häufigste Fehlerfall |
| `4sprecher_besprechung` | 4 | mehr Stimmen, gemischte Längen |
| `2sprecher_ueberlappung` | 2 | drei gebaute Überlappungen — hier fällt die DER meist auseinander |

**Der Vorbehalt gehört in jeden Bericht:** synthetisches Audio ist sauberer als
jede echte Besprechung. Ein Raum, keine Nebengeräusche, kein Stuhlrücken, und
Überlappung nur dort, wo wir sie absichtlich bauen. Eine DER auf diesem
Material ist eine **Untergrenze des Fehlers**, keine Feldmessung. Wer die Zahl
im Büro zeigt, muss diesen Satz mitliefern.

Eine echte Aufnahme als Gegenprobe wäre der nächste Schritt — sie braucht
Einwilligung der Aufgenommenen und Handarbeit beim Labeln (grob eine Stunde je
zehn Minuten Audio).

## Nächste Schritte

1. Testmaterial bauen (Adapter starten, dann `baue_testaudio.py`) und **anhören**
   — synthetische Dialoge können misslingen, und eine DER auf kaputtem Audio
   misst nichts.
2. NeMo-Image bauen (`nemo-toolkit[asr]` auf aarch64 — der erste offene Punkt,
   die Modellkarte nennt keine aarch64-Räder).
3. DER je Szene messen, offline und im Streaming.
4. Demo: Audio rein, Sprecherspuren über der Zeitachse raus.
5. Ergebnis wie gewohnt nach `results/` und auf die Seite.

## Offene Fragen

- **Läuft `nemo-toolkit[asr]` auf GB10/aarch64?** Nicht geprüft. Falls nicht,
  bleibt `NeMo-Speech.cpp` für die Demo, und die DER-Rechnung braucht einen
  anderen Weg.
- **Welche Streaming-Latenz messen wir?** Das Modell kann 0,08 s bis 30,4 s
  Puffer. Für eine Aussage im Büro sind zwei Punkte sinnvoll: die empfohlene
  Untergrenze 0,32 s und der Offline-Fall.
