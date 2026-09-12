#!/usr/bin/env bash
# Sprachprobe: kann das Modell auf Port $PORT ueberhaupt Deutsch?
#
# Warum es das gibt: Breeze-TTS-2 wurde am 01.09.2026 mit 258 Synthesen ueber
# zwei Konfigurationen gemessen, um am Ende 0.83 WER zu bestaetigen — was ein
# einziger Clip schon nach 30 Sekunden gezeigt hatte. Diese Probe kostet fuenf
# Synthesen statt 258 und entscheidet, ob ein voller Lauf sich lohnt.
#
# Sie ersetzt KEINE Messung. Sie beantwortet nur die Vorfrage: kommt
# ueberhaupt Deutsch heraus?
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8014}"
STT="${STT:-http://127.0.0.1:8007}"
VOICE="${VOICE:-de_f1}"
AUS="${AUS:-/tmp/sprachprobe_$$}"
mkdir -p "$AUS"

VERB="Alle Zahlen, Daten, Uhrzeiten und Abkürzungen werden als Wörter ausgeschrieben, niemals als Ziffern. Beispiele: acht neun sieben sechs, dreiundzwanzigster März neunzehnhundertachtzig, sechs Uhr zwanzig, zweiundvierzig Komma sieben, achtundneunzig Prozent, Absatz sieben."

curl -sf --max-time 10 "$STT/v1/models" >/dev/null || { echo "ABBRUCH: Judge auf $STT nicht erreichbar"; exit 1; }
curl -sf --max-time 10 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "ABBRUCH: TTS auf :$PORT nicht erreichbar"; exit 1; }

# Fuenf Saetze quer durch die Kategorien des Testsatzes.
SAETZE=(
  "Der Zug nach Braunschweig fährt um siebzehn Uhr fünfundvierzig von Gleis drei."
  "Die Röntgenaufnahme zeigte eine Veränderung im Brustkorb."
  "Unsere Geschäftsführerin heißt Ulrike Schäfer-Wörndle."
  "Das Bundesausbildungsförderungsgesetz regelt die Studienfinanzierung."
  "Er zahlte zweiundvierzig Komma sieben Prozent der Summe."
)

echo "Sprachprobe gegen :$PORT (Stimme $VOICE, Judge $STT)"
echo
gesamt=0; n=0
for i in "${!SAETZE[@]}"; do
  soll="${SAETZE[$i]}"
  if ! curl -sf --max-time 600 "http://127.0.0.1:$PORT/v1/audio/speech" \
       -H 'Content-Type: application/json' \
       -d "$(python3 -c "import json,sys; print(json.dumps({'input': sys.argv[1], 'voice': '$VOICE', 'language': 'de'}))" "$soll")" \
       -o "$AUS/probe_$i.wav"; then
    echo "  [$i] SYNTHESE FEHLGESCHLAGEN"; continue
  fi
  ist=$(curl -s --max-time 180 "$STT/v1/audio/transcriptions" \
        -F "file=@$AUS/probe_$i.wav" -F model=whisper-large-v3 -F language=de \
        -F "prompt=$VERB" -F temperature=0 \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('text','').strip())")
  wer=$(python3 - "$soll" "$ist" <<'PY'
import sys, re
def w(s): return re.findall(r"\w+", s.lower())
a, b = w(sys.argv[1]), w(sys.argv[2])
d = [[0]*(len(b)+1) for _ in range(len(a)+1)]
for i in range(len(a)+1): d[i][0] = i
for j in range(len(b)+1): d[0][j] = j
for i in range(1, len(a)+1):
    for j in range(1, len(b)+1):
        d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(a[i-1] != b[j-1]))
print(f"{d[len(a)][len(b)]/max(1,len(a)):.3f}")
PY
)
  # Zweites Signal: wie viele Zielwoerter kommen woertlich im Transkript vor?
  # Eine reine WER-Schwelle kann "spricht die Sprache schlecht" nicht von
  # "spricht sie nicht" unterscheiden — AuK kam am 12.09.2026 auf 0.603 und
  # traf dabei "Roentgenaufnahme" samt Umlaut, waehrend Breeze bei 0.831
  # fremde Lautung lieferte. Der Wortanteil trennt die beiden Faelle.
  anteil=$(python3 - "$soll" "$ist" <<'PY2'
import sys, re
def w(s): return set(x for x in re.findall(r"\w{4,}", s.lower()))
a, b = w(sys.argv[1]), w(sys.argv[2])
print(f"{len(a & b)/max(1,len(a)):.2f}")
PY2
)
  echo "  [$i] WER $wer  Wortanteil $anteil"
  echo "      Soll: $soll"
  echo "      Ist : $ist"
  gesamt=$(python3 -c "print($gesamt + $wer)"); n=$((n+1))
done

[ "$n" -eq 0 ] && { echo; echo "URTEIL: keine Synthese gelungen"; exit 1; }
mittel=$(python3 -c "print(f'{$gesamt/$n:.3f}')")
echo
echo "Mittlere WER ueber $n Saetze: $mittel"
echo
# Die Schwellen sind grob und bewusst so: der Judge selbst liegt bei ~0.154,
# die schwaechste deutschfaehige Konfiguration im Bestand bei 0.258,
# Breeze ohne Deutsch bei 0.831.
python3 - "$mittel" <<'PY'
import sys
m = float(sys.argv[1])
if m < 0.35:
    print("URTEIL: spricht Deutsch. Voller Messlauf lohnt sich.")
elif m < 0.75:
    print("URTEIL: GRENZFALL — die WER allein entscheidet das nicht. Jetzt die")
    print("        Transkripte oben lesen, nicht die Zahl:")
    print("        * Stehen da echte deutsche Woerter, sitzen Umlaute, stimmt die")
    print("          Grammatik stellenweise? Dann spricht das Modell Deutsch und")
    print("          scheitert kategoriespezifisch — genau das misst der volle Lauf.")
    print("        * Liest es sich als fremde Lautung (Breeze: 'von kleisteri")
    print("          Woerter')? Dann ist es eine Sprachluecke, und der Lauf lohnt nicht.")
    print("        Der Wortanteil je Satz ist dafuer das bessere Signal als die WER.")
    print("        Vorsicht: ein Teil der WER ist die Ziffernquote des Judges")
    print("        (28 %), kein TTS-Fehler — '7.45 Uhr' statt 'sieben Uhr'.")
else:
    print("URTEIL: kann kein Deutsch (Breeze lag bei 0.831). Voller Lauf misst eine")
    print("        Sprachluecke, keine Sprachqualitaet — Ergebnis so kennzeichnen")
    print("        und mit .nicht-veroeffentlichen von der Seite fernhalten.")
PY
echo "Clips: $AUS"
