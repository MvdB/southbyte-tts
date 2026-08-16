# The German evaluation

```
TTS adapter → WAV → whisper-large-v3 (vLLM) → WER/CER against accepted refs
```

No human in the loop. What the numbers mean, and what they do not, is the point
of this page.

## The testset

`testset/german_tts_v1.jsonl` — 43 German cases aimed at failure modes that
public, English-centric benchmarks do not cover:

| Category | Cases | Probes |
|---|---|---|
| normalization | 18 | currency, dates, times, phone numbers, abbreviations (`z. B.`, `GmbH & Co. KG`), §-references, units, ordinals, Roman numerals |
| compound | 6 | long and novel compounds (*Rechtsschutzversicherungsgesellschaften*) |
| loanword | 6 | French and Italian loanwords, English tech terms, Denglisch code-switching |
| umlaut | 6 | minimal pairs (*schon/schön*, *drücken/drucken*, *Höhle/Hölle*), ß |
| longform | 4 | nested clauses, enumerations, mixed sentence modes, direct speech |
| names | 3 | German and European place names, non-German surnames |

Each case lists one or more acceptable verbalizations (`refs`); scoring takes the
best match.

**Deliberately not measured:** prosody and naturalness — use the listening pages
for a human spot-check — and homograph stress (*umfahren* against *umfahren*),
which is indistinguishable in a transcript.

## Running it

```bash
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8002 \
  --stt http://127.0.0.1:8007 \
  --voice uncle_fu \
  --repeats 3 \
  --out results/$(date +%F)_qwen-unclefu_n3

# Quick smoke: --limit 5     Restrict: --category normalization,umlaut
python eval/make_listen_page.py results/<run-dir>    # → listen.html
```

`--repeats 3` synthesizes each case three times. Per case the mean plus
`wer_min`/`wer_max` are reported; the summary adds `wer_best_mean`.

Written raw data first, summary fail-safe afterwards:

- `results_raw.jsonl` — per-case transcripts, WER/CER per ref and repeat
- `audio/*.wav` — generated audio for manual inspection
- `summary.json` — means, per-category WER, real-time factor, worst cases
- `listen.html` — audio players with text, transcript and WER per case

Run directories are named `results/YYYY-MM-DD_<config>_nN`. `results/` and
`*.wav` are gitignored by design; raw runs stay local.

## Capped WER, and why it is the headline

WER is unbounded above, so a single degenerate transcript can dominate a mean.
Two different things cause that here, and only one is the TTS model's fault:

- **The judge loops.** whisper occasionally emits 1500+ characters of
  "null. null. …" for 3.7 s of audio — physically impossible speech. The
  evaluator detects the implausible transcript-length to duration ratio and
  retries once with light sampling (`temperature 0.3` breaks the deterministic
  loop), flagging survivors as `asr_runaway`. This is not cosmetic: it moved
  VoxCPM2 from 0.534 to 0.185. Its apparent collapse was mostly judge
  hallucination, with one genuine loop-babble repeat left over.
- **The model loops.** VoxCPM2 does sometimes produce off-text babble. That is a
  real failure and should count.

`wer_capped_mean` caps each repeat at 1.0 (total substitution) and is what the
comparison page reports. The gap between capped and raw shows how far individual
repeats derailed — for `de_male_news` that is 0.166 against 0.275.

## Choosing the judge is measurable, not a matter of taste

`testset/judge_calib_v1.jsonl` plus `eval/judge_bench.py` calibrate a candidate
against audio whose content is *known*: a TTS speaks the already-verbalized refs,
so any judge error is unambiguously the judge's. The calibration audio is not in
git — regenerate it with `eval/make_judge_calib.sh` before comparing.

**Absolute values depend on the audio set.** whisper scored 0.137 WER on one set
and 0.154 on another — same model, same prompt. Numbers are comparable only
*within* one set, so always re-measure the incumbent alongside the candidate.

Latest set: whisper 0.154 WER / 28 % digit rate, voxtral-mini-3b 0.208 / 67 %,
voxtral-realtime-4b 0.231 / 83 %.

**Rejected, and why:**

- **granite-speech-4.1-2b** loses content — *"siebzehn Uhr fünfundvierzig"*
  becomes *"Der Zug fährt um uhr"*. Worse than writing digits, which is only a
  format problem. It cost Voxtral roughly 0.02 WER and triggered a
  [since-retracted upstream report](https://github.com/vllm-project/vllm-omni/issues/5510).
- **Voxtral-Mini-4B-Realtime** keeps the content but inverse-normalizes 83 % of
  the normalization cases (*"Paragraf zwölf"* → *"§ 12"*), and the verbatim
  prompt has no effect on the Voxtral family — it drops whisper from 83 % to
  28 %.
- **granite-speech-plus** does not run at all: vLLM 0.25.1 fails with
  `Failed to apply prompt replacement for mm_items['audio'][0]`.

Two rules that came out of this:

**Always reach a judge through `/v1/audio/transcriptions` first.** Voxtral-Mini
happily answers `chat/completions` — with an English *translation*
(*"Das Gerät kostet 3,50 Euro"* → *"The device cost 3,500."*), which silently
turns cross-validation into nonsense at WER ~0.85. A successful response is not
proof of a correct one.

**The verbatim prompt's examples come from outside the testset**, deliberately.
Priming the judge with expected answers would mask real TTS errors.

## The published comparison

`docs/` holds the static pages (GitHub Pages: Settings → Pages → branch `main`,
folder `/docs`): `index.html` with the metric table, plus one listening page per
model-and-voice combination with all 43 clips.

`results/` is scanned automatically; per combination the newest complete run
wins, so a new run of the same model and voice **overwrites** its page, and pages
for vanished combinations are pruned. Partial runs — smoke, `--limit`,
`--category` — are skipped. Exactly one clip per case is published (repeat r0,
MP3 ~64 kbit/s mono): representative, never cherry-picked.

```bash
pip install --break-system-packages soundfile   # needs libsndfile >= 1.2 for MP3
python eval/make_docs.py
```

Cross-model comparisons are consolidated in `results/COMPARISON_*.md`.
