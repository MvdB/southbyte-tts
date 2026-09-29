# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

TTS serving and German-language evaluation on an NVIDIA DGX Spark (GB10, **aarch64** — many x86 wheels don't exist here; this drives most Dockerfile oddities). Part of a repo family: shared infra in `southbyte-core` (populates the model store at `~/hf_models/` via `hf-sync`), the vLLM/STT stack in `southbyte-vllm`.

Code comments, docstrings, commit messages, and docs are written in **German** — keep that convention.

Seit 2026-09-28 deckt das Repo **Sprache breiter ab als nur TTS**: mit
`nvidia/Nemotron-3-Diarization` kommt das erste Modell dazu, das Audio
analysiert statt es zu erzeugen (`eval/diarisierung/`). Die STT-Judges
(Whisper, Voxtral) wohnten ohnehin schon hier. Das Testmaterial der
Diarisierung entsteht aus den TTS-Stimmen dieses Repos — deshalb hier und
nicht in einem eigenen Repo. Kommt ein drittes Analyse-Modell dazu
(Sprecher-Identifikation, VAD), waere eine Umbenennung auf `southbyte-speech`
ehrlicher, als den Namen weiter zu dehnen.

## Architecture

Four TTS model adapters, all exposing the **same OpenAI-compatible API** (`POST /v1/audio/speech`, `GET /v1/voices`, `GET /health`, WAV mono 16-bit out), so the evaluator only needs a different `--tts` URL:

| Adapter | Model | Port | Image |
|---|---|---|---|
| `serving/server.py` (NeMo) | nvidia/magpie_tts_multilingual_357m | 8001 | `spark-magpie-tts:v1` / `:v1-tn` |
| `serving/server_qwen3tts.py` | Qwen3-TTS-12Hz (CustomVoice or VoiceDesign via `MODEL_DIR`) | 8002 | `spark-qwen3-tts:v1` |
| `serving/server_chatterbox.py` | ResembleAI Chatterbox Multilingual V3 | 8003 | `spark-chatterbox:v1` |
| `serving/server_voxcpm.py` | openbmb/VoxCPM2 | 8004 | `spark-voxcpm:v1` |
| `serving/server_audio8.py` | Audio8/Audio8-TTS-Preview-0.6b | 8009 | `spark-audio8:v1` |
| `serving/server_audio8_onnx.py` | Audio8/audio8-TTS-0.1B-ONNX-INT8 | 8012 | `spark-audio8-onnx:v1` |
| — (vLLM-Omni native) | mistralai/Voxtral-4B-TTS-2603 | 8005 | `spark-voxtral-tts:v1` |

Image layering matters: `Dockerfile.chatterbox` and `Dockerfile.voxcpm` build **FROM `spark-qwen3-tts:v1`** (it already contains NGC torch + source-built torchaudio); `Dockerfile.tn` builds FROM `spark-magpie-tts:v1` and adds German text normalization (pynini has no aarch64 wheel — prebuilt OpenFst/pynini artifacts are required in the build context, see its header). Without the `.tn` layer, Magpie's `apply_TN` is a **silent no-op**. Each derived Dockerfile ends its pip install with an import/CUDA guard (`torch.version.cuda`, torchaudio CUDA check, model import) — keep that when touching dependencies.

`Dockerfile.audio8` also builds **FROM `spark-qwen3-tts:v1`**, but adds no packages at all: the model ships its own remote code (`trust_remote_code`, architecture `arktts`) and needs only torch, torchaudio, soundfile and transformers — all already in the base (verified 2026-09-01 against transformers 4.57.3). It is the **second zero-shot cloning adapter** after Chatterbox, and the only one that also requires a **reference transcript**: a voice is `voices/<name>.wav` *plus* `voices/<name>.txt` with the spoken text. Missing `.txt` is rejected with HTTP 400 and the voice is hidden from `/v1/voices` — silently falling back to reference-free generation would attribute a different voice's numbers to the cloned run. The model has **no language parameter** (its system prompt is a fixed `convert the provided text to speech`); the adapter accepts `language` but ignores it and marks this with the response header `X-Language-Ignored`. Output is **44.1 kHz** where every other adapter emits 24 kHz.

The 0.1B ONNX model is the one adapter that runs **no torch at all**. It ships its own Apache-2.0 reference runtime (`github.com/Audio8-AI/Audio8_TTS`, directory `onnx_runtime_0_1b_int8`) on ONNX Runtime's `CPUExecutionProvider`, and that runtime already serves `/v1/audio/speech`. `server_audio8_onnx.py` therefore **reimplements nothing**: it imports their FastAPI app and bolts on the two endpoints our contract adds — `/health` (the evaluator reads `model` from it; without it `summary.json` records `"?"` and `make_docs.py` names the page wrong) and `/v1/voices`. `Dockerfile.audio8onnx` is consequently **not** built FROM `spark-qwen3-tts:v1` — a 24 GB torch base would be dead weight — but from `python:3.12-slim`, and it pins their repo to a **fixed commit** so a later rebuild measures the same runtime as the published run. A voice here is not a WAV on disk but a **registered code set**: POST `/api/voices/register` with audio plus its exact transcript, persisted in the mounted `voices_onnx/`.

Voxtral is different: no adapter of ours — vLLM-Omni serves `/v1/audio/speech` natively (`Dockerfile.voxtral`, FROM `vllm/vllm-openai:v0.25.1` + `vllm-omni==0.25.0rc1`; the stable 0.24.0 is **broken** for this model — it ignores the input text). Two hard-won platform facts in `vllm-omni/voxtral_tts_stages.yaml` (now in the sibling repo [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles); `run_voxtral_tts.sh` mounts it via `$SPARK_PROFILES_DIR`): `enforce_eager` is required on GB10 (CUDA graphs corrupt the audio), and vllm/vllm-omni must match in minor version. The evaluator auto-detects native endpoints via `/v1/models` (sends `model` field, falls back to WAV-length timing).

Models are mounted read-only from `~/hf_models` (`HF_MODELS_DIR`); no downloads at serve time (exception: Magpie fetches its NanoCodec vocoder from HF on first start).

Eval pipeline (`eval/roundtrip_eval.py`, no human in the loop):

```
TTS adapter → WAV → whisper-large-v3 on vLLM (judge, port 8007) → WER/CER vs. refs
```

- The judge is **whisper-large-v3** (`serving/run_whisper_judge.sh`), reached via `/v1/audio/transcriptions` with `ASR_VERBATIM_PROMPT` as the initial prompt. It replaced granite-speech-4.1-2b on 2026-07-30 after a calibration run proved granite drops number words — see below.
- **Judge selection is measurable, not a matter of taste.** `testset/judge_calib_v1.jsonl` + `eval/judge_bench.py` calibrate a candidate on audio whose content is *known*: a TTS speaks the already-verbalized refs, so any judge error is unambiguously the judge's. The audio itself is **not** in git (`results/` is ignored) — regenerate it with `eval/make_judge_calib.sh` before any judge comparison. **Absolute values depend on the audio set** (whisper scored 0.137 WER on the 2026-07-30 audio and 0.154 on the 2026-07-31 audio — same model, same prompt), so numbers are comparable only *within* one audio set: always re-measure the incumbents alongside the candidate. Latest set (`results/judge_bench_calib/calib-v2-*`): whisper 0.154 WER / 28 % digit rate, voxtral-mini-3b 0.208 / 67 %, voxtral-realtime-4b 0.231 / 83 %.
- Rejected judges and why: **granite-speech-4.1-2b** loses content ("siebzehn Uhr fünfundvierzig" → "Der Zug fährt um uhr") — worse than writing digits, which is only a format problem. **Voxtral-Mini-4B-Realtime-2602** keeps the content but inverse-normalizes 83 % of the normalization cases ("Paragraf zwölf" → "§ 12"), and the verbatim prompt has no effect on the Voxtral family (it drops whisper from 83 % to 28 %); launch script with the two required `--hf-overrides` tricks is `serving/run_voxtral_realtime_judge.sh`. **granite-speech-plus** does not run at all: vLLM 0.25.1 fails with `Failed to apply prompt replacement for mm_items['audio'][0]`.
- The verbatim prompt's examples deliberately come from **outside** the testset. Priming the judge with expected answers would mask real TTS errors.
- **Always reach a judge via `/v1/audio/transcriptions` first** (`judge_transcribe`). Voxtral-Mini happily answers `chat/completions` — with an English *translation* ("Das Gerät kostet 3,50 Euro" → "The device cost 3,500."), which silently turns cross-validation into nonsense (WER ~0.85). A successful response is not proof of a correct one.
- Testset `testset/german_tts_v1.jsonl`: 43 cases with categories (normalization, compound, loanword, umlaut, longform, names); each case has one or more acceptable `refs`, scoring takes the best match.
- Output convention: raw data is written **first** (`results_raw.jsonl`, `audio/*.wav`), summary generation is fail-safe afterwards (`summary.json`). Results dirs are named `results/YYYY-MM-DD_<config>_nN`.
- `results/` and `*.wav` are **gitignored by design** (raw runs stay local). The published comparison lives in `docs/` (GitHub-Pages-ready): `eval/make_docs.py` scans `results/`, keeps the newest complete run per (model, voice) — the page name derives from model+voice, so a new run **overwrites** the existing page and orphaned pages are pruned. One MP3 clip per case (repeat r0 — representative, never the best repeat). Regenerate and commit `docs/` after new runs.

## Commands

Build and run servers (from `serving/`):

```bash
# Magpie (+ German TN layer):
docker build -t spark-magpie-tts:v1 .
docker build -t spark-magpie-tts:v1-tn -f Dockerfile.tn .
IMAGE=spark-magpie-tts:v1-tn ./run_server.sh          # port 8001

# Qwen3-TTS (variant via MODEL_DIR):
docker build -t spark-qwen3-tts:v1 -f Dockerfile.qwen3tts .
./run_qwen3tts.sh                                     # port 8002
MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./run_qwen3tts.sh
# VoiceDesign hat keine festen Sprecher: die Stimme ist ein instruct-Text.
# Benannte Presets (de_female_news, de_male_news, de_female_calm) werden ueber
# das normale voice-Feld gewaehlt — ein Server bedient alle. QWEN_TTS_VOICE_DESIGN
# setzt den Default, QWEN_TTS_VOICE_INSTRUCT ueberschreibt mit Freitext.
# Ohne Preset-Namen hiessen alle Laeufe "design" und kollidierten in docs/.

# Chatterbox / VoxCPM2 (require spark-qwen3-tts:v1 to exist):
docker build -t spark-chatterbox:v1 -f Dockerfile.chatterbox .
VOICES_DIR=$PWD/../voices ./run_chatterbox.sh         # port 8003; /voices/<name>.wav = zero-shot clone
docker build -t spark-voxcpm:v1 -f Dockerfile.voxcpm .
./run_voxcpm.sh                                       # port 8004

# Smoke test any adapter:
curl -s http://127.0.0.1:8002/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Guten Morgen!", "voice": "serena", "language": "de"}' -o hallo.wav
```

Run the eval (needs the whisper judge on 8007 via `serving/run_whisper_judge.sh`, plus the Voxtral-Mini second judge on 8006 via `serving/run_voxtral_mini_judge.sh`). The second judge is **not optional if the run is to be published**: `southbyte-results/feeds.py::load_tts` only picks up run directories that contain a `rescore_judge2.json`, so an un-rescored run is invisible on results.southbyte.de. Until 2026-09-01 no launch script for it existed and it was started by hand:

```bash
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8002 \
  --stt http://127.0.0.1:8007 \
  --voice uncle_fu \
  --repeats 3 \
  --out results/$(date +%Y-%m-%d)_<config>_n3

# Quick smoke: --limit 5; restrict categories: --category normalization,umlaut
# Listening page for human spot-check:
python eval/make_listen_page.py results/<run-dir>    # → listen.html

# Regenerate the published docs/ pages
# (needs: pip install --break-system-packages soundfile — PEP-668-Umgebung):
python eval/make_docs.py
```

There are no unit tests or linters; the eval run *is* the test.

## Evaluation rules (learned the hard way)

- **Never fix a voice without running the eval against it.** With Qwen3-TTS the voice shifts entire error classes (a Chinese-native voice derailed German compounds into English; an English-native voice mangled German digits).
- Use `--repeats 3`: Magpie and Qwen sample stochastically; single runs swing small categories by ±0.1 WER.
- **Run-to-run spread at n=3 is ~0.02 WER on the overall mean** — measured directly: the same Qwen VoiceDesign voice (`de_female_news`, byte-identical instruct) scored 0.182 and 0.160 in two runs. Differences below ~0.02 between configurations are noise; do not rank on them. The current top three (uncle_fu 0.151, de_male_news 0.157, de_female_news 0.160) are one indistinguishable group.
- WER is unbounded above and a single degenerate repeat can dominate the mean. Report `wer_capped_mean` (each repeat capped at 1.0); the gap to the raw mean shows how far individual repeats derailed. The judge itself loops (1500+ chars of "null. null. …" for 3.7 s of audio) — `roundtrip_eval.py` detects the implausible length/duration ratio and retries once with `temperature 0.3`, flagging survivors as `asr_runaway`. This is not cosmetic: it moved VoxCPM2 from 0.534 to 0.185, i.e. its apparent collapse was mostly judge hallucination, with one genuine loop-babble repeat left over.
- Measured WER includes STT errors (it's an upper bound of TTS error) — interpret per-category **deltas**, not absolutes. Word-level WER over-penalizes German compounds when the STT hyphenates them; check CER there.
- The testset deliberately does not measure prosody/naturalness (use `listen.html` for a human spot-check) or homograph stress.
- Cross-model comparisons are consolidated in `results/COMPARISON_*.md`.
