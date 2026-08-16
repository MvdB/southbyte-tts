# southbyte-tts

Speech synthesis on an **NVIDIA DGX Spark**: five TTS models behind one API, and
an automated evaluation that measures how well they actually speak **German**.

**→ [The comparison](https://mvdb.github.io/southbyte-tts/)** — every model and
voice ranked, with all 43 clips to listen to.

> **Proof of concept, not a product.** No guaranteed availability, fitness or
> output quality, no support, no roadmap.

## What it does

Five models, one OpenAI-compatible API — the evaluator only needs a different
`--tts` URL:

| Model | Port | Licence | |
|---|---|---|---|
| [magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | 8001 | NVIDIA Open | German number/date normalization only with the `Dockerfile.tn` layer |
| [Qwen3-TTS-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 8002 | Apache-2.0 | 9 preset voices, or a voice written as a German description |
| [Chatterbox Multilingual V3](https://github.com/resemble-ai/chatterbox) | 8003 | MIT | Zero-shot voice cloning from a WAV; output carries a Perth watermark |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) | 8004 | Apache-2.0 | No fixed speakers, voice via description |
| [Voxtral-4B-TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | 8005 | **CC BY-NC 4.0** | **Non-commercial.** 20 preset voices; no cloning — the audio-encoder weights are not released |

And a **German testset that public benchmarks do not cover**: 43 cases probing
number and date normalization, long compounds, loanwords, umlaut minimal pairs,
long sentences and names. Each is synthesized, transcribed by a whisper judge and
scored as WER/CER against accepted verbalizations — no human in the loop.

Current leaders (capped WER, lower is better — full table on the page):

| Model · voice | WER |
|---|---|
| Qwen3-TTS CustomVoice · `uncle_fu` | **0.156** |
| Voxtral-4B-TTS · `de_female` | 0.158 |
| Qwen3-TTS VoiceDesign · `de_male_young` | 0.160 |
| Magpie **with** the German TN layer · `sofia` | 0.173 |
| Magpie **without** it · `sofia` | 0.224 |

That last pair is the clearest single result here: the same model, same voice,
0.05 WER apart — text normalization is worth more than the choice of model.

## Getting it running

You need Docker with GPU access, the models in `~/hf_models/` (populated by
[southbyte-sync](https://github.com/MvdB/southbyte-sync)), and for evaluation a
whisper-large-v3 endpoint — optionally a second judge (Voxtral-Mini-3B on 8006)
to cross-check it, which is where the second WER column on the comparison page
comes from. NGC base images `nvcr.io/nvidia/nemo:26.06` and
`nvcr.io/nvidia/pytorch:26.06-py3` are both multi-arch. Voxtral additionally
needs the GB10 stage config from
[southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles).

```bash
# 1. Build and start an adapter — Qwen3-TTS as the example
cd serving
docker build -t spark-qwen3-tts:v1 -f Dockerfile.qwen3tts .
./run_qwen3tts.sh                                     # port 8002

# 2. Does it speak?
curl -s http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Guten Morgen!", "voice": "serena", "language": "de"}' \
  -o hallo.wav

# 3. Start the judge, then run the eval
./run_whisper_judge.sh                                # port 8007
python eval/roundtrip_eval.py \
  --testset testset/german_tts_v1.jsonl \
  --tts http://127.0.0.1:8002 --stt http://127.0.0.1:8007 \
  --voice uncle_fu --repeats 3 \
  --out results/$(date +%F)_qwen-unclefu_n3

# 4. Rebuild the published comparison
python eval/make_docs.py
```

The other four adapters, their environment variables, the endpoints and the
image layering: [`docs/serving.md`](docs/serving.md).

## What to watch out for

**Voice choice is not cosmetic.** With Qwen3-TTS the voice shifts entire error
classes — a Chinese-native voice derailed German compounds into English, an
English-native one mangled German digits. Never fix a voice without running the
eval against it.

**Use `--repeats 3`, and do not rank on small differences.** Magpie and Qwen
sample stochastically; a single run swings small categories by ±0.1 WER. Even at
three repeats the overall mean still moves by about **0.02 WER between runs** —
measured, by running a byte-identical configuration twice: 0.182 and 0.160. Below
that it is noise.

**The measured WER includes the judge's own errors.** It is an upper bound on the
TTS error, so read per-category *deltas*, not absolute values. Word-level WER
also over-penalizes German compounds when the transcript hyphenates them — check
CER there.

**Never verify a categorical finding with a single STT model.** Voxtral's numbers
once looked bad because the *judge* was wrong: granite-speech-4.1-2b drops number
words, which cost roughly 0.02 WER and triggered a
[since-retracted upstream report](https://github.com/vllm-project/vllm-omni/issues/5510).
Under whisper-large-v3 the same audio moved from 0.177 into the leading group.
Judge selection is measurable here, not a matter of taste —
[`docs/eval.md`](docs/eval.md).

**Magpie without the `.tn` layer normalizes nothing, silently.** The NGC container
ships without `nemo_text_processing`, so `apply_TN` is a no-op rather than an
error. The table above shows what that costs.

**Voxtral is pinned for two independent reasons.** vllm-omni 0.24.0 is broken for
this model — it ignores the input text — and on GB10 CUDA graph capture corrupts
the audio, so the stage config forces `enforce_eager`. Both in
[`docs/serving.md`](docs/serving.md).

**Generated audio stays local.** `results/` and `*.wav` are gitignored; only one
representative clip per case is published as MP3.

## Licence

Code in this repository is MIT.

**Model licences travel with the models.** As declared upstream: Magpie — NVIDIA
Open Model License; Qwen3-TTS — Apache-2.0; Chatterbox — MIT, with a Perth
watermark in generated audio; VoxCPM2 — Apache-2.0, upstream forbids
impersonation, fraud and disinformation and recommends marking AI-generated
content; Voxtral TTS — **CC BY-NC 4.0, non-commercial**.

All audio under `docs/` is AI-generated and marked as such. These notes are
pointers, not legal advice — the upstream licence texts are authoritative.

## Where this is going

The German evaluation works and the comparison is published. What is open is
named rather than planned:

- **Prosody and naturalness are not measured.** WER says a sentence was
  understood, not that it sounded good. The listening pages exist for a human
  spot-check; there is no MOS.
- **Homograph stress** (*umfahren* against *umfahren*) is invisible in a
  transcript and therefore outside this method entirely.

Issues and pull requests are welcome; nobody is on call for them.

## Going deeper

| | |
|---|---|
| [`docs/serving.md`](docs/serving.md) | All five adapters: build, start, endpoints, environment, image layering, and the Voxtral pins |
| [`docs/eval.md`](docs/eval.md) | The testset by category, how the eval runs, what it writes, capped WER and runaway detection, how the judge was chosen and which candidates were rejected |

## Part of the southbyte family

- [southbyte-core](https://github.com/MvdB/southbyte-core) — shared index
- [southbyte-sync](https://github.com/MvdB/southbyte-sync) — HuggingFace mirror → local model store
- [southbyte-vllm](https://github.com/MvdB/southbyte-vllm) — vLLM runner + LLM testplan
- [southbyte-image](https://github.com/MvdB/southbyte-image) — text-to-image serving + evaluation
- [southbyte-music](https://github.com/MvdB/southbyte-music) — text-to-music serving + web interface
- [southbyte-results](https://github.com/MvdB/southbyte-results) — cross-modality results site
- [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) — GB10 profiles, kernels, benchmarks
- **southbyte-tts** — TTS/STT serving + German evaluation *(this repository)*

---

Built by [southbyte](https://southbyte.de).
