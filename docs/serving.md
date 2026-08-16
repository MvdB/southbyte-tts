# The five adapters

All expose the same OpenAI-compatible API, so the evaluator only needs a
different `--tts` URL. Models are mounted **read-only** from `~/hf_models`
(`HF_MODELS_DIR`) — no downloads at serve time, with one exception: Magpie
fetches its NanoCodec vocoder from HuggingFace on first start.

| Endpoint | |
|---|---|
| `POST /v1/audio/speech` | OpenAI Audio API compatible; returns WAV, mono, 16 bit |
| `GET /v1/voices` | Voices and languages, queried from the loaded model |
| `GET /health` | Liveness and model status |

Environment: `MAGPIE_NEMO_PATH` / `QWEN_TTS_PATH` (checkpoint path inside the
container), `MODEL_DIR` (Qwen variant), `QWEN_TTS_VOICE_INSTRUCT` (VoiceDesign
voice description, default: a German newsreader), `HOST_PORT`, `HF_MODELS_DIR`.

## Build and start

```bash
cd serving

# Magpie — base image has no German text normalization:
docker build -t spark-magpie-tts:v1 .
# + German TN (needs prebuilt OpenFst/pynini aarch64 artifacts in the build
#   context — see the Dockerfile.tn header):
docker build -t spark-magpie-tts:v1-tn -f Dockerfile.tn .
IMAGE=spark-magpie-tts:v1-tn ./run_server.sh          # port 8001

# Qwen3-TTS — CustomVoice or VoiceDesign, selected via MODEL_DIR:
docker build -t spark-qwen3-tts:v1 -f Dockerfile.qwen3tts .
./run_qwen3tts.sh                                     # port 8002
MODEL_DIR=Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign ./run_qwen3tts.sh

# Chatterbox and VoxCPM2 — both build FROM spark-qwen3-tts:v1:
docker build -t spark-chatterbox:v1 -f Dockerfile.chatterbox .
VOICES_DIR=$PWD/../voices ./run_chatterbox.sh         # port 8003
docker build -t spark-voxcpm:v1 -f Dockerfile.voxcpm .
./run_voxcpm.sh                                       # port 8004

# Voxtral-4B-TTS — no adapter of ours, vLLM-Omni serves it natively:
docker build -t spark-voxtral-tts:v1 -f Dockerfile.voxtral .
./run_voxtral_tts.sh                                  # port 8005
```

**Image layering is not incidental.** `Dockerfile.chatterbox` and
`Dockerfile.voxcpm` build *from* `spark-qwen3-tts:v1` because it already carries
NGC torch plus a source-built torchaudio; `Dockerfile.tn` builds from
`spark-magpie-tts:v1`. Each derived Dockerfile ends its pip install with an
import and CUDA guard — keep those when touching dependencies, they are what
turns a broken layer into a failed build instead of a silent runtime surprise.

## VoiceDesign has no fixed speakers

The voice *is* an instruct text. Named presets are selected through the ordinary
`voice` field, so one server serves them all:

```bash
curl … -d '{"input":"…","voice":"de_male_news","language":"de"}'
```

Presets: `de_female_news`, `de_male_news`, `de_female_calm`, `de_male_young`.
`QWEN_TTS_VOICE_DESIGN` sets the default, `QWEN_TTS_VOICE_INSTRUCT` overrides it
with free text. Without preset names every run would have been called `design`
and they would have collided in `docs/`.

## Chatterbox clones from a WAV

`/voices/<name>.wav` is the reference; the clone is zero-shot. Generated audio
carries a Perth watermark — that is upstream behaviour, not a setting.

## The Voxtral pins, learned the hard way

**vllm-omni 0.24.0 — the latest stable — is broken for this model.** Text
conditioning is lost: the model babbles fluently in the voice's language and
ignores the input entirely. It works from **0.25.0rc1** on the `v0.25.1` vLLM
base, and the two must match in minor version. Same trap as vllm/vllm-omni
elsewhere in this family.

**On GB10 (sm_120), CUDA graph capture corrupts the audio.** The stage config
`vllm-omni/voxtral_tts_stages.yaml` — now in
[southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles),
mounted via `$SPARK_PROFILES_DIR` — therefore forces `enforce_eager`, at a real
time factor of about 4.

**The native endpoint behaves differently.** It requires `model` in the payload
and returns no timing headers. `roundtrip_eval.py` detects this through
`/v1/models` and falls back to WAV-length over wall-time.

**Voice cloning is impossible**, not merely unimplemented: the audio-encoder
weights are not released.
