# tencent/AuK und AuK-Flash auf DGX Spark (GB10, aarch64, sm120).
#
# Basis: das Qwen3-TTS-Image (NGC-torch + aus Source gebautes torchaudio).
#
# Der Hersteller pinnt torch>=2.7,<2.8; wir fahren NGC-torch 2.13, das auf
# sm120 nötig ist. Deshalb dasselbe Muster wie bei Chatterbox und Breeze:
# --no-deps für das Paket selbst, Abhängigkeiten von Hand.
#
# flash-attn wird NICHT gebraucht: src/auk/model/modules.py nutzt es nur bei
# attn_backend="flash_attn", Vorgabe ist "torch". Damit entfällt die
# sm90-Kernel-Falle, die bei Breeze noch zu klären war.
FROM spark-qwen3-tts:v1

# Fester Commit statt main: sonst misst ein Nachbau anderen Code.
ARG AUK_COMMIT=5684966d4898dc4024b41a26efc08d0409e3b942

RUN git clone https://github.com/Tencent-Hunyuan/AuK.git /opt/auk \
    && cd /opt/auk \
    && git checkout "${AUK_COMMIT}" \
    && rm -rf /opt/auk/.git /opt/auk/assets

# Ohne die torch-Familie (die kommt aus dem Basis-Image) und ohne das Paket
# selbst — der Adapter legt /opt/auk/src auf den Pfad.
RUN pip install --no-cache-dir \
        "qwen-omni-utils>=0.0.9" "omegaconf>=2.3.0" torchdiffeq \
        "x_transformers>=1.31.14" safetensors librosa soundfile

# Guard wie in den übrigen abgeleiteten Images: bricht der Import, scheitert
# schon der Build und nicht erst der Messlauf.
RUN python3 -c "import torch, torchaudio; assert torch.version.cuda; \
import torchaudio._extension.utils as u; u._check_cuda_version(); \
import sys; sys.path.insert(0, '/opt/auk/src'); \
from auk.infer.infer_auk import AukInfer; \
import soundfile as sf; assert hasattr(sf, 'info'); \
import transformers; print('Import OK', transformers.__version__)"

COPY server_auk.py /opt/auk_server/server.py

ENV AUK_CODE_DIR=/opt/auk/src
EXPOSE 8014
ENTRYPOINT ["python3", "/opt/auk_server/server.py"]
