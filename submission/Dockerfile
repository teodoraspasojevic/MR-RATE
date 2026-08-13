# VLM3D `mr-volume-generation` submission — thin image, weights mounted at /weights.
#
# Build from the REPOSITORY ROOT (the context needs both NV-Generate-CTMR/ and
# contrastive-pretraining/):
#
#     docker buildx build --platform linux/amd64 -f submission/Dockerfile -t mrgen-thin:latest .
#
# `--platform linux/amd64` is not optional if you build on an Apple-silicon Mac: the platform's
# validator rejects anything that is not amd64, and it rejects it after you have paid for the
# upload.
#
# Why a CUDA base and not python:3.12-slim: the slim images carry no libcudart, so
# torch.cuda.is_available() silently returns False and the run burns its whole time budget on
# CPU. This is the organizers' own documented first troubleshooting entry.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl bzip2 ca-certificates \
        libsm6 libxext6 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. The platform's validator refuses an image whose effective USER is root, and it
# runs before the platform's own entrypoint trampoline mutates the runtime user back to root for
# its FUSE mounts — so the image must be shipped non-root even though it does not run that way.
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Miniconda for Python 3.12, matching the interpreter the adapters were trained and evaluated
# under (3.12.3). Ubuntu 22.04's system Python is 3.10.
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-2-Linux-x86_64.sh \
        -o /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p /opt/conda \
    && rm /tmp/conda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Versions are pinned to the exact set the four adapters were trained and locally evaluated with
# (see submission/README.md §"Environment fidelity"). Floating any of these turns a leaderboard
# score into a number we cannot reproduce locally, which is the whole reason to pin.
RUN pip install --no-cache-dir \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        monai==1.6.0 \
        transformers==5.14.1 \
        tokenizers==0.22.2 \
        safetensors==0.8.0 \
        huggingface-hub==1.23.0 \
        numpy==2.5.1 \
        scipy==1.18.0 \
        nibabel==5.4.2 \
        einops==0.8.2 \
        scikit-image==0.26.0 \
        sentencepiece==0.2.2 \
        PyYAML==6.0.3 \
        tqdm \
    && conda clean -afy \
    && find /opt/conda -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# No network egress at runtime. Every checkpoint arrives through /weights, so nothing should ever
# reach for the Hub — these make that a loud failure instead of a 30-second DNS hang per encoder.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The in-container tree mirrors the repository, and that is load-bearing at exactly one place:
# `mrrate_r2v/models/nvidia.py` finds the vendored NVIDIA code as `parents[3]/NV-Generate-CTMR`.
# Flattening `contrastive-pretraining/` away would make that resolve to `/NV-Generate-CTMR`.
#
#   /opt/app/NV-Generate-CTMR/                     vendored NVIDIA model + configs (Apache-2.0)
#   /opt/app/contrastive-pretraining/mrrate_r2v/   our package
#   /opt/app/{predict.py,routing.py,entrypoint.sh}
#   /opt/app/models/       -> symlinks into /weights          (made by entrypoint.sh)
#   /opt/app/pretrained/   -> /weights/pretrained             (made by entrypoint.sh)
ENV PYTHONPATH=/opt/app/contrastive-pretraining:/opt/app \
    MRRATE_PRETRAINED_DIR=/opt/app/pretrained \
    R2V_MODELS_DIR=/opt/app/models

WORKDIR /opt/app
RUN mkdir -p /opt/app/models /opt/app/pretrained /input /output /checkpoint \
    && chown -R appuser:appuser /opt/app /input /output /checkpoint

COPY --chown=appuser:appuser NV-Generate-CTMR/scripts  /opt/app/NV-Generate-CTMR/scripts
COPY --chown=appuser:appuser NV-Generate-CTMR/configs  /opt/app/NV-Generate-CTMR/configs
COPY --chown=appuser:appuser NV-Generate-CTMR/LICENSE  /opt/app/NV-Generate-CTMR/LICENSE
COPY --chown=appuser:appuser NV-Generate-CTMR/LICENSE.weights /opt/app/NV-Generate-CTMR/LICENSE.weights
COPY --chown=appuser:appuser contrastive-pretraining/mrrate_r2v \
                             /opt/app/contrastive-pretraining/mrrate_r2v
COPY --chown=appuser:appuser submission/predict.py    /opt/app/predict.py
COPY --chown=appuser:appuser submission/routing.py    /opt/app/routing.py
COPY --chown=appuser:appuser submission/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

USER appuser
ENTRYPOINT ["/opt/app/entrypoint.sh"]
