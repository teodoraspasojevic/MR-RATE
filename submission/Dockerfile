# VLM3D `mr-volume-generation` submission — thin image, weights mounted at /weights.
#
# Build from the REPOSITORY ROOT (the context needs contrastive-pretraining/ and submission/):
#
#     docker buildx build --platform linux/amd64 -f submission/Dockerfile -t mr-r2v-vlm3d-challenge:latest .
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
#
# torch/torchvision install with --no-deps: torch==2.5.1+cu121's own wheel metadata pins
# nvidia-cudnn-cu12==9.1.0.70 exactly, and that specific patch build has since been pulled from
# the index that serves it (confirmed 2026-08-14 -- pip's own reported version list jumps straight
# from 9.0.0.312 to 9.1.1.17, nothing in between). Every other nvidia-*-cu12 sibling pin from the
# same metadata still resolves untouched; only cudnn is bumped one patch within the same 9.1.x
# line below, so the torch/torchvision wheels installed are byte-identical to what was trained
# and evaluated against -- only the auxiliary cudnn runtime shim differs, by a patch version.
RUN pip install --no-cache-dir --no-deps \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        filelock typing-extensions networkx jinja2 fsspec pillow setuptools \
        sympy==1.13.1 \
        triton==3.1.0 \
        nvidia-cuda-nvrtc-cu12==12.1.105 \
        nvidia-cuda-runtime-cu12==12.1.105 \
        nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cudnn-cu12==9.1.1.17 \
        nvidia-cublas-cu12==12.1.3.1 \
        nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 \
        nvidia-cusolver-cu12==11.4.5.107 \
        nvidia-cusparse-cu12==12.1.0.106 \
        nvidia-nccl-cu12==2.21.5 \
        nvidia-nvtx-cu12==12.1.105 \
        --index-url https://download.pytorch.org/whl/cu121 \
    # monai==1.5.2, not 1.6.0: 1.6.0 raised its own install_requires floor to torch>=2.8.0, and
    # since this install has no --no-deps and no torch pin of its own, pip silently satisfies that
    # floor by upgrading the torch==2.5.1 pinned two steps above -- to whatever cu13x wheel is
    # newest on the day of the build -- which then breaks the torch==2.5.1-pinned torchvision==0.20.1
    # ABI (confirmed: `torchvision::nms does not exist` at text-encoder load time). 1.5.2 is the
    # newest monai release whose floor (torch>=2.4.1) the 2.5.1 pin still satisfies, and it still
    # carries MaisiDownsample (verified against the 1.5.2 tag). If monai is ever bumped again,
    # re-run `pip check` inside the built image before trusting the pin held.
    && pip install --no-cache-dir \
        monai==1.5.2 \
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
    && (find /opt/conda -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true)

# No network egress at runtime. Every checkpoint arrives through /weights, so nothing should ever
# reach for the Hub — these make that a loud failure instead of a 30-second DNS hang per encoder.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The in-container tree:
#
#   /opt/app/contrastive-pretraining/mrrate_r2v/   our package (NVIDIA's model-loading pieces it
#                                                   needs are copied into models/nvidia.py itself,
#                                                   Apache-2.0-attributed there -- no separate copy)
#   /opt/app/{predict.py,entrypoint.sh}
#   /opt/app/models/       -> symlinks into /weights          (made by entrypoint.sh)
#   /opt/app/pretrained/   -> /weights/pretrained             (made by entrypoint.sh)
ENV PYTHONPATH=/opt/app/contrastive-pretraining:/opt/app \
    MRRATE_PRETRAINED_DIR=/opt/app/pretrained \
    R2V_MODELS_DIR=/opt/app/models

# Submission-time knobs, baked in rather than left to predict.py's own fallback defaults (see
# submission/README.md "Knobs" for the full table this mirrors). The real platform passes no
# runtime flags or env vars at all -- `docker run -e ...` only ever works for local testing -- so
# once these are here, the ONLY participant-controlled input left at container-run time is which
# weights.zip is mounted at /weights (chosen at `forithmus submit --weights ...` time, not at
# `docker run` time). Change a value and rebuild; do not rely on `-e` for anything that needs to
# reach the real submission.
ENV R2V_ADAPTER=adapter.pt \
    R2V_NUM_INFERENCE_STEPS=30 \
    R2V_REPORT_GUIDANCE_SCALE=3 \
    R2V_MODALITY_GUIDANCE_SCALE=10.0 \
    R2V_GEOMETRY_MODE=policy \
    R2V_OUTPUT_DTYPE=float32

WORKDIR /opt/app
RUN mkdir -p /opt/app/models /opt/app/pretrained /input /output /checkpoint \
    && chown -R appuser:appuser /opt/app /input /output /checkpoint

COPY --chown=appuser:appuser contrastive-pretraining/mrrate_r2v \
                             /opt/app/contrastive-pretraining/mrrate_r2v
COPY --chown=appuser:appuser submission/predict.py    /opt/app/predict.py
COPY --chown=appuser:appuser submission/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh

USER appuser
ENTRYPOINT ["/opt/app/entrypoint.sh"]
