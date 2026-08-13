FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bzip2 libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required by platform)
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Install Miniconda (Python 3.9)
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh \
    -o /tmp/conda.sh \
    && bash /tmp/conda.sh -b -p /opt/conda \
    && rm /tmp/conda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Install Python packages into base conda env
RUN pip install --no-cache-dir \
        torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        xformers==0.0.25.post1 \
        torchdiffeq==0.2.4 \
        diffusers==0.30.3 \
        transformers==4.45.1 \
        accelerate==0.34.2 \
        einops \
        omegaconf \
        nibabel \
        SimpleITK \
        numpy \
        scipy \
        opencv-python-headless \
        safetensors \
        sentencepiece \
        peft \
        tqdm \
        pillow \
        imageio \
        scikit-learn \
        scikit-image \
        wandb \
        timm \
        decord \
        moviepy \
        matplotlib \
        pandas \
        jsonschema \
        "numpy<2" \
    && conda clean -afy \
    && find /opt/conda -name "*.pyc" -delete \
    && find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
# beartype/vector-quantize-pytorch were only needed for ct_clip_src's CTViT
# import (the CT-CLIP scorer used by v2/v3, not by the active v4 entrypoint).
# Re-enable (with --no-deps -- otherwise it drags torch up to 2.8.0 and
# breaks torchvision) if switching back to a CT-CLIP-filter script:
#   RUN pip install --no-cache-dir --no-deps \
#           beartype==0.20.2 \
#           vector-quantize-pytorch==1.31.0

# Offline mode — no network calls at runtime
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV PYTHONPATH=/opt/app
# noise+length candidate filter (inference_ft_vae11k_v4.py); K=1 disables it.
# Matches what we actually tested locally -- don't bump without re-testing.
ENV CTFLOW_NOISE_K=5
# 245, not 240: at 240 the platform logs show prompts where several candidates
# clear the gate, and the selector then publishes a short one (252, and 227 via
# the all-failed fallback). A 227-frame volume is ~340mm of anatomy in a 384mm
# output grid -- ~15 of the 128 slices become trailing air and ~12% of the
# anatomy is simply missing. Runs that published 272/259 scored ~0.26; the run
# that published 252/227 scored ~0.34, with FID and CLIPScore barely moving,
# which is what a z-extent problem looks like.
ENV CTFLOW_NOISE_MIN_FRAMES=245
# 12345 is the script's own default, so this line is a no-op that documents the
# value rather than changing behaviour -- deliberate, to keep the active script
# byte-equivalent in behaviour to the runs that scored ~0.26.
#
# It only reaches candidates k>=1. The active script leaves k=0 unseeded (it
# runs on whatever OS-entropy state the process started with), so runs are not
# reproducible. A seeded variant exists (_v4_d3_seeded.py) if that is wanted,
# but a sweep over 42/1337/3407/12345 found no seed worth pinning: the ranking
# did not reproduce across three local 10-prompt rounds (correlation +0.02
# between rounds 1-2, +0.17 between rounds 2-3, the latter two both on real
# distinct CT-RATE reports), and the spread between seeds matched the spread
# between repeats of a single seed.
ENV CTFLOW_SEED=12345

# App directory
WORKDIR /opt/app
RUN mkdir -p /input /output /weights /opt/app/cache_v3 \
    && chown -R appuser:appuser /opt/app /input /output /opt/conda

# Copy code and model configs (no weight files)
COPY --chown=appuser:appuser CTFlow                     /opt/app/CTFlow
COPY --chown=appuser:appuser models                     /opt/app/models
COPY --chown=appuser:appuser inference_ft_vae11k_v4.py    /opt/app/inference_ft_vae11k_v4.py
COPY --chown=appuser:appuser inference_ft_vae11k_v9d3.py  /opt/app/inference_ft_vae11k_v9d3.py
COPY --chown=appuser:appuser inference_ft_vae11k_v9d7.py  /opt/app/inference_ft_vae11k_v9d7.py
COPY --chown=appuser:appuser inference_ft_vae11k_v9d0.py  /opt/app/inference_ft_vae11k_v9d0.py
COPY --chown=appuser:appuser inference_ft_vae11k_v4_nodenoise.py /opt/app/inference_ft_vae11k_v4_nodenoise.py
COPY --chown=appuser:appuser inference_ft_vae11k_v4_d3.py /opt/app/inference_ft_vae11k_v4_d3.py
COPY --chown=appuser:appuser inference_ft_vae11k_v4_d3_seeded.py /opt/app/inference_ft_vae11k_v4_d3_seeded.py
COPY --chown=appuser:appuser inference_ft_vae11k_v4_d3_long.py /opt/app/inference_ft_vae11k_v4_d3_long.py
COPY --chown=appuser:appuser entrypoint.sh                 /opt/app/entrypoint.sh
# Earlier experiment scripts entrypoint.sh no longer runs -- not copied into
# the image. Re-add if reviving one of them:
#   COPY --chown=appuser:appuser ct_clip_src                     /opt/app/ct_clip_src
#   COPY --chown=appuser:appuser inference_ft_vae11k.py           /opt/app/inference_ft_vae11k.py
#   COPY --chown=appuser:appuser inference_ft_vae11k_v2.py        /opt/app/inference_ft_vae11k_v2.py
#   COPY --chown=appuser:appuser inference_ft_vae11k_v3.py        /opt/app/inference_ft_vae11k_v3.py
#   COPY --chown=appuser:appuser inference_ft_vae11k_spacing.py   /opt/app/inference_ft_vae11k_spacing.py
RUN chmod +x /opt/app/entrypoint.sh

USER appuser

ENTRYPOINT ["/opt/app/entrypoint.sh"]
