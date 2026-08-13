#!/bin/bash
set -e

# Symlink large weight files from /weights into models/
ln -sf /weights/CT-CLIP_v2.pt                                    /opt/app/models/CT-CLIP_v2.pt
ln -sf /weights/vae_step11000.pt                                 /opt/app/models/vae_step11000.pt
ln -sf /weights/ctflow_vae_ft_step25k/denoiser_ema/diffusion_pytorch_model.safetensors /opt/app/models/ctflow_vae_ft_step25k/denoiser_ema/diffusion_pytorch_model.safetensors
ln -sf /weights/FLUX_vae_checkpoint/diffusion_pytorch_model.safetensors /opt/app/models/FLUX_vae_checkpoint/diffusion_pytorch_model.safetensors
ln -sf /weights/FLUX_vae_checkpoint/rgb_imagenet.pt              /opt/app/models/FLUX_vae_checkpoint/rgb_imagenet.pt
ln -sf /weights/BiomedVLP-CXR-BERT-specialized/model.safetensors /opt/app/models/BiomedVLP-CXR-BERT-specialized/model.safetensors

mkdir -p /output /opt/app/cache_v3

export PYTHONPATH=/opt/app:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /opt/app

# We submit against the gpu-4xa100 tier (BATCH_SIZE in inference_ft_vae11k.py
# is tuned for its 40GB cards), so this is normally 4. Detect instead of
# hardcode so a tier mismatch degrades gracefully instead of crashing with
# "invalid device ordinal".
NPROC=$(/opt/conda/bin/python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
NPROC=${NPROC:-1}
[ "$NPROC" -lt 1 ] 2>/dev/null && NPROC=1

# v4_d3: v11's selection method (z-jerk closest to the group MEDIAN, not
# minimum -- avoids picking an over-smoothed outlier candidate) on the
# current pipeline (B-spline resample, z-start-aligned padding), with the
# final anisotropic-diffusion denoise step re-enabled at 3 iterations
# (CurvatureAnisotropicDiffusionImageFilter, down from v4.py's 5).
# v4_nodenoise (this same v11 selection, denoise skipped entirely) scored
# best in a small local CTNet-Frechet-proxy smoke test (N=2, noisy); this
# tries a middle ground -- light denoise instead of none -- against the real
# leaderboard.
# v9d0/v9d3/v9d5/v9d7 (v9's noise-based selection, denoise iterations 0/3/5/7)
# are left in the image, unused.
# inference_ft_vae11k.py / _v2.py / _v3.py / _spacing.py / _v4.py (denoise=5
# version) / _v4_nodenoise.py are left in the image, unused.
# _v4_d3_long: _v4_d3.py with the two selection keys swapped -- longest wins,
# jerk-closest-to-median only breaks ties -- plus the gate raised to 245 in the
# Dockerfile. Everything else (K=5, B-spline resample, z-start-aligned padding,
# anisotropic-diffusion denoise x3, k=0 left unseeded) is unchanged from the
# configuration that scored ~0.26.
#
# The platform logs separate on length, not jerk: the ~0.26 runs each had 4 of
# 5 candidates fail the gate, so the jerk rule never ran and 272/259-frame
# volumes were published; the ~0.34 run had four survivors, let jerk choose,
# and published 252 -- and on the other prompt all five failed, the fallback
# fired, and it published 227 frames. FID and CLIPScore barely moved between
# those runs while FVD did, which is what a z-extent problem looks like.
#
# Other scripts in the image, unused: _v4.py (denoise x5, jerk-primary),
# _v4_d3.py, _v4_d3_seeded.py (k=0 seeded), _v4_nodenoise.py (no denoise),
# _v9d0/_v9d3/_v9d7.py (v9 in-plane-noise selection, denoise 0/3/7).
exec /opt/conda/bin/torchrun --nproc_per_node="$NPROC" inference_ft_vae11k_v4_d3_long.py
