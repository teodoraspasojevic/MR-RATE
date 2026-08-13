#!/bin/bash
# Pack all weight files into weights.zip (store mode, no compression).
# Run from anywhere; output goes next to this script.

set -e

MODELS=/vol/idea_ramses/ba49mefe/CTNet_Synthetic/docker/models
OUT=/vol/idea_ramses/ba49mefe/CTNet_Synthetic/docker_ft/weights.zip

rm -f "$OUT"

cd "$MODELS"
zip -r0 "$OUT" \
    CT-CLIP_v2.pt \
    vae_step11000.pt \
    ctflow_vae_ft_step25k/denoiser_ema/config.json \
    ctflow_vae_ft_step25k/denoiser_ema/diffusion_pytorch_model.safetensors \
    FLUX_vae_checkpoint/config.json \
    FLUX_vae_checkpoint/diffusion_pytorch_model.safetensors \
    FLUX_vae_checkpoint/rgb_imagenet.pt \
    BiomedVLP-CXR-BERT-specialized/model.safetensors

echo "Done: $OUT"
du -sh "$OUT"
