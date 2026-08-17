#!/usr/bin/env bash
# Assemble the weights.zip that `forithmus submit --weights` mounts at /weights.
#
#     ./submission/make_weights_zip.sh C                        # arm C, adapter_best_fvd.pt (its default)
#     ./submission/make_weights_zip.sh D adapter_step0004200.pt  # arm D, a different checkpoint
#     WS=/hnvme/workspace/y100dc19-nvidia-mri-brain ./submission/make_weights_zip.sh C   # build on Helma instead
#
# Layout produced (files at the ZIP ROOT — a parent folder breaks the platform's mount):
#
#     adapter.pt                          the chosen arm's adapter, renamed
#     autoencoder_v1.pt                   frozen MAISI VAE
#     diff_unet_3d_rflow-mr-brain_v0.pt   frozen NV-Generate-MR-Brain base UNet
#     <EncoderDir>/ ...                    only the text encoders this arm actually needs
#
# Only the arm's own encoders are included: shipping all six costs ~4 GB of upload for nothing,
# and a wrong-but-same-width encoder silently produces confident nonsense, so it is better for a
# missing one to be a loud FileNotFoundError than for an extra one to be sitting there.
#
# WS layout expected (this project's own storage, not Helma's $WORKSPACE):
#     $WS/models/base-nvidia-model/{diff_unet_3d_rflow-mr-brain_v0.pt,autoencoder_v1.pt}
#     $WS/models/text-encoders/<EncoderDir>/
#     $WS/models/r2v-adapters/<run>/<checkpoint>.pt
set -euo pipefail

ARM="${1:-}"
WS="${WS:-/vol/idea_ramses/va47zasy/VLM3D-MICCAI-2026}"
OUT_DIR="${OUT_DIR:-${WS}/submission/weights}"

usage() {
    cat >&2 <<'EOF'
usage: make_weights_zip.sh <A|B|C|D|E> [adapter_checkpoint_filename]

  A  cxr_bert_cls      pooled CLS,     768x1     encoders: BiomedVLP-CXR-BERT-specialized
  B  cxr_bert_tokens   token sequence, 768x512   encoders: BiomedVLP-CXR-BERT-specialized
  C  radbert_tokens    token sequence, 768x512   encoders: RadBERT-RoBERTa-4m
  D  report2ct_style   sectioned fusion, 2560x2  encoders: MedEmbed-large-v0.1,
                                                           Bio_ClinicalBERT,
                                                           BiomedVLP-CXR-BERT-specialized
  E  report2ct_style_meta  D + acquisition token, 2560x3   same three encoders

Each arm's default checkpoint filename is whatever that run actually saved (last.pt vs
best_fvd.pt differ by arm) -- pass a second argument only to override it.
EOF
    exit 2
}

case "$ARM" in
    A) RUN=r2v_final_A_cxr_bert_cls;     CKPT_DEFAULT=adapter_last.pt;     ENCODERS=("BiomedVLP-CXR-BERT-specialized") ;;
    B) RUN=r2v_final_B_cxr_bert_tokens;  CKPT_DEFAULT=adapter_last.pt;     ENCODERS=("BiomedVLP-CXR-BERT-specialized") ;;
    C) RUN=r2v_final_C_radbert_tokens;   CKPT_DEFAULT=adapter_best_fvd.pt; ENCODERS=("RadBERT-RoBERTa-4m") ;;
    D) RUN=r2v_final_D_report2ct_style;  CKPT_DEFAULT=adapter_last.pt;     ENCODERS=("MedEmbed-large-v0.1" "Bio_ClinicalBERT" "BiomedVLP-CXR-BERT-specialized") ;;
    E) RUN=r2v_final_E_report2ct_style_meta; CKPT_DEFAULT=adapter_last.pt; ENCODERS=("MedEmbed-large-v0.1" "Bio_ClinicalBERT" "BiomedVLP-CXR-BERT-specialized") ;;
    *) usage ;;
esac
CKPT="${2:-$CKPT_DEFAULT}"

ADAPTER="${WS}/models/r2v-adapters/${RUN}/${CKPT}"
BASE="${WS}/models/base-nvidia-model/diff_unet_3d_rflow-mr-brain_v0.pt"
VAE="${WS}/models/base-nvidia-model/autoencoder_v1.pt"

for f in "$ADAPTER" "$BASE" "$VAE"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
echo "staging arm $ARM ($RUN / $CKPT) in $STAGE"

cp -L "$ADAPTER" "$STAGE/adapter.pt"
cp -L "$BASE" "$STAGE/$(basename "$BASE")"
cp -L "$VAE"  "$STAGE/$(basename "$VAE")"
for enc in "${ENCODERS[@]}"; do
    src="${WS}/models/text-encoders/${enc}"
    [ -d "$src" ] || { echo "missing text encoder: $src" >&2; exit 1; }
    # Weight + tokenizer + config files only. A pretrained snapshot directory can also carry a
    # .git, a README and a duplicate pytorch_model.bin beside model.safetensors; those are pure
    # upload cost.
    mkdir -p "$STAGE/$enc"
    find "$src" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.txt' -o -name '*.model' -o -name '*.safetensors' \) \
        -exec cp -L {} "$STAGE/$enc/" \;
    if ! ls "$STAGE/$enc"/*.safetensors >/dev/null 2>&1; then
        # Not a safe fallback: transformers==5.14.1 (pinned in submission/Dockerfile) refuses
        # `torch.load` below torch 2.6 (CVE-2025-32434 guard), and torch is pinned at 2.5.1 there.
        # A pytorch_model.bin-only encoder loads fine in a plain `python -m mrrate_r2v...` run but
        # crashes AutoModel.from_pretrained inside the container at text-encoder build time -- this
        # exact gap broke arms C and D (RadBERT-RoBERTa-4m, Bio_ClinicalBERT) until both were
        # converted in place (see git history around 2026-08-14). Convert once with:
        #   python -c "import torch; from safetensors.torch import save_file; \
        #       sd = torch.load('$src/pytorch_model.bin', map_location='cpu', weights_only=True); \
        #       save_file({k: v.clone(memory_format=torch.contiguous_format) for k, v in sd.items()}, \
        #       '$src/model.safetensors')"
        # then re-run this script -- do not remove this exit.
        echo "FATAL: no .safetensors in $src, only pytorch_model.bin -- this will build a zip that" >&2
        echo "crashes at container runtime under the pinned torch/transformers versions. Convert" >&2
        echo "it to model.safetensors first (see comment above this line in make_weights_zip.sh)." >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"
ZIP="${OUT_DIR}/weights_arm${ARM}.zip"
rm -f "$ZIP"
# -0 (store, no deflate): the .pt files are already-compressed tensors, so deflate spends minutes
# to save single-digit percent, and the platform unzips this on every run.
( cd "$STAGE" && zip -r0 -q "$ZIP" . -x '.*' )

echo
echo "wrote $ZIP  ($(du -h "$ZIP" | cut -f1))"
echo "contents (must show no parent-directory prefix):"
unzip -l "$ZIP" | sed 's/^/  /'
echo
echo "next:  forithmus submit submission.tar.gz --phase <phase> --tier gpu-a100-80 \\"
echo "           --time-budget 240 --weights $ZIP -d \"NV-Generate-MR-Brain + report adapter (arm $ARM)\""
