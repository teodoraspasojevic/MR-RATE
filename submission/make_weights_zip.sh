#!/usr/bin/env bash
# Assemble the weights.zip that `forithmus submit --weights` mounts at /weights.
#
#     ./submission/make_weights_zip.sh C                       # arm C, adapter_step0004200.pt
#     ./submission/make_weights_zip.sh D adapter_best_fvd.pt    # arm D, a different checkpoint
#     OUT_DIR=/hnvme/workspace/.../submission ./submission/make_weights_zip.sh C
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
set -euo pipefail

ARM="${1:-}"
CKPT="${2:-adapter_step0004200.pt}"
WS="${WS:-/hnvme/workspace/y100dc19-nvidia-mri-brain}"
OUT_DIR="${OUT_DIR:-${WS}/submission}"

usage() {
    cat >&2 <<'EOF'
usage: make_weights_zip.sh <A|B|C|D> [adapter_checkpoint_filename]

  A  cxr_bert_cls      pooled CLS,     768x1     encoders: BiomedVLP-CXR-BERT-specialized
  B  cxr_bert_tokens   token sequence, 768x512   encoders: BiomedVLP-CXR-BERT-specialized
  C  radbert_tokens    token sequence, 768x512   encoders: RadBERT-RoBERTa-4m
  D  report2ct_style   sectioned fusion, 2560x2  encoders: MedEmbed-large-v0.1,
                                                           Bio_ClinicalBERT,
                                                           BiomedVLP-CXR-BERT-specialized
EOF
    exit 2
}

case "$ARM" in
    A) RUN=r2v_final_A_cxr_bert_cls;     ENCODERS=("BiomedVLP-CXR-BERT-specialized") ;;
    B) RUN=r2v_final_B_cxr_bert_tokens;  ENCODERS=("BiomedVLP-CXR-BERT-specialized") ;;
    C) RUN=r2v_final_C_radbert_tokens;   ENCODERS=("RadBERT-RoBERTa-4m") ;;
    D) RUN=r2v_final_D_report2ct_style;  ENCODERS=("MedEmbed-large-v0.1" "Bio_ClinicalBERT" "BiomedVLP-CXR-BERT-specialized") ;;
    *) usage ;;
esac

ADAPTER="${WS}/runs/${RUN}/${CKPT}"
BASE="${WS}/models/diff_unet_3d_rflow-mr-brain_v0.pt"
VAE="${WS}/models/autoencoder_v1.pt"

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
    src="${WS}/pretrained/${enc}"
    [ -d "$src" ] || { echo "missing text encoder: $src" >&2; exit 1; }
    # Weight + tokenizer + config files only. A pretrained snapshot directory can also carry a
    # .git, a README and a duplicate pytorch_model.bin beside model.safetensors; those are pure
    # upload cost.
    mkdir -p "$STAGE/$enc"
    find "$src" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.txt' -o -name '*.model' -o -name '*.safetensors' \) \
        -exec cp -L {} "$STAGE/$enc/" \;
    if ! ls "$STAGE/$enc"/*.safetensors >/dev/null 2>&1; then
        echo "no .safetensors in $src; falling back to pytorch_model.bin" >&2
        cp -L "$src/pytorch_model.bin" "$STAGE/$enc/"
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
