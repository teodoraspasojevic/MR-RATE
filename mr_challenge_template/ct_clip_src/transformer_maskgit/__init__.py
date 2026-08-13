# Trimmed re-export: the scorer only needs CTViT. The upstream __init__ also imported
# MaskGITTransformer / videotextdataset / ctvit_trainer, which pull in accelerate, cv2
# and pandas -- none of which the image needs just to rank candidates.
from transformer_maskgit.ctvit import CTViT

__all__ = ["CTViT"]
