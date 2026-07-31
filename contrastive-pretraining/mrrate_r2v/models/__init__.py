"""Model loading. One module per model family; each is the only place that family's code is
imported, so a change to a vendored layout touches exactly one file.

    nvidia.py                   the vendored NV-Generate-CTMR autoencoder + diffusion UNet
    report_conditioned_unet.py  that same diffusion UNet plus report cross-attention
"""
