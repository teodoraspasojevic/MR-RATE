"""
Optimised CT-Flow inference script (self-contained)
==================================================
• Half-precision (FP16/BF16) everywhere
• `torch.compile` wrapped in a safe fallback (`safe_compile`)
• Batched tensor ops, pinned memory, non-blocking copies
• CUDA graphs for VAE decode
• Async disk I/O with `ThreadPoolExecutor`
• Compatible with your original folder layout / checkpoints

Save this as `inference_fast.py` (or any filename) and run directly.
"""

import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/opt/app/cache_v3/torch_compile_rank0"  # overridden per-rank in main()
os.environ["TORCHINDUCTOR_PARALLEL_COMPILE"] = "0"
os.environ["TORCHINDUCTOR_MAX_COMPILE_THREADS"] = "1"
os.environ["TRITON_CONCURRENCY"] = "1"
os.environ["TORCHINDUCTOR_TRITON_CUDAGRAPHS"] = "0"  # silence wrapper-graph warnings
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")
os.environ.setdefault("TORCH_CUDAGRAPHS", "0")
os.environ["TORCHINDUCTOR_AUTOTUNE_VERBOSE"] = "0"
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"

# ---------------------- stdlib & third-party ----------------------
import json
import time
import zipfile
import cv2
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np

import SimpleITK as sitk
import torch
import torch.distributed as dist
import datetime
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import autocast
from torchdiffeq import odeint
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------- CT-Flow helpers ---------------------------
from CTFlow.common import (  # noqa: F401 (wild-import is in original repo)
    get_vae_scaler,
    instantiate,
    instantiate_class_from_config,
    sample_latents,
    scale_latents,
    unscale_latents,
)

# ---------------------- global settings ---------------------------
_USE_BF16 = torch.cuda.is_bf16_supported()
if _USE_BF16:
    torch.set_default_dtype(torch.bfloat16)
else:
    torch.set_default_dtype(torch.float16)

import torch._inductor.config as inductor_cfg
import torch._dynamo
import logging
torch._dynamo.config.suppress_errors = True
inductor_cfg.triton.cudagraphs = False
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# deactivate gradients
torch.set_grad_enabled(False)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # overridden after DDP setup
_PIN_MEMORY = True  # host→device copies


# ---------------------- DDP helpers -------------------------------
def ddp_setup():
    if torch.cuda.is_available() and ("RANK" in os.environ or "LOCAL_RANK" in os.environ):
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    return 0, 1, 0, False


def ddp_cleanup(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()

# ---------------------- paths / constants -------------------------
_APP = "/opt/app"
TOKENIZER_NAME = f"{_APP}/models/BiomedVLP-CXR-BERT-specialized"
TEXT_MODEL_PATH = f"{_APP}/models/CT-CLIP_v2.pt"
CONFIG_PATH = f"{_APP}/models/lvfm_STDiT-L2_16f8_2_2_2_bsz16_v2_black_rate_0.3_v2/config.yaml"
DENOISER_CKPT_PATH = f"{_APP}/models/ctflow_vae_ft_step25k/denoiser_ema"
VAE_FT_CKPT_PATH = f"{_APP}/models/vae_step11000.pt"
_VAE_FT_MEAN  = -0.08395224064588547   # latent_mean from vae_step11000_scale.json
_VAE_FT_SCALE =  0.8422308181727547   # scale_factor (= 1/latent_std)
INPUT_PATH = "/input/prompts.json"
OUTPUT_DIR = "/output"
BATCH_SIZE = 16  # A100 40 GB
BLOCK_SIZE = 16

_COMPILE_SECONDS = 0.0  # accumulates all compile calls

# ---------------------- utility funcs -----------------------------


def safe_compile(model: torch.nn.Module, name: str = "model"):
    global _COMPILE_SECONDS
    try:
        t0 = time.perf_counter()
        print(f"[compile] {name:<10} starting... ", end="", flush=True)
        compiled = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=True)
        dt = time.perf_counter() - t0
        _COMPILE_SECONDS += dt
        print(f"[compile] {name:<10} finished in {dt:.1f} s (cumulative {_COMPILE_SECONDS:.1f} s)")
        return compiled
    except Exception as exc:
        print(f"[compile] Falling back to eager ({name}): {exc}")
        return model


"""
# ======= helper: 保存为 .mha =======
def save_as_mha(volume, out_path, value=1.0):
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    if volume.ndim == 4:
        volume = volume[0]  # 如果有 channel，取第一个
    T = volume.shape[0]
    keep_until = T
    for i in reversed(range(T)):
        frame = volume[i]
        if np.mean(np.abs(frame - value) < 0.1) > 0.9:
            keep_until -= 1
        else:
            break
    volume = volume[:keep_until]
    if volume.shape[0] == 0:
        volume = np.full((1, *volume.shape[1:]), value, dtype=volume.dtype)
    itk_image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(itk_image, out_path)
"""

def save_as_niigz(volume: torch.Tensor, path: str):
    """Write 3D volume to .nii.gz (NIfTI) using SimpleITK."""
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(image, path)

def save_as_mha(volume: torch.Tensor, path: str):
    """Write 3-D volume to .mha (MetaImage) using SimpleITK."""
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    sitk.WriteImage(sitk.GetImageFromArray(volume), path)

def save_as_mp4(volume: torch.Tensor, path: str, fps: int = 10):
    """
    Save a 3D or 4D volume as an MP4 video.
    
    Args:
        volume (Tensor): shape [D, H, W] or [1, D, H, W] or [C, D, H, W]
        path (str): output .mp4 path
        fps (int): frames per second
    """
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    
    # Handle shape: [1, D, H, W] or [C, D, H, W] → [D, H, W]
    if volume.ndim == 4:
        volume = volume[0]  # assume 1 channel

    assert volume.ndim == 3, "Expected shape [D, H, W] after squeeze."

    D, H, W = volume.shape

    # Normalize to [0, 255] uint8 for visualization
    vmin, vmax = np.min(volume), np.max(volume)
    norm_volume = (volume - vmin) / (vmax - vmin + 1e-8)
    norm_volume = (norm_volume * 255).astype(np.uint8)

    # Initialize video writer (grayscale to RGB)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H), isColor=True)

    for i in range(D):
        gray = norm_volume[i]                     # [H, W]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # [H, W, 3]
        writer.write(rgb)

    writer.release()
    print(f"✅ Saved {D} slices to MP4 at: {path}")

def trim_blank_tail(arr: torch.Tensor, var_thresh: float = 0.02) -> torch.Tensor:
    """Trim from the first blank frame onward (forward scan, [-1,1] space)."""
    for t in range(arr.shape[0]):
        if arr[t].float().var() < var_thresh:
            return arr[:max(1, t)]
    return arr


# If trim_blank_tail cuts a volume down to fewer than this many slices, the
# stop-mask triggered almost immediately (near-empty volume) -- treat it as
# a failed generation and retry.
QUALITY_MIN_FRAMES = 100
QUALITY_MAX_RETRIES = 2


def assess_volume_quality(arr: torch.Tensor, min_frames: int = QUALITY_MIN_FRAMES) -> bool:
    """Heuristic sanity check on a blank-trimmed volume: enough slices survived?"""
    return arr.shape[0] >= min_frames


# Inverse of finetune_vae_production.py's training-time normalization:
#   sl = clip(sl, HU_MIN, HU_MAX); sl = (sl - HU_MIN) / (HU_MAX - HU_MIN) * 2 - 1
# i.e. the VAE was trained on an asymmetric [-1000, 1400] HU window mapped to
# [-1, 1], NOT a symmetric [-1000, 1000] one. Reconstructing HU as `x * 1000`
# (the old behaviour) silently compressed and clipped every high-HU voxel.
_HU_MIN, _HU_MAX = -1000.0, 1400.0


def to_hu(arr: torch.Tensor) -> torch.Tensor:
    """Map decoder output in [-1, 1] back to Hounsfield units in [_HU_MIN, _HU_MAX]."""
    return (arr + 1.0) / 2.0 * (_HU_MAX - _HU_MIN) + _HU_MIN


# ---------------------- text encoder ------------------------------


class CTCLIPTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # AutoModel.from_pretrained loads the REAL BiomedVLP-CXR-BERT-specialized
        # pretrained weights (model.safetensors now present locally at
        # TOKENIZER_NAME). config.json has no "auto_map", so this resolves via
        # model_type="bert" to a plain BertModel -- that's fine, we only ever
        # call .last_hidden_state below, which BertModel provides identically to
        # the custom CXRBertModel class. The checkpoint's "bert.*"-prefixed keys
        # (same pattern as a BertForMaskedLM-style checkpoint) load correctly
        # into BertModel via transformers' standard prefix handling.
        # Previously this constructed CXRBertModel(config) directly (no weights
        # loaded at all -- random init) and relied on a later CT-CLIP_v2.pt
        # overlay to supply real weights; that overlay silently fails (its keys
        # are missing the "bert." prefix this state_dict expects, strict=False
        # swallows the mismatch) and is kept below for parity with the original
        # pipeline -- it's a no-op now that real weights are loaded here.
        self.text_transformer = AutoModel.from_pretrained(TOKENIZER_NAME).to(device, dtype=torch.get_default_dtype())

    def forward(self, input_ids, attention_mask):
        out = self.text_transformer(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(out)
        return out * mask  # [B, L, D]


def prompts_to_embeddings(reports: List[str], model, tokenizer) -> torch.Tensor:
    inputs = tokenizer(
        reports,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    ids = inputs["input_ids"].pin_memory() if _PIN_MEMORY else inputs["input_ids"]
    mask = (
        inputs["attention_mask"].pin_memory()
        if _PIN_MEMORY
        else inputs["attention_mask"]
    )
    with torch.no_grad(), autocast():
        emb = model(
            ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        )
    emb = emb[:, 0, :].unsqueeze(1)  # [B, 1, D] (CLS token)
    return emb / (emb.norm(dim=-1, keepdim=True) + 1e-6)


# ---------------------- generator ---------------------------------
class LatentAutoregressiveGenerator:
    def __init__(
        self,
        denoiser: torch.nn.Module,
        vae: torch.nn.Module,
        vae_scaling: float,
        cfg,
        block_size: int = BLOCK_SIZE,
        eps: float = 0.1,
    ):
        self.denoiser = denoiser
        self.vae = vae
        self.vae_scaling = vae_scaling
        self.cfg = cfg
        self.block_size = block_size
        self.eps = eps
        self.dtype = torch.get_default_dtype()
        self.use_vae_graph = False  # set True later if you ever want graphs
        self._is_first_run = True  # track first run for graph capture

        # cached latents
        with torch.no_grad():
            z1 = torch.ones(1, 3, 256, 256, dtype=self.dtype, device=device)
            enc = self.vae.encode(z1).latent_dist.sample()
            self.zero_latent = torch.zeros_like(enc)
            self.one_latent = scale_latents(
                sample_latents(cfg, enc),
                vae_scaling,
            )

        # capture VAE decode graph once (static batch*frames shape)
        self._graph_captured = False
        self._g = torch.cuda.CUDAGraph() if torch.cuda.is_available() else None
        self._decode_in: torch.Tensor | None = None
        self._decode_out: torch.Tensor | None = None

    # -----------------------------------------------------------
    def _decode(
        self,
        latents: torch.Tensor,
        *,
        max_batch_size: int = 64,  # tuned for ~40GB peak VRAM
    ) -> torch.Tensor:
        """
        Chunked VAE decode with optional CUDA-graph acceleration.
        Works for any (B, C, T, H, W); memory never spikes above ≈max_batch_size.
        """
        b, c, t, h, w = latents.shape

        # -------------- flatten video to slice-batch --------------
        flat = rearrange(latents, "b c t h w -> (b t) c h w").contiguous()
        slices = flat.split(max_batch_size, dim=0)  # tuple of tensors

        decoded_chunks = []

        for chunk in slices:
            # -----------------------------------------------------------------
            # per-shape CUDA-graph cache (keyed by chunk.shape)
            # -----------------------------------------------------------------
            if self.use_vae_graph and torch.cuda.is_available():
                shape_key = chunk.shape  # (N, C, H, W)

                # ➊ look up – or build – a graph for this shape
                graph_entry = getattr(self, "_graph_cache", {}).get(shape_key)
                if graph_entry is None:
                    # create cache dict on first use
                    if not hasattr(self, "_graph_cache"):
                        self._graph_cache = {}

                    # warm-up: eager forward once (ensures weights on GPU)
                    _ = self.vae.decode(chunk).sample

                    static_in = torch.empty_like(
                        chunk, memory_format=torch.channels_last
                    )
                    static_out = torch.empty_like(chunk)  # same shape / dtype
                    g = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()

                    with torch.cuda.graph(g):  # capture WITHOUT autocast
                        tmp = self.vae.decode(static_in).sample
                        static_out.copy_(tmp)

                    self._graph_cache[shape_key] = (g, static_in, static_out)
                    graph_entry = (g, static_in, static_out)

                # ➋ replay
                g, static_in, static_out = graph_entry
                static_in.copy_(chunk)
                g.replay()
                out = static_out

            else:  # ------------------- CPU / no-CUDA-graph path -------------------
                with torch.no_grad():
                    # print(f"[decode] Using eager VAE decode for shape {chunk.shape}")
                    out = self.vae.decode(chunk).sample

            decoded_chunks.append(out)

        # -------------- concat + post-process --------------
        decoded = torch.cat(decoded_chunks, dim=0)  # (B·T, 3, H, W)
        #decoded = (decoded * 255).clamp(0, 255).to(torch.uint8)
        decoded = decoded.clamp(-1.0, 1.0).to(torch.float32)
        decoded = rearrange(decoded, "(b t) c h w -> b c t h w", b=b)
        decoded = decoded.permute(0, 1, 2, 4, 3).contiguous()
        return decoded

    # -----------------------------------------------------------
    def _stop_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Stop when last latent frame is near-uniform (low variance → blank/padding)."""
        return x[:, :, -1].float().var(dim=(1, 2, 3)) < 0.05

    # -----------------------------------------------------------
    def generate(self, prompt_emb: torch.Tensor, max_blocks: int = 16):
        if self._is_first_run:
            time_start = time.perf_counter()
            print(f"[generate] Starting first run, with compile...")
        B = prompt_emb.shape[0]
        T = self.block_size
        init = scale_latents(
            self.zero_latent.unsqueeze(0).permute(0, 2, 1, 3, 4).repeat(B, 1, T, 1, 1),
            self.vae_scaling,
        )
        blocks = [init]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        with torch.no_grad(), autocast(dtype=self.dtype):
            for _ in range(max_blocks):
                last_block = blocks[-1]  # only need last block for conditioning
                noise = torch.randn(
                    (B, last_block.shape[1], T, last_block.shape[3], last_block.shape[4]),
                    dtype=self.dtype,
                    device=device,
                )
                cond_lat = sample_latents(self.cfg, last_block)

                def rhs(t, y):
                    return self.denoiser(
                        y, t, encoder_hidden_states=prompt_emb, cond_image=cond_lat
                    ).sample

                # ts = torch.tensor([1.0, 0.0], dtype=self.dtype, device=device)
                # new = odeint(
                #     rhs,
                #     noise,
                #     ts,
                #     atol=1e-5,
                #     rtol=1e-5,
                #     adjoint_params=self.denoiser.parameters(),
                # )[-1]
                timesteps = torch.linspace(
                    1.0, 0.0, steps=100 + 1, device=device, dtype=self.dtype
                )
                new_block = odeint(
                    rhs,
                    noise,
                    timesteps,
                    atol=1e-5,
                    rtol=1e-5,
                    method="euler",
                )[-1]
                finished |= self._stop_mask(new_block)
                if finished.all():
                    break  # blank block — don't append
                blocks.append(new_block)
        lat = torch.cat(blocks, dim=2)[:, :, 16:]  # trim warm-up
        if lat.shape[2] == 0:
            # stop-mask fired on the very first generated block, so nothing
            # survived the warm-up trim. A 0-frame latent crashes VAE decode
            # (diffusers can't reshape a 0-element tensor); fall back to a
            # single blank frame instead -- assess_volume_quality() downstream
            # will still flag/retry this as low quality.
            lat = torch.zeros_like(blocks[0][:, :, :1])
        if self._is_first_run:
            global _COMPILE_SECONDS
            time_stop = time.perf_counter()
            dt = time_stop - time_start
            _COMPILE_SECONDS += dt
            print(f"[generate] Total time: {dt:.1f} seconds")
            self._is_first_run = False

        return lat

    # wrapper for user: decode -> trim stop frames
    def generate_and_decode(self, prompt_emb, **kw):
        lat = self.generate(prompt_emb, **kw)
        return self._decode(unscale_latents(lat, self.vae_scaling))


# ---------------------- main entry --------------------------------


def main():
    global device
    rank, world_size, local_rank, is_ddp = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Per-rank isolated compile cache — avoids concurrent write races
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/opt/app/cache_v3/torch_compile_rank{local_rank}"

    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    if is_ddp:
        dist.barrier()

    cfg = OmegaConf.load(CONFIG_PATH)

    denoiser = (
        instantiate_class_from_config(cfg.denoiser)
        .from_pretrained(DENOISER_CKPT_PATH)
        .to(device, dtype=torch.get_default_dtype())
        .eval()
    )
    denoiser = safe_compile(denoiser)

    cfg.vae.pretrained = f"{_APP}/models/FLUX_vae_checkpoint"
    vae = instantiate(cfg.vae).to(device, dtype=torch.get_default_dtype()).eval()
    if rank == 0:
        print(f"[vae] loading fine-tuned weights from {VAE_FT_CKPT_PATH}")
    vae.load_state_dict(torch.load(VAE_FT_CKPT_PATH, map_location="cpu"), strict=True)
    vae = vae.to(device, dtype=torch.get_default_dtype()).eval()
    vae_scaling = {
        "mean": torch.tensor(_VAE_FT_MEAN).to(device),
        "std":  torch.tensor(_VAE_FT_SCALE).to(device),
    }
    gen = LatentAutoregressiveGenerator(denoiser, vae, vae_scaling, cfg)

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    text_enc = CTCLIPTextEncoder().eval()
    text_sd = torch.load(TEXT_MODEL_PATH, map_location="cpu")
    text_enc.text_transformer.load_state_dict(
        {
            k.replace("text_transformer.", ""): v
            for k, v in text_sd.items()
            if k.startswith("text_transformer.")
        },
        strict=False,
    )

    all_prompts = json.load(open(INPUT_PATH))
    total = len(all_prompts)
    names = [p["input_image_name"] for p in all_prompts]
    reports = [p["report"] for p in all_prompts]

    # distribute samples across ranks
    import numpy as np
    my_idx = np.arange(total)[rank::world_size].tolist()
    if rank == 0:
        print(f"[multi-gpu] world_size={world_size} | total={total} | per-rank={len(my_idx)}", flush=True)

    pool = ThreadPoolExecutor(max_workers=min(16, BATCH_SIZE))

    for start in tqdm(range(0, len(my_idx), BATCH_SIZE), desc=f"Rank {rank} generating", disable=(rank != 0)):
        batch_idx = my_idx[start : start + BATCH_SIZE]
        rep_batch = [reports[k] for k in batch_idx]
        name_batch = [names[k] for k in batch_idx]
        embeds = prompts_to_embeddings(rep_batch, text_enc, tokenizer)
        vols = gen.generate_and_decode(embeds, max_blocks=17)  # [B,3,T,H,W]
        # per-sample list: a retry can produce a different frame count (T) than
        # the rest of the batch, which a batched tensor can't hold
        vol_list = list(vols.unbind(0))
        del vols

        # retry any sample whose blank-trimmed volume came out too short
        for j, n in enumerate(name_batch):
            trimmed = trim_blank_tail(vol_list[j][0])
            retries = 0
            while not assess_volume_quality(trimmed) and retries < QUALITY_MAX_RETRIES:
                retries += 1
                if rank == 0:
                    print(
                        f"[quality] {n}: only {trimmed.shape[0]} slices (<{QUALITY_MIN_FRAMES}), "
                        f"regenerating (attempt {retries}/{QUALITY_MAX_RETRIES})",
                        flush=True,
                    )
                redo = gen.generate_and_decode(embeds[j : j + 1], max_blocks=17)
                vol_list[j] = redo[0]
                trimmed = trim_blank_tail(vol_list[j][0])
            if retries and not assess_volume_quality(trimmed) and rank == 0:
                print(f"[quality] {n}: still only {trimmed.shape[0]} slices after retries, keeping it anyway", flush=True)
        del embeds

        for j, n in enumerate(name_batch):
            arr = vol_list[j][0]  # take first channel, (T,H,W) in [-1,1]
            arr = trim_blank_tail(arr)
            arr = to_hu(arr).clamp(-1000.0, 1000.0)  # correct slope from to_hu, then crop to [-1000,1000]
            if rank == 0 and j == 0:
                print(f"[debug] vol shape={arr.shape} min={arr.min():.1f} max={arr.max():.1f}", flush=True)
            name_base = os.path.splitext(n)[0]  # strip any extension (.mha etc)
            path = os.path.join(OUTPUT_DIR, f"{name_base}.nii.gz")
            pool.submit(save_as_niigz, arr, path)
        del vol_list
        torch.cuda.empty_cache()

    pool.shutdown(wait=True)
    if is_ddp:
        dist.barrier()
    if rank == 0:
        print(f"Results saved to {OUTPUT_DIR}")
        print(f"🔧  total torch.compile time: {_COMPILE_SECONDS:.1f} seconds")
    ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
