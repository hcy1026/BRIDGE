import os
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Union

from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from PIL import PngImagePlugin
from torch.utils.data import DataLoader

import open_clip
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, broadcast_object_list
from accelerate.logging import get_logger
from diffusers.training_utils import free_memory, set_seed
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from main.data import load_image_dataset, ensure_image_id
from main.cache import (
    list_embedding_parquets,
    canonical_is_complete,
    delete_files,
    canonicalize_rank_shards,
    RollingParquetEmbeddingWriter,
)


logger = get_logger(__name__)

# Prevent PIL text-chunk bomb
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


# =========================================================
# Args
# =========================================================
def parse_args() -> SimpleNamespace:
    p = argparse.ArgumentParser("Build embedding cache (parquet shards)")
    p.add_argument("--config_file", type=str, required=True)
    cli = p.parse_args()

    cfg = OmegaConf.load(cli.config_file)
    return SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))


class VAEWrapper(torch.nn.Module):
    """
    Multi-scale VAE wrapper.

    Output = concat([
        gap(down_block_0),
        gap(down_block_1),
        ...,
        gap(mid_block),
        flatten(latent_mean),
    ])

    Notes:
      - uses latent_dist.mean instead of sample() so cached embeddings are deterministic
      - intermediate feature maps are GAP'ed before concatenation
      - final latent keeps spatial structure via flatten, matching the intuition in your HVF figure
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        include_down_blocks: bool = True,
        include_mid_block: bool = True,
        include_latent: bool = True,
    ):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
        self.vae.eval()
        self.include_down_blocks = include_down_blocks
        self.include_mid_block = include_mid_block
        self.include_latent = include_latent

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, output_size=1), 1)

    def forward(self, pixel_values: torch.Tensor):
        h = self.vae.encoder.conv_in(pixel_values)
        feats = []

        for down_block in self.vae.encoder.down_blocks:
            h = down_block(h)
            if self.include_down_blocks:
                feats.append(self._gap(h))

        h = self.vae.encoder.mid_block(h)
        if self.include_mid_block:
            feats.append(self._gap(h))

        h = self.vae.encoder.conv_norm_out(h)
        h = self.vae.encoder.conv_act(h)
        h = self.vae.encoder.conv_out(h)

        moments = self.vae.quant_conv(h)
        mean, _ = torch.chunk(moments, 2, dim=1)
        latents = mean * self.vae.config.scaling_factor
        if self.include_latent:
            feats.append(latents.reshape(latents.shape[0], -1))

        if not feats:
            raise ValueError('At least one VAE feature source must be enabled.')

        emb = torch.cat(feats, dim=-1)
        return (emb,)


class OpenCLIPWrapper(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.visual: torch.nn.Module = clip_model.visual
        self.visual.eval()

    def forward(self, pixel_values: torch.Tensor):
        emb = self.visual(pixel_values)
        return (emb,)


class OpenCLIPRN50MultiScaleWrapper(torch.nn.Module):
    """
    Multi-scale wrapper for OpenCLIP RN50/ModifiedResNet visual backbones.

    It keeps the original final CLIP image embedding, and concatenates it with
    pooled intermediate stage features from layer1/layer2/layer3/layer4.

    Output = concat([
        gap(layer1),
        gap(layer2),
        gap(layer3),
        gap(layer4),
        final_clip_embedding,
    ])

    Shapes for RN50 are typically:
      - gap(layer1): [B, 256]
      - gap(layer2): [B, 512]
      - gap(layer3): [B, 1024]
      - gap(layer4): [B, 2048]
      - final_clip_embedding (attnpool/proj): [B, output_dim]  # often 1024
    so the concatenated embedding dim is often 256 + 512 + 1024 + 2048 + 1024 = 4864.

    This wrapper assumes the visual tower is OpenCLIP's ModifiedResNet-like implementation.
    """

    def __init__(
        self,
        clip_model: torch.nn.Module,
        *,
        use_stage1: bool = True,
        use_stage2: bool = True,
        use_stage3: bool = True,
        use_stage4: bool = True,
        include_final_embedding: bool = True,
    ):
        super().__init__()
        self.visual: torch.nn.Module = clip_model.visual
        self.visual.eval()
        self.use_stage1 = use_stage1
        self.use_stage2 = use_stage2
        self.use_stage3 = use_stage3
        self.use_stage4 = use_stage4
        self.include_final_embedding = include_final_embedding

        required = [
            "conv1", "bn1", "conv2", "bn2", "conv3", "bn3",
            "avgpool", "layer1", "layer2", "layer3", "layer4",
        ]
        missing = [name for name in required if not hasattr(self.visual, name)]
        if missing:
            raise AttributeError(
                "RN50 multi-scale wrapper requires an OpenCLIP ModifiedResNet-like "
                f"visual backbone, but these attributes are missing: {missing}"
            )

        if not hasattr(self.visual, "attnpool"):
            raise AttributeError(
                "RN50 multi-scale wrapper expects visual.attnpool for final CLIP embedding."
            )

    def _act(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """
        Handle minor naming differences across CLIP/open_clip variants.
        Prefer act{idx}; fall back to relu{idx}; then relu.
        """
        for cand in (f"act{idx}", f"relu{idx}", "relu"):
            mod = getattr(self.visual, cand, None)
            if mod is not None:
                return mod(x)
        return torch.relu(x)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        x = self.visual.conv1(x)
        x = self.visual.bn1(x)
        x = self._act(1, x)

        x = self.visual.conv2(x)
        x = self.visual.bn2(x)
        x = self._act(2, x)

        x = self.visual.conv3(x)
        x = self.visual.bn3(x)
        x = self._act(3, x)

        x = self.visual.avgpool(x)
        return x

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, output_size=1), 1)

    def forward(self, pixel_values: torch.Tensor):
        x = self._stem(pixel_values)

        x1 = self.visual.layer1(x)
        x2 = self.visual.layer2(x1)
        x3 = self.visual.layer3(x2)
        x4 = self.visual.layer4(x3)

        feats = []
        if self.use_stage1:
            feats.append(self._gap(x1))
        if self.use_stage2:
            feats.append(self._gap(x2))
        if self.use_stage3:
            feats.append(self._gap(x3))
        if self.use_stage4:
            feats.append(self._gap(x4))

        if self.include_final_embedding:
            final_emb = self.visual.attnpool(x4)
            feats.append(final_emb)

        if not feats:
            raise ValueError("At least one feature source must be enabled for RN50 multi-scale output.")

        emb = torch.cat(feats, dim=-1)
        return (emb,)


class CLIPViTMultiScaleWrapper(torch.nn.Module):
    """
    Multi-layer CLIP ViT wrapper.

    Output = concat([
        pooled(hidden_states[idx_1]),
        pooled(hidden_states[idx_2]),
        ...,
        final_image_embeds,
    ])

    Default for ViT-B/32:
      hidden_state_indices = (2, 5, 8, 11)
      token_pool = 'cls'
    intermediate dimension 768, final dimentsion 512
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        hidden_state_indices=(2, 5, 8, 11),
        token_pool: str = 'cls',
        include_final_embedding: bool = True,
    ):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path)
        self.model.eval()
        self.hidden_state_indices = tuple(hidden_state_indices)
        self.token_pool = token_pool
        self.include_final_embedding = include_final_embedding

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_pool == 'cls':
            return x[:, 0]
        if self.token_pool == 'mean':
            return x[:, 1:].mean(dim=1)
        if self.token_pool == 'cls_mean':
            cls = x[:, 0]
            mean = x[:, 1:].mean(dim=1)
            return torch.cat([cls, mean], dim=-1)
        raise ValueError(f'Unsupported token_pool={self.token_pool}')

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        feats = []
        hidden_states = out.hidden_states
        for idx in self.hidden_state_indices:
            if idx >= len(hidden_states):
                raise ValueError(
                    f'Requested hidden state index {idx}, but model only returned {len(hidden_states)} states.'
                )
            feats.append(self._pool_tokens(hidden_states[idx]))

        if self.include_final_embedding:
            feats.append(out.image_embeds)

        emb = torch.cat(feats, dim=-1)
        return (emb,)


@dataclass
class EncoderBundle:
    encoder: torch.nn.Module
    preprocess: Union[Callable, CLIPImageProcessor, BitImageProcessor]
    output_key: int  # kept for backward-compat; but we also try .image_embeds/.pooler_output first


def build_encoder_bundle(
    *,
    pretrained_model_name_or_path: str,
    resolution: int = 224,
    cache_dir: str = ".cache",
) -> EncoderBundle:
    output_key = 0

    if "vae" in pretrained_model_name_or_path:
        encoder = VAEWrapper(pretrained_model_name_or_path)
        preprocess = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return EncoderBundle(
            encoder=encoder, preprocess=preprocess, output_key=output_key
        )

    # open_clip local checkpoint path (RN50/RN101 style)
    if os.path.isfile(pretrained_model_name_or_path) and any(
        k in pretrained_model_name_or_path for k in ("RN50", "RN101")
    ):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            Path(pretrained_model_name_or_path).stem,
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            weights_only=False,
        )

        model_stem = Path(pretrained_model_name_or_path).stem.upper()
        if "RN50" in model_stem:
            encoder = OpenCLIPRN50MultiScaleWrapper(clip_model)
        else:
            encoder = OpenCLIPWrapper(clip_model)

        return EncoderBundle(
            encoder=encoder, preprocess=preprocess, output_key=output_key
        )

    # default: HF CLIP vision -> multi-layer concat wrapper
    encoder = CLIPViTMultiScaleWrapper(pretrained_model_name_or_path)
    preprocess = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)
    return EncoderBundle(encoder=encoder, preprocess=preprocess, output_key=output_key)


def make_collate_fn(
    preprocess: Union[Callable, CLIPImageProcessor, BitImageProcessor],
) -> Callable[[List[dict]], Dict[str, object]]:
    """
    Input examples from HF dataset:
      - image: PIL.Image
      - image_id: str
    Output:
      - model_inputs: torch.Tensor [B,C,H,W]
      - image_ids: List[str]
    """

    def collate(examples: List[dict]) -> Dict[str, object]:
        image_ids = [ex["image_id"] for ex in examples]
        images = [ex["image"].convert("RGB") for ex in examples]

        if isinstance(preprocess, (CLIPImageProcessor, BitImageProcessor)):
            out = preprocess(images=images, return_tensors="pt")
            model_inputs = out["pixel_values"]
        else:
            model_inputs = torch.stack([preprocess(img) for img in images], dim=0)

        return {"model_inputs": model_inputs.contiguous(), "image_ids": image_ids}

    return collate


def extract_embeddings(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    *,
    output_key: int,
) -> torch.Tensor:
    """
    Robustly read embedding from different model output types.
    """
    out = encoder(x)

    # HF models often return ModelOutput with attributes
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output

    # tuple-like fallback
    if isinstance(out, (tuple, list)):
        return out[output_key]

    # last fallback: allow tensor
    if torch.is_tensor(out):
        return out

    raise TypeError(f"Unsupported encoder output type: {type(out)}")


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(project_config=accelerator_project_config)

    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # base name for parquet files
    model_key = Path(args.pretrained_model_name_or_path).stem
    base_name = f"{args.dataset_name}_{args.split}_{model_key}"

    # -----------------------------
    # Early skip / cleanup (main proc)
    # -----------------------------
    action = {"mode": "run"}  # run | skip
    if accelerator.is_main_process:
        canon, rankish = list_embedding_parquets(out_dir, base_name)

        if canonical_is_complete(canon):
            logger.info(
                f"[cache] Found complete canonical shards for {base_name}. Skip."
            )
            action["mode"] = "skip"
            if rankish:
                logger.warning(
                    f"[cache] Removing {len(rankish)} leftover rank shards (already complete)."
                )
                delete_files(rankish)
        else:
            # remove interrupted leftovers
            if rankish:
                logger.warning(
                    f"[cache] Found {len(rankish)} rank shards but no complete canonical set. "
                    f"Assume interrupted run; deleting and restarting."
                )
                delete_files(rankish)
            if canon:
                logger.warning(
                    f"[cache] Found {len(canon)} canonical shards but incomplete/corrupted. "
                    f"Deleting and restarting."
                )
                delete_files(canon)

    # broadcast decision
    obj = [action]
    broadcast_object_list(obj, from_process=0)
    action = obj[0]

    if action["mode"] == "skip":
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    # -----------------------------
    # 1) Load HF dataset (image + image_id)
    # -----------------------------
    ds = load_image_dataset(
        dataset_name=args.dataset_name,
        image_directory=args.image_directory,
        split=args.split,
        cache_dir=getattr(args, "cache_dir", ".cache"),
    )
    ds = ensure_image_id(ds)

    # shard per rank
    world = accelerator.num_processes
    rank = accelerator.process_index
    ds = ds.shard(num_shards=world, index=rank, contiguous=True)

    # -----------------------------
    # 2) Build encoder + preprocess (now local in this script)
    # -----------------------------
    bundle = build_encoder_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        resolution=int(args.resolution),
        cache_dir=args.cache_dir,
    )
    encoder = bundle.encoder
    preprocess = bundle.preprocess
    output_key = bundle.output_key

    # dtype selection
    weight_dtype = torch.float32
    mp = accelerator.mixed_precision
    if mp == "fp16" and ("vae" not in args.pretrained_model_name_or_path):
        weight_dtype = torch.float16
    elif mp == "bf16" and ("vae" not in args.pretrained_model_name_or_path):
        weight_dtype = torch.bfloat16

    encoder.to(accelerator.device, dtype=weight_dtype)
    encoder.eval()

    # -----------------------------
    # 3) DataLoader + collate
    # -----------------------------
    collate_fn = make_collate_fn(preprocess)
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # -----------------------------
    # 4) Infer embedding dim (first batch)
    # -----------------------------
    first = next(iter(dl))
    with torch.inference_mode():
        x = first["model_inputs"].to(accelerator.device, dtype=weight_dtype)
        y = extract_embeddings(encoder, x, output_key=output_key)
        y = y.float().detach().cpu()
    emb_dim = int(y.shape[-1])
    del first, x, y
    free_memory()

    # -----------------------------
    # 5) Streaming parquet writer (rank-style)
    # -----------------------------
    writer = RollingParquetEmbeddingWriter(
        out_dir=str(out_dir),
        base_name=base_name,
        rank=rank,
        world=world,
        dim_map={"emb": emb_dim},
        dtype=getattr(args, "dtype", "float16"),
        compression="zstd",
        max_rows_per_file=getattr(args, "max_rows_per_file", 200_000),
    )

    # -----------------------------
    # 6) Loop + flush
    # -----------------------------
    rows_per_flush = int(getattr(args, "rows_per_flush", 4096))
    buf_ids: List[str] = []
    buf_emb: List[torch.Tensor] = []

    pbar = tqdm(
        dl, desc="building embeddings", disable=not accelerator.is_local_main_process
    )
    for batch in pbar:
        model_inputs = batch["model_inputs"].to(accelerator.device)
        if model_inputs.dtype != weight_dtype:
            model_inputs = model_inputs.to(weight_dtype)

        image_ids = batch["image_ids"]

        with torch.inference_mode():
            image_embeds = extract_embeddings(
                encoder, model_inputs, output_key=output_key
            )
            image_embeds = image_embeds.float().detach().cpu()

        buf_ids.extend(image_ids)
        buf_emb.append(image_embeds)

        if len(buf_ids) >= rows_per_flush:
            embs = torch.cat(buf_emb, dim=0)
            writer.write({"image_id": buf_ids, "emb": embs})
            logger.info(f"Rank {rank}: flushed {len(buf_ids)} rows")
            buf_ids.clear()
            buf_emb.clear()
            free_memory()

    if buf_ids:
        embs = torch.cat(buf_emb, dim=0)
        writer.write({"image_id": buf_ids, "emb": embs})
        logger.info(f"Rank {rank}: final flush {len(buf_ids)} rows")

    writer.close()
    accelerator.wait_for_everyone()

    # -----------------------------
    # 7) Canonicalize shards on main proc
    # -----------------------------
    if accelerator.is_main_process:
        finals = canonicalize_rank_shards(
            out_dir, base_name, delete_existing_canon=False
        )
        logger.info(
            f"[cache] canonical shards = {len(finals)} for base_name={base_name}"
        )

    encoder.cpu()
    del encoder
    free_memory()
    accelerator.end_training()


if __name__ == "__main__":
    main()
