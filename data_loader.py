"""
Image-Pair / Single-Panel dataset for FLUX-Kontext
===========================================================

* Works in **two** modes:

    • **paired**  (source | target  side-by-side,  mask = black-left / white-right)
    • **single**  (target only,     mask matches dataset mask column or all-ones)

* Mode is chosen automatically:

    ─ if --panel_mode is given    -> honour it  
    ─ else if --source_image_column exists in the dataset -> “paired”  
    ─ else                  -> “single”

* Output dictionary (independent of mode):

    {
        "pixel_values"      : tensor  (3,H,W  or  3,H,2W),
        "mask_pixel_values" : tensor  (1,H,W  or  1,H,2W),
        "prompts"           : str,
        "bucket_idx"        : int,
    }


"""


from __future__ import annotations

import random
import logging
from typing import Optional, List, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import (
    crop,
    hflip,
    gaussian_blur,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    rgb_to_grayscale,
)
from PIL import Image, ImageOps
from PIL.ImageOps import exif_transpose

from diffusers.training_utils import parse_buckets_string, find_nearest_bucket

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _norm_to_tensor():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

_to_tensor_keep01 = transforms.ToTensor()


def _mask_tensor(h: int, w: int, two_panel: bool, device="cpu"):
    if two_panel:
        return torch.cat(
            [torch.zeros(1, h, w, device=device), torch.ones(1, h, w, device=device)], dim=2
        )
    return torch.ones(1, h, w, device=device)

# -----------------------------------------------------------------------------
# main dataset
# -----------------------------------------------------------------------------

class PairedImageDataset(Dataset):
    def __init__(self, args, split: str = "train"):
        super().__init__()
        self.args, self.split = args, split
        self.is_train = split == "train"  # toggle augments

        # ----------------------------- HF load --------------------------------
        from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

        ds_obj = None
        name = getattr(args, "dataset_name", None)

        if name is None:
            raise ValueError("args.dataset_name must be provided")

        # If dataset_name points to an existing folder, treat it as a local HF dataset
        # built with datasets.save_to_disk(...)
        if Path(name).exists() and Path(name).is_dir():
            ds_obj = load_from_disk(name)
            if isinstance(ds_obj, DatasetDict):
                if split not in ds_obj:
                    raise ValueError(
                        f"Local dataset at {name} has splits {list(ds_obj.keys())}, "
                        f"but you requested split='{split}'."
                    )
                self.ds = ds_obj[split]
            else:
                # A single-split Dataset saved to disk (no split names).
                # We ignore the requested `split` and use it directly.
                self.ds = ds_obj
                if split not in ("train", "validation", "test"):
                    # just to avoid confusion in logs
                    logging.info(f"[PairedImageDataset] Local dataset has no splits; using it as-is (requested split='{split}')")
        else:
            # Remote or scripted dataset on the Hub
            self.ds = load_dataset(
                name,
                getattr(args, "dataset_config_name", None),
                cache_dir=getattr(args, "cache_dir", None),
                split=split,
            )


        cols = self.ds.column_names
        have_src = args.source_image_column in cols

        self.mode = (
            args.panel_mode if args.panel_mode is not None else ("paired" if have_src else "single")
        )
        if self.mode == "paired" and not have_src:
            raise ValueError("panel_mode='paired' but dataset has no source column")

        self.tgt_col = args.target_image_column
        self.src_col = args.source_image_column if have_src else None
        self.mask_col = args.mask_column if args.mask_column in cols else None
        self.caption_col = args.caption_column if args.caption_column in cols else None
        self.use_caption = self.caption_col is not None

        # prompt ------------------------------------------------
        self.template = getattr(args, "caption_template", None)
        if self.template and not self.use_caption:
            raise ValueError("--caption_template needs --caption_column present in dataset")

        if not (self.use_caption or self.template) and args.instance_prompt is None:
            raise ValueError("Need caption_column, caption_template or instance_prompt")

        # buckets --------------------------------------------------------------
        self.buckets = (
            parse_buckets_string(args.aspect_ratio_buckets)
            if args.aspect_ratio_buckets
            else [(1024, 768)]
        )
        logging.info(f"[PairedImageDataset] Buckets: {self.buckets}")

        self.norm = _norm_to_tensor()

        # train‑only photometric aug setup ------------------------------------
        if self.is_train:
            self.cj_vals = list(map(float, args.color_jitter.split(","))) if args.color_jitter else None
            self.grayscale_prob = args.random_grayscale
            self.blur_sigma_max = args.gaussian_blur
            self.do_flip = args.random_flip
        else:
            self.cj_vals = None
            self.grayscale_prob = 0.0
            self.blur_sigma_max = 0.0
            self.do_flip = False

    def __len__(self):
        return len(self.ds) * self.args.repeats

    def __getitem__(self, gidx):
        row = self.ds[gidx % len(self.ds)]

        pil_tgt = exif_transpose(row[self.tgt_col].convert("RGB"))
        pil_src = (
            exif_transpose(row[self.src_col].convert("RGB")) if self.mode == "paired" else None
        )
        pil_msk = None
        if self.mask_col:
            pil_msk = exif_transpose(row[self.mask_col].convert("L"))
            if self.args.invert_mask:
                pil_msk = ImageOps.invert(pil_msk)

        # ------------ spatial bucket / crop ----------------------------
        h, w = pil_tgt.height, pil_tgt.width
        bidx, (bh, bw) = self._pick_bucket(h, w)
        top, left, ch, cw = self._compute_crop(h, w, bh, bw)

        # photometric decision flags ------------------------------------
        flip = self.do_flip and random.random() < 0.5
        cj = self._sample_color_jitter()
        gray = self.grayscale_prob and random.random() < self.grayscale_prob
        sigma = random.uniform(0.1, self.blur_sigma_max) if self.blur_sigma_max else None

        # --------- apply transforms to images --------------------------
        tgt = self._proc_rgb(pil_tgt, top, left, ch, cw, bw, bh, flip, cj, gray, sigma)
        if self.mode == "paired":
            src = self._proc_rgb(pil_src, top, left, ch, cw, bw, bh, flip, cj, False, None)
            pix = torch.cat([src, tgt], dim=2)
        else:
            pix = tgt

        # --------------------- mask ------------------------------------
        if pil_msk is not None:
            m = self._proc_mask(pil_msk, top, left, ch, cw, bw, bh, flip)
            mask = torch.cat([torch.zeros_like(m), m], dim=2) if self.mode == "paired" else m
        else:
            mask = _mask_tensor(bh, bw, self.mode == "paired")

        # ------------------- prompt selection --------------------------
        if self.template:
            prompt = self.template.format(cap=row[self.caption_col])
        elif self.use_caption:
            prompt = row[self.caption_col]
        else:
            prompt = self.args.instance_prompt

        return {
            "pixel_values": pix,
            "mask_pixel_values": mask,
            "prompts": prompt,
            "bucket_idx": bidx,
        }

    # ------------------------------------------------------------------ helpers
    def _pick_bucket(self, h: int, w: int):
        idx = find_nearest_bucket(h, w, self.buckets)
        return idx, self.buckets[idx]

    def _compute_crop(self, h: int, w: int, bh: int, bw: int):
        if not (self.args.random_crop or self.args.center_crop):
            return 0, 0, h, w
        raw_ar, buck_ar = h / w, bh / bw
        if abs(raw_ar - buck_ar) < 1e-6:
            ch, cw = h, w
        elif raw_ar > buck_ar:
            cw = w
            ch = int(round(buck_ar * cw))
        else:
            ch = h
            cw = int(round(ch / buck_ar))
        if self.args.center_crop:
            top = (h - ch) // 2
            left = (w - cw) // 2
        else:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
        return top, left, ch, cw

    # ---------------- photometric helpers -----------------------------
    def _sample_color_jitter(self):
        if not self.cj_vals:
            return None
        b, c, s, h = self.cj_vals
        jitter = lambda m: 1 + random.uniform(-m, m)
        return {
            "brightness": jitter(b),
            "contrast": jitter(c),
            "saturation": jitter(s),
            "hue": random.uniform(-h, h),
        }

    def _proc_rgb(
        self,
        img: Image.Image,
        top: int,
        left: int,
        ch: int,
        cw: int,
        bw: int,
        bh: int,
        flip: bool,
        cj: Optional[dict],
        gray: bool,
        sigma: Optional[float],
    ) -> torch.Tensor:
        if self.args.center_crop or self.args.random_crop:
            img = crop(img, top, left, ch, cw)
        if flip:
            img = hflip(img)
        if img.size != (bw, bh):
            img = img.resize((bw, bh), Image.Resampling.BILINEAR)
        if cj is not None:
            img = adjust_brightness(img, cj["brightness"])
            img = adjust_contrast(img, cj["contrast"])
            img = adjust_saturation(img, cj["saturation"])
            img = adjust_hue(img, cj["hue"])
        if gray:
            img = rgb_to_grayscale(img, num_output_channels=3)
        if sigma is not None:
            k = int(sigma * 4) | 1
            img = gaussian_blur(img, [k, k], sigma=sigma)
        return self.norm(img)

    def _proc_mask(
        self,
        mask: Image.Image,
        top: int,
        left: int,
        ch: int,
        cw: int,
        bw: int,
        bh: int,
        flip: bool,
    ) -> torch.Tensor:
        if self.args.center_crop or self.args.random_crop:
            mask = crop(mask, top, left, ch, cw)
        if flip:
            mask = hflip(mask)
        if mask.size != (bw, bh):
            mask = mask.resize((bw, bh), Image.Resampling.NEAREST)
        return (_to_tensor_keep01(mask) > 0.5).float()

# ---------------------- collate ---------------------------------------------

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_pixel_values": torch.stack([b["mask_pixel_values"] for b in batch]),
        "prompts": [b["prompts"] for b in batch],
    }



# ----------------------------------------------------------------------------
#  Quick CLI sanity‑check  (optional)
# ----------------------------------------------------------------------------
# Run this file directly (`python data_loader.py`) to load a small batch, dump a
# few processed tensors to PNG, and print the resulting prompts.  Handy for
# visual smoke‑testing bucket/crop/augment logic without launching a full
# training script.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from torchvision.utils import save_image

    # ------------------------- toy CLI args ------------------------------
    def get_args():
        p = argparse.ArgumentParser()
        # ––– dataset
        p.add_argument("--dataset_name",          default="raresense/Viton_validation")
        p.add_argument("--dataset_config_name",   default=None)
        p.add_argument("--cache_dir",             default=None)
        # ––– columns
        p.add_argument("--source_image_column",   default="source")
        p.add_argument("--target_image_column",   default="target")
        p.add_argument("--caption_column",        default="ai_name")
        p.add_argument("--mask_column",           default="mask")
        # ––– mode override (optional)
        p.add_argument("--panel_mode",            choices=["paired", "single"], default="paired")
        # ––– augs / buckets
        p.add_argument("--aspect_ratio_buckets",  default="1184,880")
        p.add_argument("--random_flip",           action="store_true", default=True)
        p.add_argument("--random_crop",           action="store_true", default=False)
        p.add_argument("--center_crop",           action="store_true", default=True)
        p.add_argument("--repeats",               type=int,  default=1)
        p.add_argument("--color_jitter",          default="0.2,0.2,0.2,0.05")
        p.add_argument("--random_grayscale",      type=float, default=0.3)
        p.add_argument("--gaussian_blur",         type=float, default=0.8)
        # ––– text
        p.add_argument("--instance_prompt",       default="flux-kontext")
        p.add_argument("--caption_template",      default="maow maow {cap} maow")
        p.add_argument("--invert_mask",           action="store_true", default=False)
        return p.parse_args()

    args = get_args()

    # ------------------------- dataset + loader --------------------------
    ds = PairedImageDataset(args, split="train")
    print(f"Dataset len    : {len(ds)} (mode: {ds.mode})")

    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))

    print("pixel_values      :", batch["pixel_values"].shape)
    print("mask_pixel_values :", batch["mask_pixel_values"].shape)
    print("prompts           :", batch["prompts"])

    # ------------------------- dump images --------------------------------
    outdir = Path("debug_samples")
    outdir.mkdir(exist_ok=True)

    denorm = lambda x: (x * 0.5 + 0.5).clamp(0, 1)

    for i_idx, (i_ds, i_dl) in enumerate(zip(range(len(batch["prompts"])), range(batch["pixel_values"].shape[0]))):
        # original PILs ---------------------------------------------------
        raw_row = ds.ds[i_ds]  # HuggingFace Datasets row (un‑augmented)
        if ds.mode == "paired":
            raw_row[args.source_image_column].save(outdir / f"sample_{i_idx}_src_orig.png")
        raw_row[args.target_image_column].save(outdir / f"sample_{i_idx}_tgt_orig.png")
        if args.mask_column in raw_row:
            raw_row[args.mask_column].save(outdir / f"sample_{i_idx}_mask_orig.png")

        # processed tensors ---------------------------------------------
        px  = batch["pixel_values"][i_dl]          # 3×H×(W or 2W)
        msk = batch["mask_pixel_values"][i_dl]     # 1×H×(W or 2W)

        save_image(denorm(px), outdir / f"sample_{i_idx}_pixels_proc.png")
        save_image(msk.repeat(3, 1, 1), outdir / f"sample_{i_idx}_mask_proc.png")

        # if paired, split the processed concat for clarity -------------
        if ds.mode == "paired":
            _, H, W2 = px.shape
            W = W2 // 2
            save_image(denorm(px[:, :, :W]), outdir / f"sample_{i_idx}_src_proc.png")
            save_image(denorm(px[:, :, W:]),  outdir / f"sample_{i_idx}_tgt_proc.png")

        print(f"sample_{i_idx}: prompt='{batch['prompts'][i_dl]}'  ➜  images dumped")

    print("All images written to", outdir.resolve())
