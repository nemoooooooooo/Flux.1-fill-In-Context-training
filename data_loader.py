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

from torch.utils.data import Dataset
import itertools, logging, random
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
from PIL import Image as PILImage
from PIL.ImageOps import exif_transpose
from PIL.ImageOps import invert

from diffusers.training_utils import parse_buckets_string, find_nearest_bucket

from torchvision import transforms
from torchvision.transforms.functional import (
    crop, hflip, gaussian_blur, adjust_brightness,
    adjust_contrast, adjust_saturation, adjust_hue, rgb_to_grayscale
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _norm_to_tensor():
    "PIL → [-1,1] tensor"
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def _pick_bucket(h: int, w: int, buckets: List[Tuple[int, int]]):
    idx = find_nearest_bucket(h, w, buckets)
    return idx, buckets[idx]


def _sample_color_jitter(max_vals):
    b, c, s, h = max_vals
    fn = lambda mag: 1.0 + random.uniform(-mag, mag)
    return {
        "brightness": fn(b),
        "contrast":   fn(c),
        "saturation": fn(s),
        "hue":        random.uniform(-h, h),
    }


def _mask_tensor(h: int, w: int, two_panel: bool, device="cpu"):
    """Synthetic mask if dataset provides none."""
    if two_panel:
        left  = torch.zeros(1, h,   w, device=device)
        right = torch.ones( 1, h,   w, device=device)
        return torch.cat([left, right], dim=2)           # 1×H×2W
    else:
        return torch.ones(1, h, w, device=device)        # 1×H×W


_to_tensor_keep01 = transforms.ToTensor()   # keeps values in 0…1


# ---------------------------------------------------------------------------
#  Data Loader
# ---------------------------------------------------------------------------
class PairedImageDataset(Dataset):

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    def __init__(self, args, split: str = "train"):
        """
        Expected `args` attributes
        -------------------------
        # dataset
        dataset_name (str, required)
        dataset_config_name, cache_dir
        source_image_column, target_image_column, caption_column, mask_column
        panel_mode  ("paired" | "single" | None)

        # image / aug params
        resolution, aspect_ratio_buckets, random_flip, random_crop,
        center_crop, repeats, color_jitter, random_grayscale, gaussian_blur

        # text
        instance_prompt
        """
        self.split = split
        self.mask_invert = args.mask_invert

        # -------------------- 0. load dataset -------------------------
        if args.dataset_name is None:
            raise ValueError("--dataset_name is required")

        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        ds = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )[split]

        cols = ds.column_names
        have_src_col = getattr(args, "source_image_column", None) in cols

        # decide mode
        self.mode = getattr(args, "panel_mode", None)
        if self.mode not in (None, "paired", "single"):
            raise ValueError("--panel_mode must be 'paired' or 'single'")
        if self.mode is None:
            self.mode = "paired" if have_src_col else "single"
        if self.mode == "paired" and not have_src_col:
            raise ValueError("panel_mode='paired' but dataset has no source column")

        # -------- required columns ----------
        if args.target_image_column not in cols:
            raise ValueError(f"--target_image_column missing, dataset columns: {cols}")
        self.tgt_imgs = ds[args.target_image_column]

        if self.mode == "paired":
            self.src_imgs = ds[args.source_image_column]
        else:
            self.src_imgs = itertools.repeat(None)  # dummy iterator

        # optional mask column
        self.has_mask = getattr(args, "mask_column", None) in cols
        if self.has_mask:
            self.msk_imgs = ds[args.mask_column]

        # -------- captions ----------
        self.use_caption = args.caption_column in cols if args.caption_column else False
        if self.use_caption:
            self.captions = ds[args.caption_column]
        else:
            if args.instance_prompt is None:
                raise ValueError("Need either --caption_column or --instance_prompt")
            self.instance_prompt = args.instance_prompt

        # ------------- buckets --------------
        self.buckets = (
            parse_buckets_string(args.aspect_ratio_buckets)
            if args.aspect_ratio_buckets else [(args.resolution, args.resolution)]
        )
        logging.info(f"Using aspect-ratio buckets: {self.buckets}")

        # ------------- transforms & flags --------------
        self._norm = _norm_to_tensor()

        self.random_flip   = args.random_flip
        self.random_crop   = args.random_crop
        self.center_crop   = args.center_crop
        self.repeats       = args.repeats

        self.color_jitter_max = (
            list(map(float, args.color_jitter.split(",")))
            if getattr(args, "color_jitter", None) else None
        )
        self.grayscale_prob = getattr(args, "random_grayscale", 0.0)
        self.blur_sigma_max = getattr(args, "gaussian_blur", 0.0)

        if split in ("validation", "test"):
            # freeze augs
            self.repeats           = 1
            self.random_flip       = False
            self.random_crop       = False
            self.color_jitter_max  = None
            self.grayscale_prob    = 0.0
            self.blur_sigma_max    = 0.0

        # ------------- preprocess all --------------
        (
            self.pixel_values,
            self.mask_pixel_values,
            self.bucket_ids,
            self.prompts,
        ) = self._preprocess_all()

    def __len__(self):  return len(self.pixel_values)

    def __getitem__(self, idx):
        return {
            "pixel_values"      : self.pixel_values[idx],
            "mask_pixel_values" : self.mask_pixel_values[idx],
            "prompts"           : self.prompts[idx],
            "bucket_idx"        : self.bucket_ids[idx],
        }

    def _preprocess_all(self):
        px_out, msk_out, bid_out, prm_out = [], [], [], []

        img_iter = zip(
            self.src_imgs,
            self.tgt_imgs,
            (self.msk_imgs if self.has_mask else itertools.repeat(None)),
            (self.captions if self.use_caption else itertools.repeat(None)),
        )

        # --------------- iterate through dataset ---------------------
        for pil_src, pil_tgt, pil_msk, caption in img_iter:
            pil_tgt = exif_transpose(pil_tgt.convert("RGB"))
            if self.mode == "paired":
                pil_src = exif_transpose(pil_src.convert("RGB"))
            if self.has_mask:
                if self.mask_invert:
                    pil_msk = exif_transpose(invert(pil_msk.convert("L")))
                else:
                    pil_msk = exif_transpose(pil_msk.convert("L"))

            w_raw, h_raw = pil_tgt.size
            bucket_idx, (bh, bw) = _pick_bucket(h_raw, w_raw, self.buckets)

            # ----- shared crop rectangle -----
            raw_ar, bucket_ar = h_raw / w_raw, bh / bw
            if abs(raw_ar - bucket_ar) == 0 or (not self.center_crop and not self.random_crop):
                crop_h, crop_w, top, left = h_raw, w_raw, 0, 0
            else:
                if raw_ar > bucket_ar:
                    crop_w = w_raw
                    crop_h = int(round(bucket_ar * crop_w))
                else:
                    crop_h = h_raw
                    crop_w = int(round(crop_h / bucket_ar))
                if self.center_crop:
                    top  = (h_raw - crop_h) // 2
                    left = (w_raw - crop_w) // 2
                else:
                    max_t, max_l = h_raw - crop_h, w_raw - crop_w
                    top  = random.randint(0, max_t) if max_t else 0
                    left = random.randint(0, max_l) if max_l else 0

            # ----- per-sample aug decisions -----
            flip_flag = self.random_flip and random.random() < 0.5
            cj_params = _sample_color_jitter(self.color_jitter_max) if self.color_jitter_max else None
            do_gray   = self.grayscale_prob and random.random() < self.grayscale_prob
            sigma     = random.uniform(0.1, self.blur_sigma_max) if self.blur_sigma_max else None

            # ----- helpers -----
            def _proc_rgb(img: PILImage.Image):
                if self.center_crop or self.random_crop:
                    img = crop(img, top, left, crop_h, crop_w)
                if flip_flag:
                    img = hflip(img)
                if img.size != (bw, bh):
                    img = img.resize((bw, bh), PILImage.Resampling.BILINEAR)
                if cj_params:
                    img = adjust_brightness(img, cj_params["brightness"])
                    img = adjust_contrast(img,   cj_params["contrast"])
                    img = adjust_saturation(img, cj_params["saturation"])
                    img = adjust_hue(img,        cj_params["hue"])
                if do_gray:
                    img = rgb_to_grayscale(img, num_output_channels=3)
                if sigma:
                    k = int(sigma * 4) | 1
                    img = gaussian_blur(img, [k, k], sigma=sigma)
                return self._norm(img)

            def _proc_mask(mask_img: PILImage.Image):
                if self.center_crop or self.random_crop:
                    mask_img = crop(mask_img, top, left, crop_h, crop_w)
                if flip_flag:
                    mask_img = hflip(mask_img)
                if mask_img.size != (bw, bh):
                    mask_img = mask_img.resize((bw, bh), PILImage.Resampling.NEAREST)
                return (_to_tensor_keep01(mask_img) > 0.5).float()   # 1×H×W

            # ----- repeats -----
            for _ in range(self.repeats):
                tgt_tensor = _proc_rgb(pil_tgt)

                if self.mode == "paired":
                    src_tensor  = _proc_rgb(pil_src)
                    pix_tensor = torch.cat([src_tensor, tgt_tensor], dim=2)  # 3×H×2W
                else:
                    pix_tensor = tgt_tensor                                 # 3×H×W

                # -------------- mask handling --------------
                if self.has_mask:
                    m = _proc_mask(pil_msk)             # 1×H×W
                    if self.mode == "paired":
                        black = torch.zeros_like(m)
                        mask_tensor = torch.cat([black, m], dim=2)          # 1×H×2W
                    else:
                        mask_tensor = m                                      # 1×H×W
                else:
                    mask_tensor = _mask_tensor(bh, bw, self.mode == "paired")

                # -------------- push --------------
                px_out.append(pix_tensor)
                msk_out.append(mask_tensor)
                bid_out.append(bucket_idx)
                prm_out.append(caption if caption is not None else self.instance_prompt)

        if not px_out:
            raise ValueError(f"No usable images in split '{self.split}'")

        return px_out, msk_out, bid_out, prm_out


def collate_fn(examples):
    pixel_values      = torch.stack([e["pixel_values"]      for e in examples]).float()
    mask_pixel_values = torch.stack([e["mask_pixel_values"] for e in examples]).float()
    prompts           = [e["prompts"] for e in examples]

    return {"pixel_values":      pixel_values,
            "mask_pixel_values": mask_pixel_values,
            "prompts":           prompts}


# ----------------------------------------------------------------------------
#  Quick CLI test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from torchvision.utils import save_image

    def get_args():
        p = argparse.ArgumentParser()
        # ---------------- dataset ----------------
        p.add_argument("--dataset_name",          default="raresense/Viton_validation")
        p.add_argument("--dataset_config_name",   default=None)
        p.add_argument("--cache_dir",             default=None)
        # ---------------- columns ----------------
        p.add_argument("--source_image_column",   default="source")
        p.add_argument("--target_image_column",   default="target")
        p.add_argument("--caption_column",        default="ai_name")
        p.add_argument("--mask_column",           default="mask")
        # ---------------- mode -------------------
        p.add_argument("--panel_mode",            choices=["paired", "single"], default="paired")
        # ---------------- augs -------------------
        p.add_argument("--resolution", type=int,  default=512)
        p.add_argument("--aspect_ratio_buckets",  default="1184,880")
        p.add_argument("--random_flip",           action="store_true", default=True)
        p.add_argument("--random_crop",           action="store_true", default=False)
        p.add_argument("--center_crop",           action="store_true", default=True)
        p.add_argument("--repeats", type=int,     default=1)
        p.add_argument("--color_jitter",          default="0.2,0.2,0.2,0.05")
        p.add_argument("--random_grayscale", type=float, default=0.3)
        p.add_argument("--gaussian_blur",    type=float, default=0.8)
        # ---------------- text -------------------
        p.add_argument("--instance_prompt", default="flux-kontext")
        p.add_argument("--mask_invert",default=False)
        return p.parse_args()

    args = get_args()

    ds = PairedImageDataset(args, split="train")
    print(f"Dataset size: {len(ds)} | mode: {ds.mode}")

    sample = collate_fn([ds[i] for i in range(min(4, len(ds)))])
    print("pixel_values      :", sample["pixel_values"].shape)
    print("mask_pixel_values :", sample["mask_pixel_values"].shape)
    print("prompts           :", sample["prompts"])

# ------------------------------------------------------------------
#  Visual sanity-check  
# ------------------------------------------------------------------
outdir = Path("debug_samples")
outdir.mkdir(exist_ok=True)

denorm = lambda x: (x * 0.5 + 0.5).clamp(0, 1)

batch_len = sample["pixel_values"].shape[0]
for k in range(batch_len):
    # ---- originals ----------------------------------------------
    if ds.mode == "paired":
        ds.src_imgs[k].save(outdir / f"sample-{k}_src_orig.png")
    ds.tgt_imgs[k].save(outdir / f"sample-{k}_tgt_orig.png")

    # ---- processed tensors --------------------------------------
    px  = sample["pixel_values"][k]          # 3×H×W  or  3×H×2W
    msk = sample["mask_pixel_values"][k]     # 1×H×W  or  1×H×2W

    # save *entire* concatenated tensor (what the model sees)
    save_image(denorm(px), outdir / f"sample-{k}_concat_proc.png")

    if ds.mode == "paired":
        # also split for easier human inspection
        _, H, W2 = px.shape
        W = W2 // 2
        src_proc = denorm(px[:, :, :W])
        tgt_proc = denorm(px[:, :, W:])
        save_image(src_proc, outdir / f"sample-{k}_src_proc.png")
        save_image(tgt_proc, outdir / f"sample-{k}_tgt_proc.png")
    else:
        save_image(denorm(px), outdir / f"sample-{k}_tgt_proc.png")

    # ---- mask ----------------------------------------------------
    save_image(msk.repeat(3, 1, 1), outdir / f"sample-{k}_mask.png")

print("Images written to", outdir.resolve())