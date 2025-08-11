import os
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import shutil
import tempfile

import numpy as np
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from torchvision.io import write_jpeg, write_png

import datasets
from datasets import Image as HFImage
from PIL import Image

# ---------------- GPU kernels ----------------

def gaussian_kernels_1d(sigma: float, device: torch.device, dtype=torch.float32):
    sigma = float(sigma)
    radius = max(1, int(round(3.0 * sigma)))
    size = radius * 2 + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    kx = k.view(1, 1, 1, size).expand(3, 1, 1, size).contiguous()
    ky = k.view(1, 1, size, 1).expand(3, 1, size, 1).contiguous()
    return kx, ky, radius

@torch.no_grad()
def masked_bg_gray_blur_batch_uint8(
    src_u8: torch.Tensor,    # [B,3,H,W] uint8 on device
    m_u8: torch.Tensor,      # [B,1,H,W] uint8 on device (0..255)
    kx, ky, radius: int,
    alpha: float
) -> torch.Tensor:
    """
    out = where(mask<128, src, alpha*gray(blur(src)) + (1-alpha)*blur(src))
    """
    x = src_u8.to(torch.float32) / 255.0
    m = (m_u8.to(torch.float32) < 128.0).to(torch.float32)  # 1 where KEEP (watch), else 0

    xw = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    h  = F.conv2d(xw, kx, groups=3)
    xh = F.pad(h, (0, 0, radius, radius), mode="reflect")
    blurred = F.conv2d(xh, ky, groups=3)

    r, g, b = blurred[:, 0:1], blurred[:, 1:2], blurred[:, 2:3]
    gray3 = (0.299 * r + 0.587 * g + 0.114 * b).repeat(1, 3, 1, 1)
    bg = gray3 * float(alpha) + blurred * (1.0 - float(alpha))

    m3 = m.repeat(1, 3, 1, 1)
    out = m3 * x + (1.0 - m3) * bg
    out = (out.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return out

# -------------- HF -> Torch loader -------------

class HFDSourceMaskDataset(Dataset):
    def __init__(self, ds: datasets.Dataset, source_col: str, mask_col: str):
        self.ds = ds
        self.source_col = source_col
        self.mask_col = mask_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        pil_src = row[self.source_col]   # PIL RGB
        pil_msk = row[self.mask_col]     # PIL L or RGB

        t_src = pil_to_tensor(pil_src)   # uint8 [C,H,W]
        if t_src.ndim == 2:  # grayscale → 3ch
            t_src = t_src.unsqueeze(0).repeat(3, 1, 1)
        if t_src.size(0) == 1:
            t_src = t_src.repeat(3, 1, 1)

        t_msk = pil_to_tensor(pil_msk.convert("L")).squeeze(0)  # uint8 [H,W]
        if t_msk.shape[0] != t_src.shape[1] or t_msk.shape[1] != t_src.shape[2]:
            t_msk = F.interpolate(
                t_msk.unsqueeze(0).unsqueeze(0).to(torch.float32),
                size=(t_src.shape[1], t_src.shape[2]),
                mode="nearest",
            ).squeeze(0).squeeze(0).to(torch.uint8)

        return idx, t_src, t_msk.unsqueeze(0)  # [1,H,W] mask

def pad_collate(batch):
    idxs, imgs, masks = zip(*batch)
    max_h = max(x.shape[1] for x in imgs)
    max_w = max(x.shape[2] for x in imgs)
    p_imgs, p_masks, sizes = [], [], []
    for im, mk in zip(imgs, masks):
        c, h, w = im.shape
        sizes.append((h, w))
        pad_w = max_w - w
        pad_h = max_h - h
        if pad_w or pad_h:
            im = F.pad(im, (0, pad_w, 0, pad_h), mode="replicate")
            mk = F.pad(mk, (0, pad_w, 0, pad_h), mode="replicate")
        p_imgs.append(im); p_masks.append(mk)
    imgs_b = torch.stack(p_imgs, dim=0)   # uint8 [B,3,Hmax,Wmax]
    masks_b= torch.stack(p_masks, dim=0)  # uint8 [B,1,Hmax,Wmax]
    return (
        torch.tensor(idxs, dtype=torch.long),
        imgs_b.contiguous(),
        masks_b.contiguous(),
        torch.tensor(sizes, dtype=torch.int32),
    )

# -------------- Builder -----------------------

def build_level_dataset_fast_masked(
    orig_name: str,
    orig_cfg: str | None,
    level: dict,
    source_col: str,
    mask_col: str,
    cache_root: Path,
    split: str = "train",
    loader_workers: int = 32,
    batch_size: int = 64,
    save_format: str = "jpg",  # 'jpg' or 'png'
    jpeg_quality: int = 92,
    device: str = "cuda",
    writer_threads: int = 8,
) -> Path:
    cache_root = Path(cache_root); cache_root.mkdir(parents=True, exist_ok=True)
    key = {
        "name": level["name"],
        "sigma": float(level["difficulty_blur_sigma"]),
        "alpha": float(level["difficulty_gray_alpha"]),
        "source_col": source_col,
        "mask_col": mask_col,
        "dataset": orig_name,
        "dataset_cfg": orig_cfg or "",
        "split": split,
        "fmt": save_format,
        "q": jpeg_quality,
    }
    uid = hashlib.md5(repr(sorted(key.items())).encode()).hexdigest()[:10]
    out_dir = cache_root / f"{level['name']}_{uid}"
    if out_dir.exists():
        return out_dir

    ds = datasets.load_dataset(orig_name, orig_cfg, split=split)
    if not isinstance(ds.features[source_col], HFImage): ds = ds.cast_column(source_col, HFImage())
    if not isinstance(ds.features[mask_col],  HFImage): ds = ds.cast_column(mask_col,  HFImage())

    imgs_dir = out_dir / "images"; imgs_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    dset = HFDSourceMaskDataset(ds, source_col, mask_col)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        collate_fn=pad_collate,
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    sigma = float(level["difficulty_blur_sigma"])
    alpha = float(level["difficulty_gray_alpha"])
    kx, ky, radius = gaussian_kernels_1d(sigma, device)

    executor = ThreadPoolExecutor(max_workers=writer_threads)
    def submit_write(idx_int: int, img_u8_chw: torch.Tensor):
        fn = imgs_dir / f"{idx_int}.{save_format}"
        if save_format == "png":
            return executor.submit(write_png, img_u8_chw, str(fn))
        else:
            return executor.submit(write_jpeg, img_u8_chw, str(fn), quality=jpeg_quality)

    futures = []
    with tqdm(total=len(dset), desc=f"{level['name']} (σ={sigma}, α={alpha})") as pbar:
        for idxs, batch_u8, masks_u8, sizes in loader:
            batch_u8 = batch_u8.to(device, non_blocking=True)   # [B,3,H,W]
            masks_u8 = masks_u8.to(device, non_blocking=True)   # [B,1,H,W]
            out_u8 = masked_bg_gray_blur_batch_uint8(batch_u8, masks_u8, kx, ky, radius, alpha).cpu()

            for i in range(out_u8.size(0)):
                h, w = sizes[i].tolist()
                img = out_u8[i, :, :h, :w].contiguous()
                futures.append(submit_write(int(idxs[i]), img))
            pbar.update(out_u8.size(0))

    for f in tqdm(futures, desc="Flush writes"): f.result()
    executor.shutdown(wait=True)

    # Bind file paths back into a HF dataset
    files = [str(imgs_dir / f"{i}.{save_format}") for i in range(len(ds))]
    def path_mapper(example, idx):
        example[source_col] = files[idx]
        return example
    ds2 = ds.map(path_mapper, with_indices=True, num_proc=1, desc="Bind file paths")
    ds2 = ds2.cast_column(source_col, HFImage())

    tmp = Path(tempfile.mkdtemp(prefix="lvl_fast_"))
    try:
        ds2.save_to_disk(tmp)
        for p in tmp.iterdir(): shutil.move(str(p), str(out_dir / p.name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return out_dir

# --------- PREVIEW (side-by-side originals vs transformed) -------------------

def save_topk_previews(out_dir: Path,
                       level_name: str,
                       ds_name: str,
                       ds_cfg: str | None,
                       split: str,
                       source_col: str,
                       fmt: str,
                       topk: int):
    """
    Uses the already-saved transformed files in out_dir/images/{idx}.{fmt}
    and the ORIGINAL dataset to write side-by-side previews for idx in [0..topk-1].
    """
    if topk <= 0: return
    ds = datasets.load_dataset(ds_name, ds_cfg, split=split)
    if not isinstance(ds.features[source_col], HFImage):
        ds = ds.cast_column(source_col, HFImage())

    prev_dir = out_dir / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(topk, len(ds))):
        orig_pil = ds[i][source_col]  # PIL
        tr_path = out_dir / "images" / f"{i}.{fmt}"
        if not tr_path.exists():
            # if the full dataset wasn't built, skip missing ones gracefully
            continue
        tr_pil = Image.open(tr_path).convert("RGB")
        # side-by-side
        w = orig_pil.width + tr_pil.width
        h = max(orig_pil.height, tr_pil.height)
        board = Image.new("RGB", (w, h))
        board.paste(orig_pil, (0, 0))
        board.paste(tr_pil, (orig_pil.width, 0))
        board.save(prev_dir / f"idx{i}_{level_name}.png")

# -------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels_yaml", default="configs/levels.yaml")
    ap.add_argument("--base_yaml",   default="configs/base.yaml")
    ap.add_argument("--cache_root",  default="levels_cache")
    ap.add_argument("--split",       default="train")
    ap.add_argument("--only_level",  type=str, default=None)
    ap.add_argument("--loader_workers", type=int, default=32)
    ap.add_argument("--batch_size",     type=int, default=64)
    ap.add_argument("--format",         type=str, default="jpg", choices=["jpg","png"])
    ap.add_argument("--jpeg_quality",   type=int, default=92)
    ap.add_argument("--device",         type=str, default="cuda")
    ap.add_argument("--writer_threads", type=int, default=8)
    # NEW: optional previews
    ap.add_argument("--preview_topk",   type=int, default=0, help="If >0, save side-by-side previews for top-K indices per level")
    args = ap.parse_args()

    levels = yaml.safe_load(open(args.levels_yaml))["levels"]
    base   = yaml.safe_load(open(args.base_yaml))
    if args.only_level:
        levels = [lv for lv in levels if lv["name"] == args.only_level]
        if not levels:
            raise SystemExit(f"No level named '{args.only_level}' in {args.levels_yaml}")

    source_col = base["source_image_column"]
    mask_col   = base.get("mask_column")
    if not mask_col:
        raise SystemExit("base.yaml must define mask_column for masked transformation.")

    orig_name  = base["dataset_name"]
    orig_cfg   = base.get("dataset_config_name", None)

    Path(args.cache_root).mkdir(parents=True, exist_ok=True)

    for lvl in levels:
        out = build_level_dataset_fast_masked(
            orig_name=orig_name,
            orig_cfg=orig_cfg,
            level=lvl,
            source_col=source_col,
            mask_col=mask_col,
            cache_root=Path(args.cache_root),
            split=args.split,
            loader_workers=args.loader_workers,
            batch_size=args.batch_size,
            save_format=args.format,
            jpeg_quality=args.jpeg_quality,
            device=args.device,
            writer_threads=args.writer_threads,
        )
        print(f"✅ {lvl['name']} → {out}")
        if args.preview_topk > 0:
            save_topk_previews(
                out_dir=out,
                level_name=lvl["name"],
                ds_name=orig_name,
                ds_cfg=orig_cfg,
                split=args.split,
                source_col=source_col,
                fmt=args.format,
                topk=args.preview_topk,
            )
            print(f"   ↳ previews: {out / 'preview'}")

if __name__ == "__main__":
    main()
