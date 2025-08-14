import os
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import shutil
import tempfile
import json

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
    alpha: float,
    brightness: float,       # 0..200 like your YAML
) -> torch.Tensor:
    """
    out = where(mask<128, src, brightness_adjust( mix(gray(blur(src)), blur(src), alpha) ))
    Brightness adjust ~= OpenCV beta shift with beta=(brightness-50)*2.55
    """
    # Normalize to [0,1]
    x = src_u8.to(torch.float32) / 255.0
    m = (m_u8.to(torch.float32) < 128.0).to(torch.float32)  # 1 where KEEP original, else 0

    # --- Gaussian blur (separable) ---
    xw = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    h  = F.conv2d(xw, kx, groups=3)
    xh = F.pad(h, (0, 0, radius, radius), mode="reflect")
    blurred = F.conv2d(xh, ky, groups=3)

    # --- Grayscale mix (alpha: 0=full color, 1=full gray) ---
    r, g, b = blurred[:, 0:1], blurred[:, 1:2], blurred[:, 2:3]
    gray3 = (0.299 * r + 0.587 * g + 0.114 * b).repeat(1, 3, 1, 1)
    mixed = gray3 * float(alpha) + blurred * (1.0 - float(alpha))

    # --- Brightness shift ---
    beta_u8 = (float(brightness) - 50.0) * 2.55   # uint8 space
    beta = beta_u8 / 255.0                        # [0..1] space
    processed = (mixed + beta).clamp(0.0, 1.0)

    # --- Compose with original using mask ---
    m3 = m.repeat(1, 3, 1, 1)
    out = m3 * x + (1.0 - m3) * processed
    out = (out.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return out


@torch.no_grad()
def create_progressive_mask_batch(
    original_masks: torch.Tensor,  # [B,1,H,W] uint8 masks (jewelry=black)
    white_coverage_percent: float,
    patch_size_range: tuple = (20, 80),
    device: torch.device = None,
    seed: int = None,
    max_iterations: int = 2000,  # Maximum attempts to reach target coverage
) -> torch.Tensor:
    """
    Improved progressive mask creation that accurately achieves target white coverage.
    """
    if device is None:
        device = original_masks.device
    
    B, _, H, W = original_masks.shape
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Process on GPU if possible
    masks_float = original_masks.float() / 255.0  # [B,1,H,W] in [0,1]
    jewelry_mask = (masks_float < 0.5).float()  # 1 where jewelry, 0 elsewhere
    
    if white_coverage_percent >= 99:
        # For very high coverage, all white except jewelry
        progressive_masks = torch.ones_like(masks_float)
        progressive_masks = progressive_masks * (1 - jewelry_mask)  # Set jewelry to black
        return (progressive_masks * 255).byte()
    
    # Create base masks (all black)
    progressive_masks = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
    
    # Calculate target white pixels for each image
    non_jewelry_mask = 1 - jewelry_mask  # Areas where we can place white
    total_non_jewelry_pixels = non_jewelry_mask.sum(dim=(2,3))  # [B,1] pixels per image
    target_white_pixels = (total_non_jewelry_pixels * white_coverage_percent / 100).long()
    
    # Track current white pixel count per image
    current_white_pixels = torch.zeros(B, device=device)
    
    # Adaptive patch generation
    iterations = 0
    min_patch_size = patch_size_range[0]
    max_patch_size = patch_size_range[1]
    
    while iterations < max_iterations:
        # Check which images need more white pixels
        needs_more = current_white_pixels < target_white_pixels.squeeze()
        if not needs_more.any():
            break  # All images have reached target
        
        # Adaptive patch size: use smaller patches as we get closer to target
        for b in range(B):
            if not needs_more[b]:
                continue
                
            # Calculate how far we are from target
            pixels_needed = target_white_pixels[b, 0].item() - current_white_pixels[b].item()
            total_target = target_white_pixels[b, 0].item()
            
            # Adaptive patch size based on remaining pixels needed
            if pixels_needed < 500:
                # Use smaller patches when close to target for fine control
                patch_h = torch.randint(min_patch_size//2, min_patch_size, (1,)).item()
                patch_w = torch.randint(min_patch_size//2, min_patch_size, (1,)).item()
            elif pixels_needed < 2000:
                # Medium patches for moderate coverage
                patch_h = torch.randint(min_patch_size, (min_patch_size + max_patch_size)//2, (1,)).item()
                patch_w = torch.randint(min_patch_size, (min_patch_size + max_patch_size)//2, (1,)).item()
            else:
                # Large patches for initial coverage
                patch_h = torch.randint((min_patch_size + max_patch_size)//2, max_patch_size, (1,)).item()
                patch_w = torch.randint((min_patch_size + max_patch_size)//2, max_patch_size, (1,)).item()
            
            # Random position
            max_y = max(1, H - patch_h)
            max_x = max(1, W - patch_w)
            y = torch.randint(0, max_y, (1,)).item()
            x = torch.randint(0, max_x, (1,)).item()
            
            # Create patch mask for this position
            patch_mask = torch.zeros_like(progressive_masks[b:b+1])
            patch_mask[0, 0, y:y+patch_h, x:x+patch_w] = 1.0
            
            # Only add white where: (1) not already white, (2) not jewelry
            valid_patch = patch_mask * (1 - progressive_masks[b:b+1]) * non_jewelry_mask[b:b+1]
            
            # Add the valid patch to the mask
            progressive_masks[b:b+1] += valid_patch
            
            # Update white pixel count
            current_white_pixels[b] += valid_patch.sum().item()
        
        iterations += 1
        
        # Check overall progress every 50 iterations
        if iterations % 50 == 0:
            current_coverage = (current_white_pixels / total_non_jewelry_pixels.squeeze()).mean().item() * 100
            if iterations % 200 == 0:  # Log less frequently
                print(f"Iteration {iterations}: Current coverage = {current_coverage:.1f}%, Target = {white_coverage_percent}%")
    
    # Final check and adjustment if needed
    for b in range(B):
        actual_coverage = (current_white_pixels[b] / total_non_jewelry_pixels[b, 0]).item() * 100
        if abs(actual_coverage - white_coverage_percent) > 5:  # Allow 5% tolerance
            print(f"Warning: Image {b} achieved {actual_coverage:.1f}% coverage instead of {white_coverage_percent}%")
    
    # Ensure jewelry pixels remain black (double-check)
    progressive_masks = progressive_masks * (1 - jewelry_mask)
    
    # Clamp values to [0,1] and convert to uint8
    progressive_masks = progressive_masks.clamp(0, 1)
    
    return (progressive_masks * 255).byte()


# -------------- HF -> Torch loader -------------

class HFDSourceMaskDataset(Dataset):
    def __init__(self, ds: datasets.Dataset, process_col: str, mask_col: str, 
                 mask_coverage: float = None):
        """
        process_col: The column to process (might be same as target)
        mask_col: The mask column
        mask_coverage: White coverage percentage for progressive masking (None = use original)
        """
        self.ds = ds
        self.process_col = process_col
        self.mask_col = mask_col
        self.mask_coverage = mask_coverage

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        pil_src = row[self.process_col]   # PIL RGB - the image to process
        pil_msk = row[self.mask_col]      # PIL L or RGB

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

        return idx, t_src, t_msk.unsqueeze(0), self.mask_coverage  # [1,H,W] mask


def pad_collate(batch):
    idxs, imgs, masks, coverages = zip(*batch)
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
    
    # Return mask coverage for this batch (should be same for all items)
    mask_coverage = coverages[0] if coverages[0] is not None else None
    
    return (
        torch.tensor(idxs, dtype=torch.long),
        imgs_b.contiguous(),
        masks_b.contiguous(),
        torch.tensor(sizes, dtype=torch.int32),
        mask_coverage
    )


# -------------- Builder -----------------------

def build_level_dataset_fast_masked(
    orig_name: str,
    orig_cfg: str | None,
    level: dict,
    source_col: str,
    target_col: str,
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
    """
    Build a processed dataset for one level where:
      - If source_col == target_col: Create a synthetic 'source' column with processed target
      - Otherwise: Process the source_col and leave target_col untouched
      - Additionally process masks according to difficulty_mask_coverage
    """
    cache_root = Path(cache_root); cache_root.mkdir(parents=True, exist_ok=True)
    
    # Keep UID stable 
    key = {
        "name": level["name"],
        "sigma": float(level["difficulty_blur_sigma"]),
        "alpha": float(level["difficulty_gray_alpha"]),
        "brightness": float(level.get("difficulty_brightness", 50.0)),
        "mask_coverage": float(level.get("difficulty_mask_coverage", 100.0)),  
        "source_col": source_col,
        "target_col": target_col,
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
    
    # Special case: when source and target are the same column
    same_column = (source_col == target_col)
    process_col = target_col if same_column else source_col
    
    # Cast columns to Image type
    if not isinstance(ds.features[process_col], HFImage): 
        ds = ds.cast_column(process_col, HFImage())
    if not isinstance(ds.features[mask_col], HFImage): 
        ds = ds.cast_column(mask_col, HFImage())

    imgs_dir = out_dir / "images"; imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"; masks_dir.mkdir(parents=True, exist_ok=True)  # NEW

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    
    # Get mask coverage for this level
    mask_coverage = float(level.get("difficulty_mask_coverage", 100.0))
    
    dset = HFDSourceMaskDataset(ds, process_col, mask_col, mask_coverage)
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
    brightness = float(level.get("difficulty_brightness", 50.0))
    kx, ky, radius = gaussian_kernels_1d(sigma, device)

    executor = ThreadPoolExecutor(max_workers=writer_threads)
    
    def submit_write_img(idx_int: int, img_u8_chw: torch.Tensor):
        fn = imgs_dir / f"{idx_int}.{save_format}"
        if save_format == "png":
            return executor.submit(write_png, img_u8_chw, str(fn))
        else:
            return executor.submit(write_jpeg, img_u8_chw, str(fn), quality=jpeg_quality)
    
    def submit_write_mask(idx_int: int, mask_u8: torch.Tensor):
        fn = masks_dir / f"{idx_int}.png"  # Always PNG for masks
        return executor.submit(write_png, mask_u8, str(fn))

    futures = []
    with tqdm(total=len(dset), desc=f"{level['name']} (σ={sigma}, α={alpha}, β={brightness}, mask={mask_coverage}%)") as pbar:
        for idxs, batch_u8, masks_u8, sizes, batch_mask_coverage in loader:
            batch_u8 = batch_u8.to(device, non_blocking=True)   # [B,3,H,W]
            masks_u8 = masks_u8.to(device, non_blocking=True)   # [B,1,H,W]
            
            # Process masks if coverage is specified and < 100
            if batch_mask_coverage is not None and batch_mask_coverage < 100.0:
                processed_masks_u8 = create_progressive_mask_batch(
                    masks_u8, 
                    batch_mask_coverage,
                    patch_size_range=(80, 150),
                    device=device
                )
            else:
                processed_masks_u8 = masks_u8
            
            # Process images with original masks (for blur/gray/brightness)
            out_u8 = masked_bg_gray_blur_batch_uint8(
                batch_u8, masks_u8, kx, ky, radius, alpha, brightness
            ).cpu()
            
            # Save both processed images and processed masks
            processed_masks_cpu = processed_masks_u8.cpu()

            for i in range(out_u8.size(0)):
                h, w = sizes[i].tolist()
                img = out_u8[i, :, :h, :w].contiguous()
                mask = processed_masks_cpu[i, :, :h, :w].contiguous()
                
                futures.append(submit_write_img(int(idxs[i]), img))
                futures.append(submit_write_mask(int(idxs[i]), mask))
            pbar.update(out_u8.size(0))

    for f in tqdm(futures, desc="Flush writes"): f.result()
    executor.shutdown(wait=True)

    # Bind processed file paths back into a HF dataset
    img_files = [str(imgs_dir / f"{i}.{save_format}") for i in range(len(ds))]
    mask_files = [str(masks_dir / f"{i}.png") for i in range(len(ds))]

    if same_column:
        # Special case: source and target are the same column
        print(f"INFO: source_col == target_col ('{source_col}'). Creating synthetic 'source' column.")
        
        def path_mapper(example, idx):
            # Create a new 'source' column with processed image
            example['source'] = img_files[idx]
            # Update mask column with processed mask
            example[mask_col] = mask_files[idx]
            # Keep the original target column unchanged
            return example
        
        ds2 = ds.map(path_mapper, with_indices=True, num_proc=1, desc="Create synthetic source column")
        ds2 = ds2.cast_column('source', HFImage())  # processed source
        ds2 = ds2.cast_column(mask_col, HFImage())  # processed mask
    else:
        # Normal case: source and target are different columns
        def path_mapper(example, idx):
            # Replace source_col with processed path
            example[source_col] = img_files[idx]
            # Update mask column with processed mask
            example[mask_col] = mask_files[idx]
            # DO NOT touch target column (GT remains original)
            return example
        
        ds2 = ds.map(path_mapper, with_indices=True, num_proc=1, desc="Bind processed paths")
        ds2 = ds2.cast_column(source_col, HFImage())  # processed source
        ds2 = ds2.cast_column(mask_col, HFImage())   # processed mask

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
                       process_col: str,
                       mask_col: str,
                       fmt: str,
                       topk: int):
    """
    Uses the already-saved transformed files in out_dir/images/{idx}.{fmt}
    and out_dir/masks/{idx}.png to create preview comparisons
    """
    if topk <= 0: return
    ds = datasets.load_dataset(ds_name, ds_cfg, split=split)
    if not isinstance(ds.features[process_col], HFImage):
        ds = ds.cast_column(process_col, HFImage())
    if not isinstance(ds.features[mask_col], HFImage):
        ds = ds.cast_column(mask_col, HFImage())

    prev_dir = out_dir / "preview"
    prev_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(topk, len(ds))):
        orig_pil = ds[i][process_col]  # original image
        orig_mask = ds[i][mask_col]    # original mask
        
        tr_path = out_dir / "images" / f"{i}.{fmt}"
        mask_path = out_dir / "masks" / f"{i}.png"
        
        if not tr_path.exists() or not mask_path.exists():
            continue
            
        tr_pil = Image.open(tr_path).convert("RGB")
        tr_mask = Image.open(mask_path).convert("L")
        
        # Create 2x2 grid: [orig_img, trans_img], [orig_mask, trans_mask]
        w = orig_pil.width
        h = orig_pil.height
        board = Image.new("RGB", (w * 2, h * 2))
        
        # Top row: images
        board.paste(orig_pil, (0, 0))
        board.paste(tr_pil, (w, 0))
        
        # Bottom row: masks
        board.paste(orig_mask.convert("RGB"), (0, h))
        board.paste(tr_mask.convert("RGB"), (w, h))
        
        board.save(prev_dir / f"idx{i}_{level_name}.png")


# -------------------------- CLI / Main ---------------------------------------

def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser("Build processed source-only datasets for curriculum levels")
    parser.add_argument("--levels_yaml", type=str, required=True, help="Path to levels.yaml")
    parser.add_argument("--base_yaml", type=str, required=True, help="Path to base.yaml")
    parser.add_argument("--cache_root", type=str, required=True, help="Where to write level caches")
    parser.add_argument("--only_level", type=str, default=None, help="Build only this level name")
    parser.add_argument("--split", type=str, default="train", help="Split to process (default: train)")
    parser.add_argument("--fmt", type=str, default="jpg", choices=["jpg", "png"], help="Save format")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (if fmt=jpg)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for processing kernels")
    parser.add_argument("--writer_threads", type=int, default=8, help="Parallel file writers")
    parser.add_argument("--preview_topk", type=int, default=8, help="Optional preview grid count")
    args = parser.parse_args()

    levels_cfg = _load_yaml(args.levels_yaml)
    base_cfg = _load_yaml(args.base_yaml)

    # Required fields from base config
    ds_name = base_cfg["dataset_name"]
    ds_cfg  = base_cfg.get("dataset_config_name", None)
    source_col = base_cfg["source_image_column"]
    target_col = base_cfg.get("target_image_column", "target")
    mask_col   = base_cfg["mask_column"]

    levels = levels_cfg.get("levels", [])
    if args.only_level:
        levels = [lvl for lvl in levels if lvl["name"] == args.only_level]

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    built = []
    for lvl in levels:
        out_dir = build_level_dataset_fast_masked(
            orig_name=ds_name,
            orig_cfg=ds_cfg,
            level=lvl,
            source_col=source_col,
            target_col=target_col,
            mask_col=mask_col,
            cache_root=cache_root,
            split=args.split,
            save_format=args.fmt,
            jpeg_quality=args.quality,
            device=args.device,
            writer_threads=args.writer_threads,
        )
        built.append({"name": lvl["name"], "path": str(out_dir)})
        
        # For preview, use the column that was actually processed
        process_col = target_col if source_col == target_col else source_col
        try:
            save_topk_previews(out_dir, lvl["name"], ds_name, ds_cfg, args.split, 
                             process_col, mask_col, args.fmt, args.preview_topk)
        except Exception:
            pass  # previews are best-effort

    # Emit a small summary file
    summary = cache_root / "last_build_summary.json"
    with open(summary, "w") as f:
        json.dump(built, f, indent=2)
    print(json.dumps(built, indent=2))


if __name__ == "__main__":
    main()