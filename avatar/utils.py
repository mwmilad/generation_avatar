from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def step(name: str):
    print(f"[avatar] {name} ...", flush=True)
    t0 = time.time()
    yield
    print(f"[avatar] {name} done in {time.time() - t0:.1f}s", flush=True)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8, honouring EXIF orientation."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def save_image(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def resize_long_side(rgb: np.ndarray, target: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) == target:
        return rgb
    scale = target / max(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=interp)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def colorize_depth(depth: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Depth to an 8-bit near=white map, which is what depth ControlNets expect."""
    valid = np.isfinite(depth) if mask is None else (mask & np.isfinite(depth))
    out = np.zeros(depth.shape, np.float32)
    if valid.any():
        d = depth[valid]
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        out[valid] = 1.0 - np.clip((d - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def focal_px_from_35mm(focal_mm_equiv: float, width: int, height: int) -> float:
    """Pixel focal length for a 35mm-equivalent lens on an image of this size.

    The 35mm equivalence is defined on the frame diagonal (43.27mm), so the
    conversion has to go through the image diagonal rather than the width.
    """
    diag_px = float(np.hypot(width, height))
    return focal_mm_equiv * diag_px / 43.266615
