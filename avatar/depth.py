from __future__ import annotations

import cv2
import numpy as np


def estimate_depth(rgb: np.ndarray, model_id: str) -> tuple[np.ndarray, bool]:
    """Per-pixel depth for the photo.

    Returns (depth, is_metric). Metric checkpoints return depth in metres;
    relative ones return inverse depth, which we invert here and which is then
    only correct up to an unknown scale and shift.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from .utils import device

    dev = device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(dev).eval()

    inputs = processor(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        pred = model(**inputs).predicted_depth

    h, w = rgb.shape[:2]
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1).float(), size=(h, w), mode="bicubic", align_corners=False
    )
    depth = pred.squeeze().cpu().numpy().astype(np.float32)

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    is_metric = "metric" in model_id.lower()
    if not is_metric:
        depth = 1.0 / np.clip(depth, 1e-3, None)
    return depth, is_metric


def smooth_depth(depth: np.ndarray, mask: np.ndarray, diameter: int = 9) -> np.ndarray:
    """Edge-preserving smoothing inside the subject, so splats don't get speckled.

    Depth is shifted into a positive range before the bilateral filter because
    the filter's sigmas are expressed in value units, then shifted back.
    """
    inside = depth[mask]
    if inside.size == 0:
        return depth
    lo, hi = float(inside.min()), float(inside.max())
    span = max(hi - lo, 1e-6)

    norm = np.clip((depth - lo) / span, 0, 1).astype(np.float32)
    filtered = cv2.bilateralFilter(norm, diameter, 0.05, diameter)
    out = filtered * span + lo

    return np.where(mask, out, depth).astype(np.float32)


def fill_invalid(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace non-finite or non-positive depths inside the mask by inpainting."""
    bad = mask & (~np.isfinite(depth) | (depth <= 0))
    if not bad.any():
        return depth
    good = depth[mask & ~bad]
    fallback = float(np.median(good)) if good.size else 1.0
    filled = np.where(np.isfinite(depth) & (depth > 0), depth, fallback).astype(np.float32)

    lo, hi = float(filled.min()), float(filled.max())
    span = max(hi - lo, 1e-6)
    as8 = ((filled - lo) / span * 255).astype(np.uint8)
    painted = cv2.inpaint(as8, bad.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return (painted.astype(np.float32) / 255.0 * span + lo).astype(np.float32)
