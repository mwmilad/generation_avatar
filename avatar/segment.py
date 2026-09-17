from __future__ import annotations

import cv2
import numpy as np

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _preprocess(rgb: np.ndarray, size: int, mean, std):
    import torch

    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def subject_alpha(rgb: np.ndarray, model_id: str = "briaai/RMBG-1.4") -> np.ndarray:
    """Soft alpha matte of the person, float32 in [0, 1] at the input resolution."""
    import torch
    from transformers import AutoModelForImageSegmentation

    from .utils import device

    dev = device()
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model.to(dev).eval()

    if "rmbg" in model_id.lower():
        # RMBG-1.4 was trained with a plain 0.5/1.0 normalisation, not ImageNet stats.
        x = _preprocess(rgb, 1024, (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
    else:
        x = _preprocess(rgb, 1024, _IMAGENET_MEAN, _IMAGENET_STD)

    with torch.no_grad():
        out = model(x.to(dev))
    while isinstance(out, (list, tuple)):
        out = out[0]
    alpha = torch.sigmoid(out).squeeze().float().cpu().numpy()

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    alpha = alpha - alpha.min()
    if alpha.max() > 1e-6:
        alpha = alpha / alpha.max()
    h, w = rgb.shape[:2]
    return cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Drop stray blobs so a mirror, shadow or second person doesn't get lifted."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 2:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def clean_mask(alpha: np.ndarray, threshold: float = 0.5, erode_px: int = 2) -> np.ndarray:
    """Binary subject mask: largest blob, holes filled, fringe eroded away."""
    mask = alpha > threshold
    if not mask.any():
        raise ValueError("segmentation found no subject - check the input photo")
    mask = largest_component(mask)

    filled = mask.astype(np.uint8)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    mask = filled.astype(bool)

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        mask = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    return mask
