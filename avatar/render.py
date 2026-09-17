from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import RenderConfig
from .geometry import Camera, Subject
from .utils import hex_to_rgb


@dataclass
class RenderResult:
    color: np.ndarray  # HxWx3 uint8, subject composited on the backdrop
    depth: np.ndarray  # HxW float32, +inf outside the subject
    alpha: np.ndarray  # HxW float32 in [0, 1]
    novel: np.ndarray  # HxW float32, surfaces the source photo never saw


def _zbuffer(uv: np.ndarray, z: np.ndarray, width: int, height: int):
    """Nearest-surface index per pixel, or -1. Plain painter's algorithm by depth."""
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok = (z > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        raise ValueError("nothing projects into the target view - check camera settings")

    flat = v[idx] * width + u[idx]
    # Sort by pixel, then by depth, so the first row of each pixel group is the
    # nearest surface hitting it.
    order = np.lexsort((z[idx], flat))
    flat_sorted = flat[order]
    _, first = np.unique(flat_sorted, return_index=True)

    winners = np.full(width * height, -1, np.int64)
    winners[flat_sorted[first]] = idx[order[first]]
    return winners.reshape(height, width)


def _silhouette(filled: np.ndarray) -> np.ndarray:
    """Solid subject outline from the splatted pixels, holes included."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(closed, contours, -1, 1, cv2.FILLED)
    return closed.astype(bool)


def _inpaint_holes(color: np.ndarray, depth: np.ndarray, holes: np.ndarray):
    if not holes.any():
        return color, depth
    mask8 = holes.astype(np.uint8)
    color = cv2.inpaint(color, mask8, 4, cv2.INPAINT_TELEA)

    finite = np.isfinite(depth)
    if finite.any():
        lo, hi = float(depth[finite].min()), float(depth[finite].max())
        span = max(hi - lo, 1e-6)
        as8 = np.where(finite, (depth - lo) / span * 255, 0).astype(np.uint8)
        painted = cv2.inpaint(as8, mask8, 4, cv2.INPAINT_TELEA)
        depth = np.where(holes, painted.astype(np.float32) / 255.0 * span + lo, depth)
    return color, depth


def _ground_shadow(camera: Camera, subject: Subject, strength: float) -> np.ndarray:
    """Soft contact shadow, rasterised on the real ground plane under the feet."""
    along = subject.points @ subject.up
    ground_level = float(np.percentile(along[~subject.is_back], 0.5))
    feet = subject.points[along < ground_level + 0.12 * subject.height_m]
    if len(feet) < 10:
        return np.zeros((camera.height, camera.width), np.float32)

    offsets = feet - subject.base
    horizontal = offsets - np.outer(offsets @ subject.up, subject.up)
    radius = float(np.percentile(np.linalg.norm(horizontal, axis=1), 95)) * 1.8
    radius = max(radius, 0.12)

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 40000)
    r = radius * np.sqrt(rng.uniform(0, 1, 40000))
    axis_a = subject.radial
    axis_b = np.cross(subject.up, subject.radial)
    disc = subject.base + np.outer(r * np.cos(theta), axis_a) + np.outer(r * np.sin(theta), axis_b)

    uv, z = camera.project(disc)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok = (z > 1e-4) & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

    shadow = np.zeros((camera.height, camera.width), np.float32)
    if ok.any():
        np.add.at(shadow, (v[ok], u[ok]), 1.0)
    shadow = np.clip(shadow, 0, 1)
    blur = max(3, int(0.05 * camera.height) | 1)
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    peak = shadow.max()
    if peak > 1e-6:
        shadow /= peak
    return shadow * strength


def render(camera: Camera, subject: Subject, cfg: RenderConfig) -> RenderResult:
    uv, z = camera.project(subject.points)
    winners = _zbuffer(uv, z, camera.width, camera.height)
    filled = winners >= 0

    color = np.zeros((camera.height, camera.width, 3), np.uint8)
    depth = np.full((camera.height, camera.width), np.inf, np.float32)
    novel = np.zeros((camera.height, camera.width), np.float32)

    hit = winners[filled]
    color[filled] = subject.colors[hit]
    depth[filled] = z[hit].astype(np.float32)
    novel[filled] = subject.is_back[hit].astype(np.float32)

    silhouette = _silhouette(filled)
    holes = silhouette & ~filled
    color, depth = _inpaint_holes(color, depth, holes)
    novel[holes] = 1.0
    depth[~silhouette] = np.inf

    alpha = cv2.GaussianBlur(silhouette.astype(np.float32), (3, 3), 0)

    background = np.zeros_like(color, np.float32)
    background[:] = hex_to_rgb(cfg.background)
    if cfg.shadow:
        shadow = _ground_shadow(camera, subject, cfg.shadow_strength)
        background *= (1.0 - shadow)[:, :, None]

    blended = color.astype(np.float32) * alpha[:, :, None] + background * (1 - alpha[:, :, None])
    return RenderResult(
        color=np.clip(blended, 0, 255).astype(np.uint8),
        depth=depth,
        alpha=alpha,
        novel=novel,
    )


def downsample(result: RenderResult, width: int, height: int) -> RenderResult:
    size = (width, height)
    finite = np.isfinite(result.depth)
    far = float(result.depth[finite].max()) if finite.any() else 1.0
    depth_small = cv2.resize(np.where(finite, result.depth, far), size, interpolation=cv2.INTER_AREA)
    alpha_small = cv2.resize(result.alpha, size, interpolation=cv2.INTER_AREA)
    return RenderResult(
        color=cv2.resize(result.color, size, interpolation=cv2.INTER_AREA),
        depth=np.where(alpha_small > 0.5, depth_small, np.inf).astype(np.float32),
        alpha=alpha_small,
        novel=cv2.resize(result.novel, size, interpolation=cv2.INTER_AREA),
    )
