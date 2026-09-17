from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .depth import estimate_depth, fill_invalid, smooth_depth
from .geometry import (
    Camera,
    Subject,
    build_subject,
    equivalent_lens_mm,
    fit_camera_to_subject,
    resolve_camera_height,
    target_camera,
)
from .render import downsample, render
from .segment import clean_mask, refine_edges, subject_alpha
from .utils import colorize_depth, load_image, resize_long_side, save_image, step


def _head_box(camera: Camera, subject: Subject, crop_frac: float) -> tuple[int, int, int, int] | None:
    front = subject.points[~subject.is_back]
    along = front @ subject.up
    top = float(np.percentile(along, 99.5))
    head = front[along > top - crop_frac * subject.height_m]
    if len(head) < 50:
        return None

    uv, z = camera.project(head)
    uv = uv[z > 1e-4]
    if len(uv) < 50:
        return None
    x0, y0 = np.percentile(uv, 2, axis=0)
    x1, y1 = np.percentile(uv, 98, axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def _export_ply(path: Path, subject: Subject, max_points: int = 200_000) -> None:
    n = len(subject.points)
    idx = np.random.default_rng(0).choice(n, min(n, max_points), replace=False)
    pts, cols = subject.points[idx], subject.colors[idx]
    with open(path, "w") as fh:
        fh.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for (x, y, z), (r, g, b) in zip(pts, cols):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def generate_avatar(image_path: str | Path, cfg: PipelineConfig, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def emit(name: str, filename: str, image: np.ndarray, always: bool = False) -> None:
        if always or cfg.save_intermediates:
            path = out_dir / filename
            save_image(path, image)
            written[name] = path

    rgb = resize_long_side(load_image(image_path), cfg.work_resolution)

    with step("segmenting subject"):
        alpha = subject_alpha(rgb, cfg.segmenter)
        if cfg.refine_mask_edges:
            alpha = refine_edges(alpha, rgb)
        mask = clean_mask(alpha, erode_px=cfg.geometry.mask_erode_px)
        emit("mask", "01_mask.png", (mask * 255).astype(np.uint8)[:, :, None].repeat(3, 2))

    with step("estimating depth"):
        depth, is_metric = estimate_depth(rgb, cfg.depth_model)
        if not is_metric:
            print("[avatar] warning: relative depth model - geometry is scale/shift approximate")
        depth = fill_invalid(depth, mask)
        depth = smooth_depth(depth, mask)
        emit("depth", "02_depth.png", colorize_depth(depth, mask)[:, :, None].repeat(3, 2))

    ss = max(1, int(cfg.render.supersample))

    with step("reconstructing 3D subject"):
        # The subject is framed to roughly 84% of the render height, which is
        # what the point density needs to be able to cover.
        subject = build_subject(
            rgb, depth, mask, cfg.camera, cfg.geometry, 0.84 * cfg.render.height * ss
        )
        print(
            f"[avatar] source camera height {subject.source_camera_height_m:.2f}m, "
            f"distance {subject.source_distance_m:.2f}m, subject {subject.height_m:.2f}m"
        )
        if cfg.save_intermediates:
            ply = out_dir / "03_pointcloud.ply"
            _export_ply(ply, subject)
            written["pointcloud"] = ply

    with step("rendering from the target camera"):
        camera = target_camera(subject, cfg.camera, cfg.render.width * ss, cfg.render.height * ss)
        if cfg.camera.out_focal_mm_equiv is None:
            camera = fit_camera_to_subject(camera, subject, cfg.camera.frame_margin)
        print(
            f"[avatar] target camera height {resolve_camera_height(subject, cfg.camera):.2f}m, "
            f"distance {cfg.camera.target_distance_m or subject.source_distance_m:.2f}m, "
            f"lens {equivalent_lens_mm(camera):.0f}mm equivalent"
        )
        big = render(camera, subject, cfg.render)
        result = downsample(big, cfg.render.width, cfg.render.height) if ss > 1 else big
        visible = result.alpha > 0.5
        novel_frac = float(((result.novel > 0.5) & visible).sum()) / max(int(visible.sum()), 1)
        print(f"[avatar] surfaces the photo never saw: {novel_frac * 100:.1f}% of the subject")
        if novel_frac > 0.45:
            print(
                "[avatar] warning: most of the view is reconstructed rather than observed. "
                "Move the camera less, or raise geometry.max_point_upsample."
            )
        emit("render", "04_render.png", result.color, always=True)
        emit("render_depth", "05_render_depth.png", colorize_depth(result.depth)[:, :, None].repeat(3, 2))

    if not cfg.refine.enabled:
        return written

    with step("refining with SDXL + depth ControlNet"):
        from .refine import Refiner

        head = _head_box(camera, subject, cfg.refine.face_crop_frac)
        if head is not None and ss > 1:
            head = tuple(int(v / ss) for v in head)  # type: ignore[assignment]
        final = Refiner(cfg.refine)(result, head)
        emit("avatar", "avatar.png", final, always=True)

    return written
