from __future__ import annotations

import argparse

from .config import PipelineConfig
from .pipeline import generate_avatar


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="avatar",
        description="Turn a full-body photo into a studio avatar re-shot from a different camera height.",
    )
    p.add_argument("image", help="path to the source full-body photo")
    p.add_argument("-o", "--out-dir", default="output")

    cam = p.add_argument_group("camera")
    cam.add_argument("--subject-height", type=float, default=1.70, help="real height of the person, metres")
    cam.add_argument("--camera-height", type=float, default=1.15, help="target camera height, metres")
    cam.add_argument("--camera-height-delta", type=float, help="instead, shift the estimated source height")
    cam.add_argument("--distance", type=float, help="target camera distance, metres")
    cam.add_argument("--orbit", type=float, default=0.0, help="orbit around the subject, degrees")
    cam.add_argument("--aim", type=float, default=0.55, help="aim point as a fraction of body height")
    cam.add_argument("--source-lens", type=float, default=28.0, help="35mm-equivalent lens of the source photo")
    cam.add_argument(
        "--output-lens",
        type=float,
        help="35mm-equivalent lens for the render; omit to fit the framing to the subject",
    )

    out = p.add_argument_group("output")
    out.add_argument("--width", type=int, default=896)
    out.add_argument("--height", type=int, default=1216)
    out.add_argument("--background", default="#f2f0ee")
    out.add_argument("--no-shadow", action="store_true")
    out.add_argument("--no-refine", action="store_true", help="stop after the 3D render")
    out.add_argument("--no-face-pass", action="store_true")
    out.add_argument("--strength", type=float, default=0.30, help="body refinement denoise strength")
    out.add_argument("--seed", type=int, default=0)
    out.add_argument("--no-intermediates", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.camera.subject_height_m = args.subject_height
    if args.camera_height_delta is not None:
        cfg.camera.target_camera_height_m = None
        cfg.camera.camera_height_delta_m = args.camera_height_delta
    else:
        cfg.camera.target_camera_height_m = args.camera_height
    cfg.camera.target_distance_m = args.distance
    cfg.camera.orbit_deg = args.orbit
    cfg.camera.aim_height_frac = args.aim
    cfg.camera.focal_mm_equiv = args.source_lens
    cfg.camera.out_focal_mm_equiv = args.output_lens

    cfg.render.width = args.width
    cfg.render.height = args.height
    cfg.render.background = args.background
    cfg.render.shadow = not args.no_shadow

    cfg.refine.enabled = not args.no_refine
    cfg.refine.face_pass = not args.no_face_pass
    cfg.refine.strength = args.strength
    cfg.refine.seed = args.seed

    cfg.save_intermediates = not args.no_intermediates
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    written = generate_avatar(args.image, config_from_args(args), args.out_dir)
    print("\n[avatar] wrote:")
    for name, path in written.items():
        print(f"  {name:14s} {path}")


if __name__ == "__main__":
    main()
