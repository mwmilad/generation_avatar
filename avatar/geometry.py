from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CameraConfig, GeometryConfig
from .utils import focal_px_from_35mm


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, k: float) -> "Intrinsics":
        return Intrinsics(self.fx * k, self.fy * k, (self.cx + 0.5) * k - 0.5, (self.cy + 0.5) * k - 0.5)


@dataclass
class Camera:
    intrinsics: Intrinsics
    width: int
    height: int
    # World -> camera rigid transform. The world frame here is the *source*
    # camera's frame, so the source camera has R = I and t = 0.
    rotation: np.ndarray
    position: np.ndarray

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project Nx3 world points. Returns (Nx2 pixel coords, Nx1 depth)."""
        rotation = self.rotation.astype(points.dtype, copy=False)
        position = self.position.astype(points.dtype, copy=False)
        cam = (points - position) @ rotation.T
        z = cam[:, 2]
        safe = np.clip(z, 1e-6, None)
        u = self.intrinsics.fx * cam[:, 0] / safe + self.intrinsics.cx
        v = self.intrinsics.fy * cam[:, 1] / safe + self.intrinsics.cy
        return np.stack([u, v], 1), z


@dataclass
class Subject:
    """The reconstructed person, in the source camera's coordinate frame."""

    points: np.ndarray  # Nx3 surface points, metres
    colors: np.ndarray  # Nx3 uint8
    is_back: np.ndarray  # N bool, True for synthesised back-shell points
    up: np.ndarray  # unit vector, feet -> head
    base: np.ndarray  # point on the body axis at ground level
    radial: np.ndarray  # unit horizontal vector from body axis towards the camera
    height_m: float
    source_camera_height_m: float
    source_distance_m: float


def _unproject(depth: np.ndarray, mask: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.nonzero(mask)
    z = depth[vs, us].astype(np.float64)
    x = (us - intr.cx) * z / intr.fx
    y = (vs - intr.cy) * z / intr.fy
    return np.stack([x, y, z], 1), np.stack([vs, us], 1)


def _trim_depth_edges(depth: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    """Drop mask pixels sitting on a depth cliff - they are matting fringe."""
    d = np.where(mask, depth, np.nan)
    hi = cv2.dilate(np.nan_to_num(d, nan=-1e9), np.ones((3, 3), np.uint8))
    lo = -cv2.dilate(np.nan_to_num(-d, nan=-1e9), np.ones((3, 3), np.uint8))
    return mask & ((hi - lo) < threshold)


def _body_axis(points: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Unit vector from feet to head, assuming the subject stands upright.

    Taken from the head and foot slabs of the reconstruction. Any roll
    component is dropped: a standing person's axis lies in the camera's
    vertical plane unless the camera itself was rolled, and forcing that
    keeps a noisy depth map from tilting the whole world frame.
    """
    rows = pixels[:, 0]
    top = rows <= np.percentile(rows, 6)
    bottom = rows >= np.percentile(rows, 94)
    axis = points[top].mean(0) - points[bottom].mean(0)
    axis[0] = 0.0
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0])
    return axis / norm


def _back_shell(
    points: np.ndarray,
    pixels: np.ndarray,
    mask: np.ndarray,
    scale: float,
    fx: float,
) -> np.ndarray:
    """Push each surface point backwards to give the body volume.

    Thickness comes from the silhouette itself: the distance transform gives
    the half-width of the body at every pixel, so a limb of width w gets a
    roughly circular cross-section of depth w. Deriving it from the real
    outline is what keeps a heavy body heavy instead of applying a constant
    slab thickness.
    """
    dist_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    half_width_px = dist_px[pixels[:, 0], pixels[:, 1]].astype(np.float64)

    rays = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-9)
    # Pixel size in metres at this point's distance, from the pinhole relation.
    metres_per_px = points[:, 2] / fx
    thickness = 2.0 * scale * half_width_px * metres_per_px
    return points + rays * thickness[:, None]


def _choose_upsample(
    mask: np.ndarray, geo_cfg: GeometryConfig, target_pixel_height: float | None
) -> int:
    """How many surface samples per source pixel the render will need.

    The subject usually occupies a small part of the source photo and most of
    the output frame. Unprojecting one point per source pixel then leaves the
    render mostly holes, which get inpainted and flagged as unseen surface -
    and handing that much of the frame to the refiner is exactly how the body
    shape drifts. Sampling at the magnification ratio avoids it.
    """
    if geo_cfg.point_upsample is not None:
        return max(1, int(geo_cfg.point_upsample))
    if target_pixel_height is None:
        return 2

    rows = np.nonzero(mask.any(axis=1))[0]
    source_height = max(int(rows.max() - rows.min()) + 1, 1)
    ratio = target_pixel_height / source_height
    return int(np.clip(np.ceil(ratio), 1, geo_cfg.max_point_upsample))


def build_subject(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    cam_cfg: CameraConfig,
    geo_cfg: GeometryConfig,
    target_pixel_height: float | None = None,
) -> Subject:
    h, w = rgb.shape[:2]
    focal = focal_px_from_35mm(cam_cfg.focal_mm_equiv, w, h)
    intr = Intrinsics(focal, focal, (w - 1) / 2.0, (h - 1) / 2.0)

    k = _choose_upsample(mask, geo_cfg, target_pixel_height)
    if k > 1:
        size = (w * k, h * k)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        intr = intr.scaled(k)

    mask = _trim_depth_edges(depth, mask, geo_cfg.depth_edge_threshold_m)
    if mask.sum() < 1000:
        raise ValueError("too few valid subject pixels after depth edge trimming")

    front, pixels = _unproject(depth, mask, intr)
    up = _body_axis(front, pixels)

    # Put the reconstruction on a metric scale using the subject's real height.
    # Scaling about the camera centre leaves the source projection unchanged.
    along = front @ up
    measured = float(np.percentile(along, 99.5) - np.percentile(along, 0.5))
    if measured <= 1e-6:
        raise ValueError("degenerate reconstruction - subject has no measurable height")
    front *= cam_cfg.subject_height_m / measured

    back = _back_shell(front, pixels, mask, geo_cfg.back_shell_scale, intr.fx)

    colors = rgb[pixels[:, 0], pixels[:, 1]]
    points = np.concatenate([front, back])
    colors = np.concatenate([colors, (colors * 0.72).astype(np.uint8)])
    is_back = np.concatenate([np.zeros(len(front), bool), np.ones(len(back), bool)])

    along = front @ up
    ground_level = float(np.percentile(along, 0.5))
    height_m = float(np.percentile(along, 99.5)) - ground_level

    # The body axis has to be averaged over the whole volume, front and back
    # shell together. Using the visible surface alone puts it half a body
    # depth too close to the camera, which then biases the rendered distance.
    centroid = points.mean(0)
    base = centroid - (float(centroid @ up) - ground_level) * up

    to_camera = -base  # the source camera sits at the origin
    radial = to_camera - float(to_camera @ up) * up
    distance = float(np.linalg.norm(radial))
    radial = radial / max(distance, 1e-9)

    return Subject(
        points=points.astype(np.float32),
        colors=colors,
        is_back=is_back,
        up=up,
        base=base,
        radial=radial,
        height_m=height_m,
        source_camera_height_m=float((-base) @ up),
        source_distance_m=distance,
    )


def resolve_camera_height(subject: Subject, cam_cfg: CameraConfig) -> float:
    if cam_cfg.target_camera_height_m is not None:
        return cam_cfg.target_camera_height_m
    return subject.source_camera_height_m + cam_cfg.camera_height_delta_m


def target_camera(subject: Subject, cam_cfg: CameraConfig, width: int, height: int) -> Camera:
    """Place a virtual camera at the requested height and aim it at the subject."""
    cam_height = resolve_camera_height(subject, cam_cfg)
    distance = cam_cfg.target_distance_m or subject.source_distance_m

    up = subject.up
    theta = np.radians(cam_cfg.orbit_deg)
    # Rodrigues rotation of the radial direction about the body axis.
    r = subject.radial
    direction = r * np.cos(theta) + np.cross(up, r) * np.sin(theta) + up * (up @ r) * (1 - np.cos(theta))

    position = subject.base + direction * distance + up * cam_height
    aim = subject.base + up * (cam_cfg.aim_height_frac * subject.height_m)

    z_axis = aim - position
    z_axis /= max(np.linalg.norm(z_axis), 1e-9)
    down = -up
    x_axis = np.cross(down, z_axis)
    x_axis /= max(np.linalg.norm(x_axis), 1e-9)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis])

    focal_mm = cam_cfg.out_focal_mm_equiv or 50.0
    focal = focal_px_from_35mm(focal_mm, width, height)
    intr = Intrinsics(focal, focal, (width - 1) / 2.0, (height - 1) / 2.0)
    return Camera(intr, width, height, rotation, position)


def fit_camera_to_subject(camera: Camera, subject: Subject, margin: float = 0.08) -> Camera:
    """Pick the focal length that frames the subject, leaving the camera put.

    Framing is a lens choice, not a position one. Dollying to fit would change
    the distance to the subject, and distance is what sets the perspective the
    caller asked for - so fitting that way would quietly undo the camera move.
    """
    front = subject.points[~subject.is_back]
    cam = (front - camera.position.astype(front.dtype)) @ camera.rotation.astype(front.dtype).T
    visible = cam[:, 2] > 1e-3
    if not visible.any():
        return camera
    cam = cam[visible]

    nx = np.abs(cam[:, 0] / cam[:, 2])
    ny = np.abs(cam[:, 1] / cam[:, 2])
    # 99.9th percentile rather than max, so one stray point cannot zoom out.
    extent_x = max(float(np.percentile(nx, 99.9)), 1e-6)
    extent_y = max(float(np.percentile(ny, 99.9)), 1e-6)

    focal = min(
        (camera.width / 2) * (1 - 2 * margin) / extent_x,
        (camera.height / 2) * (1 - 2 * margin) / extent_y,
    )
    intr = Intrinsics(focal, focal, camera.intrinsics.cx, camera.intrinsics.cy)
    return Camera(intr, camera.width, camera.height, camera.rotation, camera.position)


def equivalent_lens_mm(camera: Camera) -> float:
    """The 35mm-equivalent focal length a camera's intrinsics correspond to."""
    diagonal = float(np.hypot(camera.width, camera.height))
    return camera.intrinsics.fx * 43.266615 / diagonal
