from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Source-photo camera assumptions and the target camera to re-render from."""

    # 35mm-equivalent focal length of the lens that took the source photo.
    # Phone main cameras are ~26-28mm; a dedicated portrait lens is 50-85mm.
    # Getting this wrong mostly changes how strong the perspective change looks.
    focal_mm_equiv: float = 28.0

    # Real-world height of the subject in metres. Used to put the reconstruction
    # on a metric scale so that camera heights below are meaningful.
    subject_height_m: float = 1.70

    # Target camera height above the ground in metres. `None` keeps the source
    # height; a value lower than the estimated source height lowers the camera.
    target_camera_height_m: float | None = 1.15

    # Alternatively, shift the camera relative to the estimated source height.
    # Applied only when target_camera_height_m is None.
    camera_height_delta_m: float = 0.0

    # Distance from camera to the subject's vertical axis. `None` keeps the
    # distance estimated from the source photo.
    target_distance_m: float | None = None

    # Where the camera aims, as a fraction of body height measured from the
    # feet (0.5 = waist, 0.9 = head). Keeps the subject framed after moving.
    aim_height_frac: float = 0.55

    # Yaw the camera around the subject, in degrees. 0 keeps the original side.
    orbit_deg: float = 0.0

    # 35mm-equivalent focal length for the render. This only decides framing -
    # perspective comes from where the camera stands, so use target_distance_m
    # to flatten or exaggerate it. `None` fits the lens to the subject.
    out_focal_mm_equiv: float | None = None

    # Fraction of the frame kept clear around the subject when fitting.
    frame_margin: float = 0.08


@dataclass
class GeometryConfig:
    # Supersampling factor applied to the depth/colour grid before unprojection.
    # `None` picks it from how much the subject is being magnified between the
    # source photo and the render, which is what keeps the point cloud dense
    # enough to cover the output without guessing. Higher costs memory.
    point_upsample: int | None = None
    max_point_upsample: int = 4

    # Thickness of the synthesised back shell, relative to a circular
    # cross-section. 1.0 makes a limb of width w roughly w deep.
    back_shell_scale: float = 0.9

    # Depth discontinuity (metres) above which neighbouring pixels are treated
    # as different surfaces and the boundary is trimmed.
    depth_edge_threshold_m: float = 0.04

    # Erode the subject mask by this many pixels to drop matting fringe.
    mask_erode_px: int = 2


@dataclass
class RenderConfig:
    width: int = 896
    height: int = 1216

    # Supersampling of the render buffer; downsampled at the end.
    supersample: int = 2

    # Radius in output pixels of each splat. Larger closes holes but softens.
    splat_radius: float = 1.0

    background: str = "#f2f0ee"

    # Soft contact shadow on the ground under the subject.
    shadow: bool = True
    shadow_strength: float = 0.28


@dataclass
class RefineConfig:
    enabled: bool = True

    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "diffusers/controlnet-depth-sdxl-1.0"

    prompt: str = (
        "full body studio photograph of a person, standing, plain light grey "
        "seamless backdrop, soft diffused key light, sharp focus, natural skin "
        "texture, detailed fabric texture, high resolution, photorealistic"
    )
    negative_prompt: str = (
        "cartoon, illustration, painting, cgi, 3d render, plastic skin, "
        "distorted body, extra limbs, deformed hands, blurry, low quality, "
        "watermark, text"
    )

    # Swap in a VAE that is numerically safe in fp16. Saves the fp32 upcast
    # SDXL's own VAE needs, which matters on a 16GB card.
    vae_fp16_fix: bool = True

    # Denoising strength for the body pass. Keep this low: the render already
    # carries the true body geometry and high strength lets the model reshape it.
    strength: float = 0.30
    steps: int = 34
    guidance_scale: float = 5.5
    controlnet_scale: float = 0.8
    seed: int = 0

    # Extra pass at a higher strength over surfaces the source photo never saw
    # (the synthesised back shell, and filled holes), which the low-strength
    # body pass leaves looking flat. Skipped when there is little of it.
    novel_pass: bool = True
    novel_strength: float = 0.55
    novel_min_area: float = 0.03  # as a fraction of the subject, not the frame

    # Second pass over the head crop only, to recover facial detail.
    face_pass: bool = True
    face_strength: float = 0.28
    face_steps: int = 30
    # Head crop as a fraction of body height, measured down from the top.
    face_crop_frac: float = 0.22


@dataclass
class PipelineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)

    # Default is a plain SegFormer human-parsing model: no `trust_remote_code`,
    # so it does not break when transformers changes its model API.
    # "briaai/RMBG-1.4" gives a finer matte and still works, but relies on
    # custom code that lags behind transformers releases.
    segmenter: str = "mattmdjaga/segformer_b2_clothes"

    # Sharpen the mask against the photo's edges with GrabCut. Worth it for
    # semantic segmentation, unnecessary after a dedicated matting model.
    refine_mask_edges: bool = True
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

    # Longest side the source photo is resized to before depth estimation.
    work_resolution: int = 1024

    # Write every intermediate (mask, depth, raw render, depth buffer) next to
    # the output. Useful while tuning; costs a little disk.
    save_intermediates: bool = True
