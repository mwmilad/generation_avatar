from __future__ import annotations

import cv2
import numpy as np

from .config import RefineConfig
from .render import RenderResult
from .utils import colorize_depth, device


def _to_multiple_of_8(value: int) -> int:
    return max(8, int(round(value / 8)) * 8)


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(3, radius | 1)
    return cv2.GaussianBlur(mask.astype(np.float32), (radius, radius), 0)


def _first_method(obj, *names) -> bool:
    """Call whichever of these methods exists. Returns whether one did."""
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            method()
            return True
    return False


def _load(cls, model_id: str, **kwargs):
    """Load a checkpoint, preferring its fp16 weights when it publishes them."""
    if kwargs.get("torch_dtype") is not None:
        try:
            return cls.from_pretrained(model_id, variant="fp16", **kwargs)
        except Exception:
            pass
    return cls.from_pretrained(model_id, **kwargs)


def _fp16_vae(dtype) -> dict:
    """SDXL's own VAE overflows in fp16 and has to be run upcast to fp32.

    On a 16GB card that upcast is the difference between fitting and not, so
    the drop-in fp16-safe VAE is used when it is available. Failing to fetch
    it is not fatal - diffusers falls back to upcasting.
    """
    try:
        from diffusers import AutoencoderKL

        return {"vae": AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)}
    except Exception as exc:
        print(f"[avatar] fp16-safe VAE unavailable ({type(exc).__name__}); using the default VAE")
        return {}


def _apply_memory_savings(pipe, dev: str) -> None:
    """Enable offloading and tiling, tolerating diffusers' API moves.

    VAE tiling and slicing used to be pipeline methods and now live on the VAE
    itself, so both spellings are attempted rather than assuming a version.
    """
    if dev != "cuda":
        pipe.to(dev)
        return

    if not _first_method(pipe, "enable_model_cpu_offload"):
        pipe.to(dev)

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        _first_method(vae, "enable_tiling") or _first_method(pipe, "enable_vae_tiling")
        _first_method(vae, "enable_slicing") or _first_method(pipe, "enable_vae_slicing")
    _first_method(pipe, "enable_attention_slicing")


class Refiner:
    """SDXL + depth-ControlNet passes that add photographic texture to the render.

    Every pass runs at low denoising strength with the rendered depth as the
    control signal. That combination is deliberate: the geometry already
    encodes the subject's real proportions, and the control plus low strength
    is what stops the model from quietly slimming or idealising the body.
    """

    def __init__(self, cfg: RefineConfig):
        self.cfg = cfg
        self._pipe = None

    def _pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        dev = device()
        dtype = torch.float16 if dev == "cuda" else torch.float32

        controlnet = _load(ControlNetModel, self.cfg.controlnet_model, torch_dtype=dtype)
        extra = {}
        if dev == "cuda" and self.cfg.vae_fp16_fix:
            extra = _fp16_vae(dtype)

        pipe = _load(
            StableDiffusionXLControlNetImg2ImgPipeline,
            self.cfg.base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            **extra,
        )
        pipe.set_progress_bar_config(disable=True)
        _apply_memory_savings(pipe, dev)
        self._pipe = pipe
        return pipe

    def _run(self, rgb: np.ndarray, control: np.ndarray, strength: float, steps: int) -> np.ndarray:
        import torch
        from PIL import Image

        h, w = rgb.shape[:2]
        size = (_to_multiple_of_8(w), _to_multiple_of_8(h))
        init = Image.fromarray(rgb).resize(size, Image.LANCZOS)
        ctrl = Image.fromarray(control).convert("RGB").resize(size, Image.LANCZOS)

        pipe = self._pipeline()
        generator = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        out = pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            image=init,
            control_image=ctrl,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=self.cfg.guidance_scale,
            controlnet_conditioning_scale=self.cfg.controlnet_scale,
            generator=generator,
        ).images[0]
        return np.asarray(out.resize((w, h), Image.LANCZOS))

    def __call__(self, result: RenderResult, head_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
        cfg = self.cfg
        control = colorize_depth(result.depth, np.isfinite(result.depth))

        image = self._run(result.color, control, cfg.strength, cfg.steps)

        visible = result.alpha > 0.5
        novel = (result.novel > 0.5) & visible
        novel_fraction = novel.sum() / max(int(visible.sum()), 1)
        if cfg.novel_pass and novel_fraction > cfg.novel_min_area:
            stronger = self._run(result.color, control, cfg.novel_strength, cfg.steps)
            weight = _feather(novel, int(0.02 * image.shape[0]))[:, :, None]
            image = np.clip(image * (1 - weight) + stronger * weight, 0, 255).astype(np.uint8)

        if cfg.face_pass and head_box is not None:
            image = self._refine_face(image, control, head_box)
        return image

    def _refine_face(self, image: np.ndarray, control: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        h, w = image.shape[:2]
        x0, y0, x1, y1 = box
        # Square, padded crop, so the face keeps SDXL's expected framing.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * 0.75
        x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(h, cy + half))
        if x1 - x0 < 64 or y1 - y0 < 64:
            return image

        crop = image[y0:y1, x0:x1]
        ctrl_crop = control[y0:y1, x0:x1]
        scale = 1024 / max(crop.shape[:2])
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        big_ctrl = cv2.resize(ctrl_crop, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        refined = self._run(big, big_ctrl, self.cfg.face_strength, self.cfg.face_steps)
        refined = cv2.resize(refined, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)

        blend = np.zeros(crop.shape[:2], np.float32)
        inset = int(0.08 * min(crop.shape[:2])) + 1
        blend[inset:-inset, inset:-inset] = 1.0
        blend = _feather(blend, int(0.12 * min(crop.shape[:2])))[:, :, None]

        out = image.copy()
        out[y0:y1, x0:x1] = np.clip(crop * (1 - blend) + refined * blend, 0, 255).astype(np.uint8)
        return out
