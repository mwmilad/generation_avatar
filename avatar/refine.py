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
        controlnet = ControlNetModel.from_pretrained(self.cfg.controlnet_model, torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            self.cfg.base_model, controlnet=controlnet, torch_dtype=dtype, variant="fp16" if dev == "cuda" else None
        )
        pipe.set_progress_bar_config(disable=True)
        if dev == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()
        else:
            pipe.to(dev)
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
