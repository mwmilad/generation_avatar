from __future__ import annotations

import cv2
import numpy as np

# Human-parsing SegFormer. Plain transformers architecture, no `trust_remote_code`,
# which is what makes it survive transformers major versions.
HUMAN_PARSING_MODEL = "mattmdjaga/segformer_b2_clothes"

# Classes of that model that are not part of the body. 0 is the background and
# 16 is a carried bag, which would otherwise distort the silhouette - and the
# silhouette is what the reconstruction derives body thickness from.
_NON_BODY_LABELS = (0, 16)


def _install_tied_weights_shim() -> None:
    """Let models whose custom code predates `post_init()` still load.

    transformers sets `all_tied_weights_keys` on the instance inside
    `post_init()`. Hand-written remote code from before that exists - RMBG-1.4
    among it - never calls `post_init()`, so loading dies on a missing
    attribute. This installs a non-data descriptor as a class-level fallback:
    instance attributes take precedence over it, so a model that does call
    `post_init()` is completely unaffected, and one that does not gets its own
    empty dict rather than a shared one.
    """
    from transformers.modeling_utils import PreTrainedModel

    if "all_tied_weights_keys" in vars(PreTrainedModel):
        return

    class _PerInstanceDefault:
        def __get__(self, obj, objtype=None):
            if obj is None:
                return {}
            value: dict = {}
            obj.__dict__["all_tied_weights_keys"] = value
            return value

    PreTrainedModel.all_tied_weights_keys = _PerInstanceDefault()


def _is_human_parsing(model_id: str) -> bool:
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        return False
    architectures = getattr(config, "architectures", None) or [""]
    return "Segformer" in architectures[0] or len(getattr(config, "id2label", {}) or {}) > 2


def _parsing_alpha(rgb: np.ndarray, model_id: str) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    from .utils import device

    dev = device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(dev).eval()

    inputs = processor(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = model(**inputs).logits

    h, w = rgb.shape[:2]
    logits = torch.nn.functional.interpolate(
        logits.float(), size=(h, w), mode="bilinear", align_corners=False
    )
    probabilities = logits.softmax(dim=1)[0]

    body = torch.ones(probabilities.shape[0], dtype=torch.bool)
    for label in _NON_BODY_LABELS:
        if label < len(body):
            body[label] = False
    alpha = probabilities[body.to(probabilities.device)].sum(0).cpu().numpy()

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.clip(alpha, 0, 1).astype(np.float32)


def _matting_alpha(rgb: np.ndarray, model_id: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForImageSegmentation

    from .utils import device

    _install_tied_weights_shim()
    dev = device()
    model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
    model.to(dev).eval()

    # RMBG-1.4 was trained with a plain 0.5/1.0 normalisation, not ImageNet stats.
    if "rmbg" in model_id.lower():
        mean, std = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    x = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(tensor)
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


def subject_alpha(rgb: np.ndarray, model_id: str = HUMAN_PARSING_MODEL) -> np.ndarray:
    """Soft alpha matte of the person, float32 in [0, 1] at the input resolution."""
    if _is_human_parsing(model_id):
        return _parsing_alpha(rgb, model_id)
    try:
        return _matting_alpha(rgb, model_id)
    except Exception as exc:
        print(
            f"[avatar] '{model_id}' failed to load ({type(exc).__name__}: {exc}). "
            f"Falling back to '{HUMAN_PARSING_MODEL}'."
        )
        return _parsing_alpha(rgb, HUMAN_PARSING_MODEL)


def refine_edges(alpha: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Snap a coarse mask to the photo's real edges with GrabCut.

    Semantic segmentation gives labels at a reduced resolution, so the outline
    lands a few pixels off. That outline is what sets the body's reconstructed
    thickness, so it is worth sharpening - but GrabCut can also collapse on a
    low-contrast photo, so the result is discarded if the area moves sharply.
    """
    seed = np.full(alpha.shape, cv2.GC_PR_BGD, np.uint8)
    seed[alpha > 0.5] = cv2.GC_PR_FGD
    seed[alpha > 0.9] = cv2.GC_FGD
    seed[alpha < 0.1] = cv2.GC_BGD
    if not (seed == cv2.GC_FGD).any():
        return alpha

    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            seed,
            None,
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return alpha

    refined = ((seed == cv2.GC_FGD) | (seed == cv2.GC_PR_FGD)).astype(np.float32)
    before, after = float((alpha > 0.5).sum()), float(refined.sum())
    if before < 1 or not 0.75 < after / before < 1.25:
        return alpha
    return refined


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
