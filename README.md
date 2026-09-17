# generation_avatar

Turn a single full-body photo into a studio avatar, re-shot from a **different camera height**.

The person is reconstructed in 3D and re-rendered from a virtual camera you place yourself, then
finished with SDXL so the result reads as a photograph rather than a render.

## Why it is built this way

Most avatar pipelines hand the photo to a generative model and ask for a new view. That model has
to invent every pixel, and it invents them toward the average body in its training data - which is
how people quietly come out slimmer, taller and younger than they are.

Here the body comes from geometry instead:

1. **Segment** the subject (`briaai/RMBG-1.4`).
2. **Estimate metric depth** (`Depth-Anything-V2-Metric`), giving a real-scale surface.
3. **Reconstruct** a point cloud, and give it volume with a back shell whose thickness is derived
   from the subject's own silhouette width - so a heavy body stays heavy.
4. **Re-render** from a virtual camera at the height, distance and angle you choose.
5. **Refine** with SDXL under depth ControlNet at low denoising strength, which adds skin and
   fabric texture without being free to move the outline.

The proportions live in steps 1-4. Step 5 only gets to paint.

## Install

```bash
pip install -r requirements.txt   # plus torch, if you are not on Colab
```

## Colab

Open `notebooks/avatar_colab.ipynb`. It installs dependencies, takes an uploaded photo, offers a
fast geometry-only preview, then runs the full refinement. A T4 is enough.

## Command line

```bash
python -m avatar.cli photo.jpg -o output \
    --subject-height 1.72 \
    --source-lens 28 \
    --camera-height 1.15
```

Useful flags:

| flag | meaning |
|---|---|
| `--subject-height` | the person's real height in metres; sets the scale for everything else |
| `--source-lens` | 35mm-equivalent lens that took the photo (26-28 phone, 50-85 portrait) |
| `--camera-height` | height of the virtual camera; **below** the estimated source height looks up at the subject |
| `--camera-height-delta` | shift relative to the estimated source height instead |
| `--distance` | camera distance; **this** is what flattens or exaggerates perspective |
| `--orbit`, `--aim` | rotation around the subject, and the height the camera aims at |
| `--output-lens` | lens for the render, which only sets framing; omit to fit it to the subject |
| `--no-refine` | stop after the 3D render - seconds instead of minutes, for dialling in the camera |
| `--strength` | refinement denoising strength (default 0.30) |

Outputs land in `output/`: the mask, depth, a debug point cloud (`.ply`), the raw render, its depth
buffer, and `avatar.png`.

## Python

```python
from avatar import PipelineConfig, generate_avatar

cfg = PipelineConfig()
cfg.camera.subject_height_m = 1.72
cfg.camera.target_camera_height_m = 1.15
cfg.refine.enabled = False            # fast geometry preview

generate_avatar("photo.jpg", cfg, "output")
```

## Choosing the camera height

The pipeline prints the camera height it estimates for your source photo. Photos taken by another
person standing up land around 1.5-1.6m, which looks down at the subject and foreshortens the legs.
Dropping the virtual camera to roughly chest height (1.1-1.3m) gives the level, slightly upward
look that studio and fashion photography uses.

Large moves cost fidelity: the further the camera travels, the more of the body was never visible
in the original photo and has to be reconstructed. The pipeline reports that percentage and warns
past 45%.

## Limits

- **Single photo, single view.** The back of the subject is synthesised, not observed. Modest camera
  moves are reliable; walking the camera around to the side is not.
- **Depth quality caps geometry quality.** Loose clothing, motion blur and busy backgrounds degrade
  the reconstruction.
- **Feet should be in frame.** Ground level, and therefore camera height, is estimated from them.
- **`--strength` above ~0.45** lets SDXL start reshaping the body. If output looks plastic, raise
  steps or improve the photo instead.

## Optional: parametric body fitting

Steps 1-3 recover the visible surface. A SMPL-X fit (PIXIE, PyMAF-X) or a clothed-mesh
reconstruction (ECON, PIFuHD) would give a true back surface instead of the silhouette-derived
shell, at the cost of gated model downloads that are the usual thing to break on Colab. The
`Subject` produced in `avatar/geometry.py` is the seam to swap them in behind: anything that yields
points, colours and an up axis will render and refine unchanged.
