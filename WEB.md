# VisoMaster Web

Browser front-end + headless Python API over the existing VisoMaster engine.
No desktop UI, no Qt window — runs on a GPU server (Google Colab + Drive).

## Layout

| Path | Role |
|------|------|
| `app/engine/config.py` | Builds `parameters` / `control` dicts from the layout data with no widgets |
| `app/engine/face_data.py` | `TargetFace` — Qt-free face/embedding container |
| `app/engine/frame_engine.py` | `FrameEngine` — all frame compute, extracted from `FrameWorker` |
| `app/engine/api.py` | `swap_image`, `swap_video`, `enhance_face`, `edit_face` |
| `webapp.py` | Gradio UI |
| `requirements_colab.txt` | Server deps (adds gradio, drops pyvirtualcam; keeps TensorRT + PySide6) |
| `VisoMaster_Colab.ipynb` | Mount Drive → install → download models → launch |

The desktop app is unchanged: `FrameWorker` now subclasses `FrameEngine` and
`ModelsProcessor` runs with or without a `main_window`.

## Python API

```python
from app.engine import api
import cv2

# Image (BGR numpy arrays or file paths in; BGR array out)
out = api.swap_image("target.jpg", "source.jpg",
                     parameters={"SwapModelSelection": "Inswapper128",
                                 "SimilarityThresholdSlider": 60},
                     control={"DetectorModelSelection": "RetinaFace"},
                     max_side=1920)
cv2.imwrite("out.png", out)

# Video (batch render-to-file; original audio muxed back)
path = api.swap_video("clip.mp4", "source.jpg",
                      progress_cb=lambda done, total: print(done, total),
                      max_side=1280)

api.enhance_face("img.jpg", control={"FrameEnhancerTypeSelection": "RealEsr-General-x4v3"})
api.edit_face("img.jpg", parameters={"MouthSmileDecimalSlider": 0.3})
```

`parameters` keys are face-parameter widget names (swapper/common/face-editor
layouts); `control` keys are settings widget names. Anything omitted falls back
to the layout defaults.

## Persistence (Colab + Drive)

Two env vars redirect the two things worth persisting to a Drive mount so they
survive session restarts:

- `VISOMASTER_MODELS_DIR` → `model_assets/` (the ~GB of ONNX models)
- `VISOMASTER_TRT_CACHE` → built TensorRT engines

The notebook sets both to `MyDrive/VisoMaster/…`.

## Concurrency & safety

- One Colab = one GPU. The Gradio app calls `.queue()` so requests serialize;
  this is **not** a multi-user throughput design.
- A bare `--share` link is an open face-swap tool. Use `--auth user:pass`, and
  note deepfake workloads may violate Colab's terms.

## Limits / metrics

The engine floors every frame to 512px for detection, so `max_side` only reduces
work for inputs **larger** than 512. Suggested caps on a free T4: images ≤ 1920,
video ≤ 1280.

Fill in on a real T4 run (the numbers depend on model + resolution):

| Metric | Target | Measured |
|--------|--------|----------|
| Cold start (models load from Drive, no TRT rebuild) | < 60 s | _tbd_ |
| Image swap @ 1080p, Inswapper128 | < 5 s | _tbd_ |
| Video swap throughput @ 720p | ≥ 5 fps | _tbd_ |
| TRT engine reuse on 2nd session (no rebuild) | yes | _tbd_ |
