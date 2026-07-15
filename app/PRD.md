PRODUCT REQUIREMENTS DOCUMENT (PRD)
VisoMaster Web
Vision

Transform VisoMaster from a desktop-only application into a browser-based AI face editing platform that can run entirely on Google Colab's free GPU and be accessed from any device through a web browser.

Goals

Users should be able to:

Open a browser
Upload images/videos
Configure face swap settings
Click Generate
Watch progress
Download results

without installing Python, CUDA, or ONNX locally.

Success Metrics
Functional

✅ Image face swap

✅ Video face swap

✅ Face enhancement

✅ LivePortrait editing

✅ Workspace save/load

Non-functional

Processing occurs on GPU

No desktop UI

Runs in Google Colab

Accessible from browser

No installation required for users

User Flow
Open browser

↓

Upload source face

↓

Upload target image/video

↓

Adjust settings

↓

Generate

↓

Progress bar

↓

Preview

↓

Download
Scope Version 1

Include:

image swap
video swap
face enhancer
face editor
settings
download

Exclude:

webcam
ffplay preview
pyvirtualcam
desktop drag-drop
native menus
multi-window
Architecture
Browser

↓

Gradio

↓

API Layer

↓

VideoProcessor

↓

FrameWorker

↓

ModelsProcessor

↓

ONNX Runtime

Notice that everything below the API layer already exists.

Major Milestones
Phase 1

Understand repository

Goal:

Know where inference starts.

Output:

Architecture document.

Phase 2

Separate processing engine.

Goal:

No Qt dependency.

Output:

Engine callable from Python.

Phase 3

Create Python API.

Example

swap_image()

swap_video()

enhance_face()

edit_face()
Phase 4

Build Gradio UI.

Phase 5

Deploy to Colab.

Phase 6

Optimize.

Non-goals

Don't rewrite AI models.

Don't rewrite ONNX code.

Don't rewrite TensorRT.

Don't replace processing pipeline.