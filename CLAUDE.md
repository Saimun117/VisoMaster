# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VisoMaster is a PySide6 desktop application for AI face swapping and editing in images/videos, built on ONNX Runtime + PyTorch, with optional TensorRT acceleration on Nvidia GPUs.

## Commands

```sh
# Run the app (Windows users use Start.bat)
python main.py

# Install deps (conda env, Python 3.10.13)
pip install -r requirements_cu124.txt   # or requirements_cu118.txt; requirements_mac.txt on main branch

# Download all required ONNX models into model_assets/ (driven by app/processors/models_data.py)
python download_models.py

# Regenerate UI code after editing MainWindow.ui (Windows batch; runs pyside6-uic + pyside6-rcc)
app/ui/core/convert_ui_to_py.bat
```

There is no test suite or linter configured.

## Branches

- `main` includes macOS support (device auto-selection cuda → mps → cpu, `requirements_mac.txt`, io_binding falls back to cpu on MPS). ONNX Runtime io_binding does not support MPS.
- `dev` and `webversion` also exist; check the current branch before assuming mac-support code is present.

## Architecture

Everything hangs off `MainWindow` (`app/ui/main_ui.py`), which owns the two core engines and all shared state. Cross-references go through `main_window` — most modules take it as their first argument.

### Processing pipeline

- **`ModelsProcessor`** (`app/processors/models_processor.py`) — central owner of all ONNX/TensorRT inference sessions. Models are lazy-loaded by name and registered declaratively in `app/processors/models_data.py` (name, local path, sha256 hash, download URL). Delegates inference to category classes: `FaceDetectors`, `FaceLandmarkDetectors`, `FaceMasks`, `FaceRestorers`, `FaceSwappers`, `FrameEnhancers`, `FaceEditors` (LivePortrait), plus `DFMModel` for DeepFaceLab models. Holds `model_lock` (RLock) guarding model access across worker threads. Execution providers: TensorRT / CUDA / CPU (and torch device cuda/mps/cpu on main).
- **`VideoProcessor`** (`app/processors/video_processor.py`) — QObject managing playback/recording. Reads frames via cv2.VideoCapture into a bounded `frame_queue`, spawns one **`FrameWorker`** thread per frame, and emits processed frames back to the UI via Qt signals (`frame_processed_signal` etc.). Also handles recording via ffmpeg subprocess, audio preview via ffplay, and webcam streaming via pyvirtualcam.
- **`FrameWorker`** (`app/processors/workers/frame_worker.py`) — per-frame pipeline: applies per-frame marker parameters, then detect → landmarks → swap → restore → mask → enhance, using `main_window.parameters` and `main_window.target_faces`. Frames flow as RGB numpy/torch tensors; output converted to BGR for display.

### UI layer

- `app/ui/core/main_window.py` and `media_rc.py` are **generated code** (pyside6-uic/rcc from `MainWindow.ui` / `media.qrc`) — edit the `.ui` file and regenerate, don't hand-edit.
- Parameter widgets are **not** in the .ui file; they are built at runtime from declarative layout dicts: `common_layout_data.py`, `swapper_layout_data.py`, `settings_layout_data.py`, `face_editor_layout_data.py` in `app/ui/widgets/`. Each entry defines widget type/options/default/help and optional `parentSelection`/`requiredSelectionValue` for conditional visibility. Adding a new user-facing parameter means adding an entry here; it flows automatically into `main_window.parameters`.
- Event-handler logic lives in `app/ui/widgets/actions/` as module-level functions taking `main_window` (procedural style, e.g. `video_control_actions.py`, `card_actions.py`, `save_load_actions.py`) — not methods on the window class.
- Long-running UI tasks (media loading, thumbnail generation) run in `app/ui/widgets/ui_workers.py` QThread workers.

### Parameters and state

- `main_window.parameters` maps target face id → `ParametersDict` (defined in `app/helpers/miscellaneous.py`, wraps defaults from the layout dicts).
- **Markers** attach parameter snapshots to specific frame numbers; `FrameWorker` applies them under `model_lock` before processing each frame.
- Workspaces (loaded/saved via `save_load_actions.py`) persist media paths, target/source faces, embeddings, and parameters; `last_workspace.json` in repo root.

### Models

- `model_assets/` holds downloaded ONNX models; TensorRT engines are cached under `tensorrt-engines/` and in versioned `.trt` files.
- Model files are hash-verified on download (`app/helpers/downloader.py`).
- DFM (DeepFaceLive) models are user-supplied and discovered at runtime (`misc_helpers.get_dfm_models_*`).
