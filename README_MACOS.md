# VisoMaster on macOS (Apple Silicon)

## Install

Download `VisoMaster-<version>-arm64.dmg` from the
[Releases](../../releases) page, or from the latest run of the *macOS app (DMG)* workflow
under Actions. Open it and drag **VisoMaster** to **Applications**. It needs an Apple Silicon
Mac (M1 or newer) running macOS 13 or later.

The app isn't notarized by Apple. The first time, **right-click VisoMaster → Open → Open**.
After that it opens normally.

Models aren't inside the DMG. Each one downloads on first use (an internet connection is
needed then) into `~/Library/Application Support/VisoMaster`. Your workspaces, thumbnails and
caches live there too. ffmpeg is bundled.

## Build

**DMG (standalone app):** on an Apple Silicon Mac with Python 3.12:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements_mac.txt pyinstaller==6.16.0
PYTHON=.venv/bin/python ./scripts/build_macos_dmg.sh   # -> dist/VisoMaster-<version>-arm64.dmg
```

**CI:** `.github/workflows/macos-dmg.yml` runs the same build on GitHub's Apple Silicon
runners for every push to `macos-support`. It then self-tests the packaged app: it downloads
a model, runs it on Core ML and checks ffmpeg. Each run uploads the DMG as an artifact. Pushing
a tag like `v0.1.6-mac` also attaches the DMG to a GitHub Release.

**Run from source** (development): `./scripts/build_macos_app.sh` creates `.venv` and a
`dist/VisoMaster.app` launcher that runs this folder. Or run `.venv/bin/python main.py`.

## Look and feel

The **macOS** theme (Settings → Appearance, default on Mac) uses the system font and accent
blue, and switches with macOS light/dark mode. It also gives:

- Standard menus: Settings… (⌘,), About, Window (Minimize ⌘M, Zoom, show/hide panels), and
  Help (this guide, keyboard shortcuts).
- Shortcuts: ⌘O open videos, ⌘I open source faces, ⌘S save workspace, ⌃⌘F full screen.
- The memory bar shows unified memory use. It turns orange above 85%.

## Workflow

1. **Add media.** Use the **+** button next to *Videos* or *Faces* (⌘O / ⌘I), or drop files
   onto the lists. Files are added to what's already there.
2. **Pick faces.** Select a video, click **Find Faces**, choose a target face, then click the
   source photo to use for it.
3. **Process Video** (⌘R) runs the swap over the whole video. The toolbar shows progress and
   the time remaining, and **Cancel** stops it.
4. When it's done, the window switches to **Result**, where you can play the finished video
   with sound. **Edit** goes back to the editor.
5. **Export…** (⌘E) saves the result wherever you choose. **Show in Finder** reveals it.

Processed videos wait in `~/Library/Caches/VisoMaster/processed` until you export them. For
a still image, use **Save Image**.

The window resizes freely. The sidebars share the width, and long setting names shorten
with "…"; hover over a setting to see its full name. Show or hide panels from the
**View** and **Window** menus.

## Defaults

The defaults balance quality against speed on Apple Silicon:

| Setting | Default | Why |
|---|---|---|
| Swapper | HyperSwap 1C | FaceFusion's 2025 swapper. In tests it kept expressions, mouth shapes and skin tone most natural, with the best resemblance score, at the same speed as CSCS (~47 ms per face). CSCS reproduces facial hair more strongly. HyperSwap models come from FaceFusion; check their license before commercial use |
| Detector | YOLOv8, every 2 frames | ~20 ms, and tracking skips about half the detections |
| Occlusion Mask | On | ~8 ms; keeps hands and microphones in front of the face |
| AutoColor Transfer | On | Almost free; matches the swapped face to the scene's lighting |
| Face Restorer | Off, type GPEN-256 | Sharpens eyes and teeth. Turning it on roughly doubles the processing time (GPEN-256 runs on CPU, ~120 ms per face) |
| Threads | 3 | More threads don't help, because the GPU is already fully used |

Workspaces you saved earlier keep their own settings. A setting's ↺ button resets it to
the default.

## Speed

**Settings → Providers Priority → CoreML** (the default on Mac) runs the models on the
Apple GPU through Core ML:

- The first time each model loads, an optimized copy is written to
  `model_assets/mac_optimized/` and Core ML compiles it. Expect the first run of a model to
  take 5–30 s. After that, loading is quick.
- If Core ML can't run a model, that model falls back to the CPU on its own (the log says
  `[CoreML] ... falling back to CPU`).
- The first time a model loads, its Core ML output is checked against the CPU output. Core ML
  gets some models wrong; GPEN, for example, comes out as a mesh pattern. Those models run
  on the CPU from then on. The results are stored in `model_assets/mac_optimized/coreml_verdicts.json`.
- Use 3–4 threads (Settings → Number of Threads). More than that doesn't help.
- **Settings → Detectors → Detect Every N Frames** (default 2): when playing or recording a
  video, face detection and recognition run only on every Nth frame. The frames in between
  move the faces with optical flow, which lands within ~2–3 px of the detector on average.
  On scene cuts, fast motion or blur the tracker gives up and that frame is detected
  normally. Set it to 1 to detect every frame.
- Videos are saved with the hardware H.264 encoder (VideoToolbox), so encoding doesn't
  take CPU away from processing.

Measured on an M3 with 8 GB, for a 1920×806 25 fps clip with one face per frame
(GhostFace-v3, YOLOv8 detector):

| | ms / frame | 3-min video (4,582 frames) |
|---|---|---|
| Before (CPU provider) | 557 | ~43 min |
| CoreML + fast warps, detect every frame | ~80 | ~6 min |
| + detect every 2 frames | 17–28% less | ~4.5–5 min |

These are best-case numbers on an idle, cool machine. Under memory pressure (heavy swap)
or after long sustained GPU use on a fanless MacBook Air, the same run can take twice as
long.

The heaviest per-frame costs are the swapper model, then the detector and the face
recognizer. Lighter swapper models, a smaller detector size, and turning off restorers or
the face editor all cut the time further. Background jobs that use the GPU or Neural
Engine, such as Photos' `mediaanalysisd`, slow processing down noticeably.

To measure throughput for your own workspace:

```bash
.venv/bin/python tools/benchmark_pipeline.py last_workspace.json 80 3 CoreML
```
