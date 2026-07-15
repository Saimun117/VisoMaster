"""VisoMaster Web — Gradio front-end over the headless engine API.

Run locally:   python webapp.py
On Colab:      launched from the deploy notebook with share=True.

Deliberately excludes desktop-only features (webcam, virtual camera, ffplay
preview, native menus) — see app/PRD.md scope.
"""
import argparse
import tempfile
import uuid
from pathlib import Path

import cv2
import gradio as gr

from app.engine import api, config

DISCLAIMER = """
### ⚠️ Responsible use
This tool creates synthetic face-swapped media. Only use faces you have the
right and consent to use. Do not create deceptive, defamatory, or non-consensual
content. You are solely responsible for what you generate.
"""

# A small, useful subset of settings surfaced in the UI. Defaults come straight
# from the layout data so they stay in sync with the desktop app.
_DEFAULTS = config.build_default_parameters()
_CTRL_DEFAULTS = config.build_default_control()
SWAP_MODELS = config.SWAPPER_LAYOUT_DATA['Swapper']['SwapModelSelection']['options']
DETECTORS = config.SETTINGS_LAYOUT_DATA['Detectors']['DetectorModelSelection']['options']


def _save_bgr(img_bgr, suffix='.png'):
    path = str(Path(tempfile.gettempdir()) / f"visomaster_{uuid.uuid4().hex[:8]}{suffix}")
    cv2.imwrite(path, img_bgr)
    return path


def _rgb_to_bgr(img_rgb):
    # Gradio hands us RGB numpy arrays; the API works in BGR.
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def ui_swap_image(source_rgb, target_rgb, swap_model, detector, similarity):
    if source_rgb is None or target_rgb is None:
        raise gr.Error("Please provide both a source face and a target image.")
    params = {'SwapModelSelection': swap_model, 'SimilarityThresholdSlider': int(similarity)}
    control = {'DetectorModelSelection': detector}
    out = api.swap_image(_rgb_to_bgr(target_rgb), _rgb_to_bgr(source_rgb),
                         parameters=params, control=control)
    return _bgr_to_rgb(out)


def ui_swap_video(source_rgb, target_video, swap_model, detector, similarity,
                  progress=gr.Progress()):
    if source_rgb is None or not target_video:
        raise gr.Error("Please provide a source face and a target video.")
    params = {'SwapModelSelection': swap_model, 'SimilarityThresholdSlider': int(similarity)}
    control = {'DetectorModelSelection': detector}

    def cb(done, total):
        progress((done, total or None), desc="Swapping frames")

    return api.swap_video(target_video, _rgb_to_bgr(source_rgb),
                          parameters=params, control=control, progress_cb=cb)


def ui_enhance(image_rgb, enhancer):
    if image_rgb is None:
        raise gr.Error("Please provide an image.")
    out = api.enhance_face(_rgb_to_bgr(image_rgb),
                           control={'FrameEnhancerTypeSelection': enhancer})
    return _bgr_to_rgb(out)


def build_ui():
    with gr.Blocks(title="VisoMaster Web") as demo:
        gr.Markdown("# VisoMaster Web")
        gr.Markdown(DISCLAIMER)

        with gr.Tab("Image Swap"):
            with gr.Row():
                src = gr.Image(label="Source face", type="numpy")
                tgt = gr.Image(label="Target image", type="numpy")
                out_img = gr.Image(label="Result", type="numpy")
            with gr.Row():
                swap_model = gr.Dropdown(SWAP_MODELS, value=_DEFAULTS['SwapModelSelection'], label="Swapper model")
                detector = gr.Dropdown(DETECTORS, value=_CTRL_DEFAULTS['DetectorModelSelection'], label="Detector")
                sim = gr.Slider(0, 100, value=_DEFAULTS['SimilarityThresholdSlider'], step=1, label="Similarity threshold")
            gr.Button("Swap", variant="primary").click(
                ui_swap_image, [src, tgt, swap_model, detector, sim], out_img)

        with gr.Tab("Video Swap"):
            with gr.Row():
                vsrc = gr.Image(label="Source face", type="numpy")
                vtgt = gr.Video(label="Target video")
                vout = gr.Video(label="Result")
            with gr.Row():
                vswap = gr.Dropdown(SWAP_MODELS, value=_DEFAULTS['SwapModelSelection'], label="Swapper model")
                vdet = gr.Dropdown(DETECTORS, value=_CTRL_DEFAULTS['DetectorModelSelection'], label="Detector")
                vsim = gr.Slider(0, 100, value=_DEFAULTS['SimilarityThresholdSlider'], step=1, label="Similarity threshold")
            gr.Button("Swap video", variant="primary").click(
                ui_swap_video, [vsrc, vtgt, vswap, vdet, vsim], vout)

        with gr.Tab("Enhance"):
            enhancers = config.SETTINGS_LAYOUT_DATA['Frame Enhancer']['FrameEnhancerTypeSelection']['options']
            with gr.Row():
                e_in = gr.Image(label="Input", type="numpy")
                e_out = gr.Image(label="Enhanced", type="numpy")
            e_type = gr.Dropdown(enhancers, value=_CTRL_DEFAULTS['FrameEnhancerTypeSelection'], label="Enhancer")
            gr.Button("Enhance", variant="primary").click(ui_enhance, [e_in, e_type], e_out)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true', help="Create a public Gradio share link (Colab).")
    parser.add_argument('--auth', default=None, help="user:password to protect the app.")
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    auth = tuple(args.auth.split(':', 1)) if args.auth else None
    demo = build_ui().queue()  # serialize requests: one GPU (see PRD P7)
    demo.launch(share=args.share, auth=auth, server_port=args.port)


if __name__ == '__main__':
    main()
