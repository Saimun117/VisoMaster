"""Headless Python API for VisoMaster.

Four entry points mirroring the desktop features, all Qt-free:
    swap_image, swap_video, enhance_face, edit_face

Images are BGR numpy arrays (OpenCV convention) or file paths; videos are file
paths. The heavy lifting is delegated to FrameEngine (app/engine/frame_engine.py)
and ModelsProcessor (app/processors/models_processor.py), both of which now run
without a Qt main window.
"""
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from app.processors.models_processor import ModelsProcessor
from app.engine import config
from app.engine.face_data import TargetFace
from app.engine.frame_engine import FrameEngine

ImageLike = Union[str, np.ndarray]

_MODELS_CACHE: Dict[str, ModelsProcessor] = {}


def pick_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_models_processor(device: Optional[str] = None) -> ModelsProcessor:
    """Cached ModelsProcessor per device (model sessions are expensive to build)."""
    device = pick_device(device)
    if device not in _MODELS_CACHE:
        _MODELS_CACHE[device] = ModelsProcessor(main_window=None, device=device)
    return _MODELS_CACHE[device]


# --- image helpers -------------------------------------------------------

def _read_bgr(image: ImageLike) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img


def _bgr_to_rgb_tensor(frame_bgr: np.ndarray, device: str) -> torch.Tensor:
    """HxWxC BGR uint8 -> CxHxW RGB tensor on device (matches the detect path)."""
    frame_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    img = torch.from_numpy(frame_rgb.astype('uint8')).to(device)
    return img.permute(2, 0, 1)


def _maybe_downscale(frame_bgr: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """Downscale so max(H, W) <= max_side. Bounds compute time on large inputs
    (mainly video on a Colab T4). No-op when max_side is None or already small."""
    if not max_side:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr
    scale = max_side / longest
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# --- embedding helpers ---------------------------------------------------

def _detect_faces(mp: ModelsProcessor, img: torch.Tensor, control: dict):
    """Return kpss_5 for every detected face (same call the desktop app uses)."""
    _, kpss_5, _ = mp.run_detect(
        img,
        control['DetectorModelSelection'],
        max_num=control['MaxFacesToDetectSlider'],
        score=control['DetectorScoreSlider'] / 100.0,
        input_size=(512, 512),
        use_landmark_detection=control['LandmarkDetectToggle'],
        landmark_detect_mode=control['LandmarkDetectModelSelection'],
        landmark_score=control['LandmarkDetectScoreSlider'] / 100.0,
        from_points=control['DetectFromPointsToggle'],
        rotation_angles=[0] if not control['AutoRotationToggle'] else [0, 90, 180, 270],
    )
    return kpss_5


def _embedding_store(mp: ModelsProcessor, img: torch.Tensor, kps_5, control: dict) -> Dict[str, np.ndarray]:
    """Embeddings keyed by every recognition/arcface model, as the UI builds them."""
    store: Dict[str, np.ndarray] = {}
    for option in config.RECOGNITION_OPTIONS:
        emb, _ = mp.run_recognize_direct(img, kps_5, control['SimilarityTypeSelection'], option)
        store[option] = emb
    return store


def _source_embedding(mp: ModelsProcessor, source_images: Sequence[ImageLike], control: dict) -> Dict[str, np.ndarray]:
    """Merged source embedding (keyed by arcface model) from one or more source faces."""
    holder = TargetFace('source')
    for idx, src in enumerate(source_images):
        img = _bgr_to_rgb_tensor(_read_bgr(src), mp.device)
        kpss_5 = _detect_faces(mp, img, control)
        if len(kpss_5) == 0:
            continue
        holder.assigned_input_faces[f'src{idx}'] = _embedding_store(mp, img, kpss_5[0], control)
    holder.calculate_assigned_input_embedding(control)
    if not holder.assigned_input_embedding:
        raise ValueError("No face detected in any source image.")
    return holder.assigned_input_embedding


# --- core runner ---------------------------------------------------------

def _make_engine(mp: ModelsProcessor, target_faces: Dict[str, TargetFace],
                 parameters: Dict[str, object], control: dict) -> FrameEngine:
    return FrameEngine(
        models_processor=mp,
        target_faces=target_faces,
        parameters=parameters,
        default_parameters=config.build_default_parameters(),
        control=control,
    )


def _build_targets_from_frame(mp: ModelsProcessor, img: torch.Tensor, control: dict,
                              source_embedding: Optional[Dict[str, np.ndarray]],
                              param_overrides: Optional[dict]):
    """Detect faces in a frame and turn each into a TargetFace + parameter set."""
    target_faces: Dict[str, TargetFace] = {}
    parameters: Dict[str, object] = {}
    for kps_5 in _detect_faces(mp, img, control):
        face_id = str(uuid.uuid1().int)
        tf = TargetFace(face_id, embedding_store=_embedding_store(mp, img, kps_5, control))
        if source_embedding is not None:
            tf.assigned_input_embedding = dict(source_embedding)
        target_faces[face_id] = tf
        parameters[face_id] = config.new_parameters(param_overrides)
    return target_faces, parameters


def swap_image(target_image: ImageLike,
               source_images: Union[ImageLike, Sequence[ImageLike]],
               parameters: Optional[dict] = None,
               control: Optional[dict] = None,
               device: Optional[str] = None,
               max_side: Optional[int] = None) -> np.ndarray:
    """Swap the source face onto every detected face in target_image. Returns BGR.

    max_side optionally caps the longest edge of the target to bound compute time."""
    mp = get_models_processor(device)
    ctrl = config.new_control(control)
    ctrl['SwapFacesEnabled'] = True
    ctrl['EditFacesEnabled'] = False

    if isinstance(source_images, (str, np.ndarray)):
        source_images = [source_images]
    source_embedding = _source_embedding(mp, source_images, ctrl)

    frame_bgr = _maybe_downscale(_read_bgr(target_image), max_side)
    img = _bgr_to_rgb_tensor(frame_bgr, mp.device)
    target_faces, params = _build_targets_from_frame(mp, img, ctrl, source_embedding, parameters)

    engine = _make_engine(mp, target_faces, params, ctrl)
    return engine.process_frame(np.ascontiguousarray(frame_bgr[..., ::-1]))  # RGB in, BGR out


def enhance_face(image: ImageLike,
                 parameters: Optional[dict] = None,
                 control: Optional[dict] = None,
                 device: Optional[str] = None) -> np.ndarray:
    """Run the frame enhancer (and any restorer params) with no face swap. Returns BGR."""
    mp = get_models_processor(device)
    ctrl = config.new_control(control)
    ctrl['SwapFacesEnabled'] = False
    ctrl['EditFacesEnabled'] = False
    ctrl['FrameEnhancerEnableToggle'] = True

    frame_bgr = _read_bgr(image)
    engine = _make_engine(mp, {}, {}, ctrl)
    return engine.process_frame(np.ascontiguousarray(frame_bgr[..., ::-1]))


def edit_face(image: ImageLike,
              parameters: Optional[dict] = None,
              control: Optional[dict] = None,
              device: Optional[str] = None) -> np.ndarray:
    """Apply LivePortrait face-editor parameters to every detected face. Returns BGR."""
    mp = get_models_processor(device)
    ctrl = config.new_control(control)
    ctrl['SwapFacesEnabled'] = False
    ctrl['EditFacesEnabled'] = True

    overrides = dict(parameters or {})
    overrides.setdefault('FaceEditorEnableToggle', True)

    frame_bgr = _read_bgr(image)
    img = _bgr_to_rgb_tensor(frame_bgr, mp.device)
    target_faces, params = _build_targets_from_frame(mp, img, ctrl, None, overrides)

    engine = _make_engine(mp, target_faces, params, ctrl)
    return engine.process_frame(np.ascontiguousarray(frame_bgr[..., ::-1]))


def swap_video(target_video: str,
               source_images: Union[ImageLike, Sequence[ImageLike]],
               output_path: Optional[str] = None,
               parameters: Optional[dict] = None,
               control: Optional[dict] = None,
               device: Optional[str] = None,
               progress_cb: Optional[Callable[[int, int], None]] = None,
               mux_audio: bool = True,
               max_side: Optional[int] = None) -> str:
    """Batch render-to-file: swap the source face across every frame.

    Detects target faces once on the first frame, then reuses them for the whole
    clip. Processed BGR frames are piped to ffmpeg (libx264); original audio is
    muxed back afterwards. Returns the output file path."""
    mp = get_models_processor(device)
    ctrl = config.new_control(control)
    ctrl['SwapFacesEnabled'] = True
    ctrl['EditFacesEnabled'] = False

    if isinstance(source_images, (str, np.ndarray)):
        source_images = [source_images]
    source_embedding = _source_embedding(mp, source_images, ctrl)

    cap = cv2.VideoCapture(str(target_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {target_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if output_path is None:
        stem = Path(target_video).stem
        output_path = str(Path(tempfile.gettempdir()) / f"{stem}_swapped_{uuid.uuid4().hex[:8]}.mp4")

    tmp_dir = tempfile.mkdtemp(prefix='visomaster_')
    silent_path = os.path.join(tmp_dir, 'video_noaudio.mp4')

    target_faces: Dict[str, TargetFace] = {}
    params: Dict[str, object] = {}
    engine: Optional[FrameEngine] = None
    writer_proc = None
    index = 0
    try:
        ret, frame_bgr = cap.read()
        while ret:
            frame_bgr = _maybe_downscale(frame_bgr, max_side)
            img = _bgr_to_rgb_tensor(frame_bgr, mp.device)
            if engine is None:
                # Establish target faces + engine once, on the first frame.
                target_faces, params = _build_targets_from_frame(mp, img, ctrl, source_embedding, parameters)
                engine = _make_engine(mp, target_faces, params, ctrl)

            out_bgr = engine.process_frame(np.ascontiguousarray(frame_bgr[..., ::-1]))
            out_bgr = np.ascontiguousarray(out_bgr).astype(np.uint8)

            if writer_proc is None:
                h, w = out_bgr.shape[:2]
                writer_proc = subprocess.Popen(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",
                        "-c:v", "libx264", "-crf", "18",
                        silent_path,
                    ],
                    stdin=subprocess.PIPE,
                )
            writer_proc.stdin.write(out_bgr.tobytes())

            index += 1
            if progress_cb:
                progress_cb(index, total)
            ret, frame_bgr = cap.read()
    finally:
        cap.release()
        if writer_proc is not None:
            writer_proc.stdin.close()
            writer_proc.wait()

    if index == 0:
        raise ValueError("No frames read from the input video.")

    if mux_audio:
        if Path(output_path).is_file():
            os.remove(output_path)
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", silent_path, "-i", str(target_video),
                "-c", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-shortest",
                output_path,
            ],
            check=False,
        )
        if result.returncode != 0 or not Path(output_path).is_file():
            # No audio track (or mux failed): fall back to the silent render.
            os.replace(silent_path, output_path)
    else:
        os.replace(silent_path, output_path)

    return output_path
