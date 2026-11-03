"""
Renders the same video frames with several swapper configurations and writes a side-by-side
contact sheet of the swapped face crops, plus timing per configuration.

    .venv/bin/python tools/compare_swappers.py out.jpg [workspace.json] [frame,frame,...]
"""
import os
import sys
import time
import queue
import copy

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PySide6 import QtWidgets

from app.ui import main_ui
from app.ui.widgets import widget_components
from app.ui.widgets.actions import save_load_actions
from app.processors.workers.frame_worker import FrameWorker

CONFIGS = [
    ('GhostFace-v3 (current)', {'SwapModelSelection': 'GhostFace-v3'}),
    ('GhostFace-v2', {'SwapModelSelection': 'GhostFace-v2'}),
    ('Inswapper128', {'SwapModelSelection': 'Inswapper128', 'SwapperResSelection': '128'}),
    ('Inswapper128 @256', {'SwapModelSelection': 'Inswapper128', 'SwapperResSelection': '256'}),
    ('InStyleSwapper256 A', {'SwapModelSelection': 'InStyleSwapper256 Version A'}),
    ('SimSwap512', {'SwapModelSelection': 'SimSwap512'}),
    ('CSCS', {'SwapModelSelection': 'CSCS'}),
    ('Inswapper @256 + GPEN-256', {'SwapModelSelection': 'Inswapper128', 'SwapperResSelection': '256',
                                    'FaceRestorerEnableToggle': True, 'FaceRestorerTypeSelection': 'GPEN-256',
                                    'FaceRestorerBlendSlider': 60}),
    ('HyperSwap 1A', {'SwapModelSelection': 'HyperSwap 1A'}),
    ('HyperSwap 1B', {'SwapModelSelection': 'HyperSwap 1B'}),
    ('HyperSwap 1C', {'SwapModelSelection': 'HyperSwap 1C'}),
    ('HyperSwap 1B + GPEN-256', {'SwapModelSelection': 'HyperSwap 1B', 'FaceRestorerEnableToggle': True,
                                 'FaceRestorerTypeSelection': 'GPEN-256', 'FaceRestorerBlendSlider': 60}),
    ('CSCS + GPEN-256', {'SwapModelSelection': 'CSCS', 'FaceRestorerEnableToggle': True,
                         'FaceRestorerTypeSelection': 'GPEN-256', 'FaceRestorerBlendSlider': 60}),
]


def score_identity(window, frames, results):
    """How much each result looks like the source photo, by three independent face recognizers
    (cosine similarity on a 0-100 scale, higher = more like the source)."""
    import torch
    mp = window.models_processor
    recognizers = ['Inswapper128ArcFace', 'GhostArcFace', 'CSCSArcFace']
    target = next(t for t in window.target_faces.values() if t.assigned_input_embedding)
    sources = {r: target.assigned_input_embedding.get(r) for r in recognizers}
    print(f"{'model':28s}" + ''.join(f'{r:>22s}' for r in recognizers) + '      mean')
    for name, outs in [('original video', [f for _, f in frames])] + list(results.items()):
        scores = {r: [] for r in recognizers}
        for out in outs:
            img = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1)
            _, kpss_5, _ = mp.run_detect(img, 'Yolov8', max_num=1, score=0.5)
            if len(kpss_5) == 0:
                continue
            for r in recognizers:
                emb, _ = mp.run_recognize_direct(img, kpss_5[0], 'Opal', r)
                scores[r].append(mp.findCosineDistance(emb, sources[r]))
        means = [np.mean(v) if v else float('nan') for v in scores.values()]
        print(f'{name:28s}' + ''.join(f'{m:22.1f}' for m in means) + f'{np.nanmean(means):10.1f}', flush=True)


def main():
    out = sys.argv[1]
    workspace = sys.argv[2] if len(sys.argv) > 2 else 'last_workspace.json'
    frame_numbers = [int(f) for f in sys.argv[3].split(',')] if len(sys.argv) > 3 else [700, 1300, 2300, 3400]
    only = os.environ.get('VISO_COMPARE_ONLY')

    app = QtWidgets.QApplication(sys.argv[:1])
    widget_components.LoadLastWorkspaceDialog.exec = lambda self: 0
    # never auto-save over the user's last_workspace.json when this tool exits
    save_load_actions.save_current_workspace = lambda *args, **kwargs: None
    window = main_ui.MainWindow()
    save_load_actions.load_saved_workspace(window, workspace)
    window.swapfacesButton.setChecked(True)
    window.control['DetectIntervalSlider'] = 1
    original_params = copy.deepcopy(dict(window.parameters))

    cap = cv2.VideoCapture(window.video_processor.media_path)
    frames = []
    for n in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, f = cap.read()
        if ok:
            frames.append((n, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))

    configs = [c for c in CONFIGS if not only or any(o in c[0] for o in only.split(','))]
    results = {}
    for name, overrides in configs:
        for face_id, params in original_params.items():
            p = window.parameters[face_id]
            for k, v in params.items():
                p[k] = v
            # a config without a restorer runs without one, whatever the workspace has
            p['FaceRestorerEnableToggle'] = False
            for k, v in overrides.items():
                p[k] = v
        outs, times = [], []
        for i, (n, frame) in enumerate(frames):
            w = FrameWorker(frame, window, n, queue.Queue(), is_single_frame=True)
            w.parameters = window.parameters.copy()
            t0 = time.time()
            outs.append(w.process_frame()[..., ::-1])  # BGR -> RGB
            if i:  # first frame includes model loading
                times.append(time.time() - t0)
        results[name] = outs
        print(f"{name:28s} {1000 * np.mean(times) if times else float('nan'):6.0f} ms/frame", flush=True)

    if os.environ.get('VISO_COMPARE_SCORE'):
        score_identity(window, frames, results)

    # crop each frame around where the swap changed pixels, same box for every config
    tiles_rows = []
    for i, (n, frame) in enumerate(frames):
        diff = np.zeros(frame.shape[:2], bool)
        for outs in results.values():
            o = outs[i]
            if o.shape == frame.shape:
                diff |= np.abs(o.astype(int) - frame.astype(int)).sum(2) > 40
        ys, xs = np.where(diff)
        if len(ys) < 100:
            continue
        cy, cx = int(np.median(ys)), int(np.median(xs))
        half = int(max(np.percentile(ys, 98) - np.percentile(ys, 2), np.percentile(xs, 98) - np.percentile(xs, 2)) * 0.6) + 20
        y0, y1 = max(0, cy - half), min(frame.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(frame.shape[1], cx + half)
        row = [('original', frame)] + [(name, outs[i]) for name, outs in results.items()]
        tiles = []
        for name, img in row:
            crop = cv2.resize(np.ascontiguousarray(img[y0:y1, x0:x1]), (300, 300), interpolation=cv2.INTER_AREA)
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.rectangle(crop, (0, 0), (300, 22), (0, 0, 0), -1)
            cv2.putText(crop, name if i == 0 else f'#{n}' if name == 'original' else '', (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        tiles_rows.append(np.hstack(tiles))
    if tiles_rows:
        cv2.imwrite(out, np.vstack(tiles_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('saved', out)


if __name__ == '__main__':
    main()
