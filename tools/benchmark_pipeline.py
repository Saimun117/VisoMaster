"""
Headless throughput benchmark: loads a saved workspace and runs the full swap pipeline
on N frames of its selected video, printing per-frame timings.

    .venv/bin/python tools/benchmark_pipeline.py [workspace.json] [num_frames] [threads] [provider]
"""
import os
import sys
import time
import threading
import queue

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from PySide6 import QtWidgets

from app.ui import main_ui
from app.ui.widgets import widget_components
from app.ui.widgets.actions import save_load_actions
from app.processors.workers.frame_worker import FrameWorker


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else 'last_workspace.json'
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    app = QtWidgets.QApplication(sys.argv)
    # skip the "load last workspace?" modal, the workspace is loaded explicitly below
    widget_components.LoadLastWorkspaceDialog.exec = lambda self: 0
    # never auto-save over the user's last_workspace.json when this tool exits
    save_load_actions.save_current_workspace = lambda *args, **kwargs: None
    window = main_ui.MainWindow()
    save_load_actions.load_saved_workspace(window, workspace)
    window.swapfacesButton.setChecked(True)
    mp = window.models_processor
    if len(sys.argv) > 4:
        from app.ui.widgets.actions import control_actions
        window.control['ProvidersPrioritySelection'] = sys.argv[4]
        control_actions.change_execution_provider(window, sys.argv[4])
    print(f"provider={mp.provider_name} device={mp.device} media={window.video_processor.media_path}")

    if os.environ.get('VISO_BENCH_SWAPPER'):
        for face_id in window.parameters:
            window.parameters[face_id]['SwapModelSelection'] = os.environ['VISO_BENCH_SWAPPER']
    cap = cv2.VideoCapture(window.video_processor.media_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    frames = []
    for _ in range(num_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    if os.environ.get('VISO_DETECT_INTERVAL'):
        window.control['DetectIntervalSlider'] = int(os.environ['VISO_DETECT_INTERVAL'])
    window.video_processor.file_type = 'video'
    tracker = window.video_processor.face_tracker
    tracker.reset()
    for i in range(len(frames)):
        tracker.frame_queued(1000 + i)
    print(f"detect interval={window.control.get('DetectIntervalSlider')}")
    track_results = []
    original_track = tracker.track
    def counting_track(key, gray):
        result = original_track(key, gray)
        track_results.append(result is not None)
        return result
    tracker.track = counting_track

    def run_one(i, frame):
        w = FrameWorker(frame, window, 1000 + i, queue.Queue())
        w.parameters = window.parameters.copy()
        return w.process_frame()

    # warm-up (model loading / CoreML compilation)
    t0 = time.time()
    first = run_one(0, frames[0])
    print(f"warm-up {time.time() - t0:.1f}s")
    if os.environ.get('VISO_BENCH_SAVE'):
        # process_frame returns BGR
        cv2.imwrite(os.environ['VISO_BENCH_SAVE'], first)
        cv2.imwrite(os.environ['VISO_BENCH_SAVE'].replace('.png', '_orig.png'), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))

    work = queue.Queue()
    for i, f in enumerate(frames):
        work.put((i, f))

    def loop():
        while True:
            try:
                i, f = work.get_nowait()
            except queue.Empty:
                return
            run_one(i, f)

    t0 = time.time()
    pool = [threading.Thread(target=loop) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    dt = time.time() - t0
    fps = len(frames) / dt
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print()
    print(f"{len(frames)} frames in {dt:.1f}s -> {fps:.2f} fps, {1000/fps:.0f} ms/frame; "
          f"full video ({total} frames) ~ {total / fps / 60:.1f} min")
    if track_results:
        print(f"tracked frames: {sum(track_results)}/{len(track_results)} (rest fell back to detection)")


if __name__ == '__main__':
    main()
