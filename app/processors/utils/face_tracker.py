"""
Detect-every-N-frames face tracking for video playback/recording.

Face detection and recognition run only on every Nth frame ("key frames"). The frames in
between reuse the key frame's faces and embeddings, and move their landmarks with sparse
optical flow (Lucas-Kanade), which costs ~1-2 ms instead of a detector + recognizer pass.

Frames are processed by several worker threads at once, so a follower frame waits for its
key frame's detection to finish (a few ms, detection is the first step of the pipeline).
Whenever tracking is unsafe (key frame not queued in this run, no faces, lost points, a
scene cut) the follower just runs the full detection itself.
"""
import threading

import cv2
import numpy as np

# Minimum fraction of points LK must keep to trust the track, and the max mean
# forward-backward error in pixels
_MIN_TRACKED_FRACTION = 0.8
_MAX_FB_ERROR = 1.5
_KEYFRAME_WAIT_S = 2.0
_LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class _KeyFrame:
    def __init__(self):
        self.ready = threading.Event()
        self.gray = None
        self.faces = None  # list of dicts: kps_5, kps_all, embedding, bbox


class FaceTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._queued = set()
        self._keyframes = {}

    def reset(self):
        """Call when a new playback/recording run starts or the video is seeked."""
        with self._lock:
            for kf in self._keyframes.values():
                kf.ready.set()
            self._queued.clear()
            self._keyframes.clear()

    def frame_queued(self, frame_number):
        """Called from the main thread, in order, for every frame handed to a worker."""
        with self._lock:
            self._queued.add(frame_number)

    def _keyframe(self, frame_number):
        with self._lock:
            kf = self._keyframes.get(frame_number)
            if kf is None:
                kf = self._keyframes[frame_number] = _KeyFrame()
            return kf

    @staticmethod
    def to_gray(img_chw):
        # img is the CxHxW uint8 RGB tensor the pipeline detects on
        return cv2.cvtColor(np.ascontiguousarray(img_chw.permute(1, 2, 0).cpu().numpy()), cv2.COLOR_RGB2GRAY)

    def key_frame_for(self, frame_number, interval):
        """Return the key frame number a follower frame should track from, or None if this
        frame must run its own detection."""
        if interval <= 1:
            return None
        key = frame_number - frame_number % interval
        if key == frame_number:
            return None
        with self._lock:
            if key not in self._queued:
                return None
        return key

    def store(self, frame_number, gray, faces):
        """Publish a key frame's detection results (or None if detection failed)."""
        kf = self._keyframe(frame_number)
        kf.gray, kf.faces = gray, faces
        kf.ready.set()
        with self._lock:
            # keep memory bounded: drop key frames well behind this one
            for old in [k for k in self._keyframes if k < frame_number - 64]:
                del self._keyframes[old]
            self._queued = {q for q in self._queued if q >= frame_number - 64}

    def release(self, frame_number):
        """Key frame worker failed before storing; unblock its followers."""
        kf = self._keyframe(frame_number)
        kf.ready.set()

    def track(self, key, gray):
        """Move the key frame's faces into this frame. Returns a list of face dicts, or
        None if the caller should run full detection instead."""
        kf = self._keyframe(key)
        if not kf.ready.wait(_KEYFRAME_WAIT_S) or not kf.faces or kf.gray is None:
            return None
        if kf.gray.shape != gray.shape:
            return None

        tracked = []
        for face in kf.faces:
            kps_5 = np.asarray(face['kps_5'], dtype=np.float32).reshape(-1, 2)
            kps_all = face['kps_all']
            has_all = kps_all is not None and len(kps_all) > 0
            pts = np.vstack([kps_5, np.asarray(kps_all, dtype=np.float32).reshape(-1, 2)]) if has_all else kps_5
            new_pts = self._flow(kf.gray, gray, pts)
            if new_pts is None:
                return None
            shift = (new_pts[:5] - kps_5).mean(axis=0)
            bbox = np.asarray(face['bbox'], dtype=np.float32).copy()
            bbox[:4] += np.tile(shift, 2)
            tracked.append({
                'kps_5': new_pts[:5].astype(kps_5.dtype),
                'kps_all': new_pts[5:].astype(np.asarray(kps_all).dtype).reshape(np.asarray(kps_all).shape) if has_all else kps_all,
                'embedding': face['embedding'],
                'bbox': bbox,
            })
        return tracked

    @staticmethod
    def _flow(prev_gray, gray, pts):
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **_LK_PARAMS)
        if p1 is None:
            return None
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **_LK_PARAMS)
        if p0r is None:
            return None
        fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < _MAX_FB_ERROR * 2)
        # all 5 key points must be good, plus most of the dense landmarks
        if not good[:5].all() or good.mean() < _MIN_TRACKED_FRACTION or fb_err[good].mean() > _MAX_FB_ERROR:
            return None
        p1 = p1.reshape(-1, 2)
        if not good.all():
            # dense landmarks that were lost follow the mean motion of the 5 key points
            shift = (p1[:5] - pts[:5]).mean(axis=0)
            p1[~good] = pts[~good] + shift
        return p1
