"""
onnxruntime session wrapper for Apple Silicon.

Runs models through the CoreML execution provider (GPU / Neural Engine) and transparently
falls back to a plain CPU session if CoreML can't compile or execute a particular model.
"""
import json
import os
import threading

import numpy as np

import onnxruntime

from app.helpers.mac_model_optimizer import optimized_model_path

VERDICTS_PATH = os.path.abspath(os.path.join('model_assets', 'mac_optimized', 'coreml_verdicts.json'))
# Mean output error relative to the output's mean magnitude above which Core ML's result is
# considered wrong. Correct models land around 0.1-1% (fp16 compute); broken conversions
# (e.g. GPEN) are 15%+.
_MAX_RELATIVE_ERROR = 0.05
_verdicts_lock = threading.Lock()


def coreml_providers(compute_units='ALL'):
    # No ModelCacheDirectory: onnxruntime's cache keeps an uncompressed copy of every model
    # (12+ GB), and macOS already caches the compiled Core ML programs itself.
    return [
        ('CoreMLExecutionProvider', {
            'ModelFormat': 'MLProgram',
            'MLComputeUnits': compute_units,
            'AllowLowPrecisionAccumulationOnGPU': '1',
        }),
        'CPUExecutionProvider',
    ]


def create_session(model_path, providers, sess_options=None):
    """InferenceSession that survives onnxruntime graph-fusion bugs: some stock models
    (e.g. CodeFormer) crash ORT's layer-norm fusion at load time, so on failure the session
    is retried with only basic graph optimizations."""
    opts = sess_options or onnxruntime.SessionOptions()
    try:
        return onnxruntime.InferenceSession(model_path, sess_options=opts, providers=providers)
    except Exception:
        if opts.graph_optimization_level == onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC:
            raise
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return onnxruntime.InferenceSession(model_path, sess_options=opts, providers=providers)


class _RecordingIOBinding:
    """Forwards bind calls to the real IOBinding and remembers them so they can be
    replayed on a CPU session if the CoreML session has to be replaced."""

    def __init__(self, owner: 'CoreMLSession'):
        self._owner = owner
        self._calls = []
        self._session = owner._session
        self._binding = self._session.io_binding()

    def _rebind(self):
        self._session = self._owner._session
        self._binding = self._session.io_binding()
        for name, args, kwargs in self._calls:
            getattr(self._binding, name)(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._binding, name)
        if not callable(attr) or not name.startswith(('bind_', 'clear_', 'synchronize')):
            return attr

        def call(*args, **kwargs):
            if name.startswith('bind_'):
                self._calls.append((name, args, kwargs))
            elif name.startswith('clear_binding_'):
                kind = 'bind_input' if name.endswith('inputs') else 'bind_output'
                self._calls = [c for c in self._calls if not c[0].startswith(kind)]
            return attr(*args, **kwargs)
        return call


def _load_verdicts():
    try:
        with open(VERDICTS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_verdict(key, ok):
    with _verdicts_lock:
        verdicts = _load_verdicts()
        verdicts[key] = ok
        os.makedirs(os.path.dirname(VERDICTS_PATH), exist_ok=True)
        with open(VERDICTS_PATH, 'w') as f:
            json.dump(verdicts, f, indent=1, sort_keys=True)


def _test_inputs(session):
    """Deterministic, image-like inputs: smooth patterns in [-1, 1] for image tensors and
    small values for vectors. Symbolic dims become 1 (batch) or 256 (spatial)."""
    rng = np.random.default_rng(0)
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 256) for i, d in enumerate(inp.shape)]
        if 'double' in inp.type:
            feed[inp.name] = np.full(shape, 0.5, dtype=np.float64)
            continue
        if 'float' not in inp.type:
            return None
        if len(shape) == 4 and shape[2] >= 16 and shape[3] >= 16:
            h, w = shape[2], shape[3]
            yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
            planes = [np.sin(6 * xx + c) * np.cos(5 * yy - c) * 0.8 for c in range(shape[1])]
            arr = np.stack(planes)[None].repeat(shape[0], 0)
        else:
            arr = rng.standard_normal(shape) * 0.1
        feed[inp.name] = arr.astype(np.float16 if 'float16' in inp.type else np.float32)
    return feed


def coreml_matches_cpu(model_path, coreml_session, sess_options=None):
    """Compare a Core ML session against onnxruntime's CPU kernels on the same graph.
    Returns True/False, or None if the model couldn't be tested (unknown input shapes)."""
    feed = _test_inputs(coreml_session)
    if feed is None:
        return None
    try:
        cpu = create_session(optimized_model_path(model_path), ['CPUExecutionProvider'], sess_options)
        expected = cpu.run(None, feed)
        actual = coreml_session.run(None, feed)
    except Exception:
        return None
    for e, a in zip(expected, actual):
        e = np.asarray(e, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        if e.shape != a.shape or not np.isfinite(a).all():
            return False
        scale = np.abs(e).mean() + 1e-6
        if np.abs(e - a).mean() / scale > _MAX_RELATIVE_ERROR:
            return False
    return True


class CoreMLSession:
    def __init__(self, model_path, sess_options=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self._lock = threading.Lock()
        self.on_cpu = False
        verdict_key = f"{os.path.basename(model_path)}:{int(os.path.getmtime(model_path))}"
        verdict = _load_verdicts().get(verdict_key)
        if verdict is False:
            self._fallback_to_cpu('Core ML gives wrong results for this model', quiet=True)
            return
        try:
            self._session = create_session(optimized_model_path(model_path), coreml_providers(), sess_options)
        except Exception as e:
            self._fallback_to_cpu(f'session creation failed ({str(e)[:120]})')
            return
        if verdict is None:
            ok = coreml_matches_cpu(model_path, self._session, sess_options)
            if ok is not None:
                _save_verdict(verdict_key, ok)
            if ok is False:
                self._fallback_to_cpu('Core ML output differs from the reference')

    def _fallback_to_cpu(self, reason, quiet=False):
        with self._lock:
            if self.on_cpu:
                return
            if not quiet:
                print(f"[CoreML] {os.path.basename(self.model_path)}: {reason}, falling back to CPU")
            try:
                self._session = create_session(optimized_model_path(self.model_path), ['CPUExecutionProvider'], self.sess_options)
            except Exception:
                # the optimized copy itself is broken: use the stock model
                self._session = create_session(self.model_path, ['CPUExecutionProvider'], self.sess_options)
            self.on_cpu = True

    def io_binding(self):
        return _RecordingIOBinding(self)

    def run_with_iobinding(self, binding, run_options=None):
        if binding._session is not self._session:
            binding._rebind()
        try:
            return self._session.run_with_iobinding(binding._binding, run_options)
        except Exception as e:
            if self.on_cpu:
                raise
            self._fallback_to_cpu(f'CoreML execution failed ({str(e)[:120]})')
            binding._rebind()
            return self._session.run_with_iobinding(binding._binding, run_options)

    def run(self, output_names, input_feed, run_options=None):
        try:
            return self._session.run(output_names, input_feed, run_options)
        except Exception as e:
            if self.on_cpu:
                raise
            self._fallback_to_cpu(f'CoreML execution failed ({str(e)[:120]})')
            return self._session.run(output_names, input_feed, run_options)

    def __getattr__(self, name):
        return getattr(self._session, name)
