"""
Runtime setup for running from source or as a packaged macOS app.

The app reads styles and writes models, workspaces, thumbnails and temp files relative to its
working directory. From source that is the project folder. Inside VisoMaster.app the bundle is
read-only, so the working directory becomes ~/Library/Application Support/VisoMaster, with an
`app` link back to the bundled resources and the small tracked assets copied over. Models are
downloaded there on first use.
"""
import os
import shutil
import sys

APP_NAME = 'VisoMaster'


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def data_dir() -> str:
    return os.path.expanduser(f'~/Library/Application Support/{APP_NAME}')


def _link(target: str, link: str):
    if os.path.islink(link):
        if os.readlink(link) == target:
            return
        os.unlink(link)
    elif os.path.exists(link):
        return
    os.symlink(target, link)


def _ensure_ffmpeg(bin_dir: str):
    """Use the ffmpeg shipped with imageio-ffmpeg when none is on PATH (Finder-launched apps
    don't get the shell PATH, and the packaged app bundles its own)."""
    if shutil.which('ffmpeg') and not is_frozen():
        return
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return
    os.makedirs(bin_dir, exist_ok=True)
    _link(exe, os.path.join(bin_dir, 'ffmpeg'))
    os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')


def prepare(project_dir: str):
    """Set the working directory and environment before the app imports its modules."""
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    if not is_frozen():
        os.chdir(project_dir)
        if sys.platform == 'darwin':
            _ensure_ffmpeg(os.path.join(project_dir, '.bin'))
        return

    bundle = getattr(sys, '_MEIPASS', project_dir)
    data = data_dir()
    os.makedirs(os.path.join(data, 'model_assets', 'dfm_models'), exist_ok=True)
    os.makedirs(os.path.join(data, 'model_assets', 'liveportrait_onnx'), exist_ok=True)

    # small assets tracked in git (landmark mean shape, lip array) ship inside the bundle
    bundled_assets = os.path.join(bundle, 'model_assets')
    for root, _, files in os.walk(bundled_assets):
        for name in files:
            src = os.path.join(root, name)
            dst = os.path.join(data, 'model_assets', os.path.relpath(src, bundled_assets))
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    _link(os.path.join(bundle, 'app'), os.path.join(data, 'app'))
    for doc in ('README_MACOS.md', 'LICENSE'):
        if os.path.exists(os.path.join(bundle, doc)):
            _link(os.path.join(bundle, doc), os.path.join(data, doc))
    os.chdir(data)
    _ensure_ffmpeg(os.path.join(data, 'bin'))
