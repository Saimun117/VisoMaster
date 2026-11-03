import os
import sys

from app.helpers import mac_runtime
# Paths to models, styles and workspaces are relative to the working directory: the project
# folder from source, ~/Library/Application Support/VisoMaster in the packaged Mac app
mac_runtime.prepare(os.path.dirname(os.path.abspath(__file__)))

from app.ui import main_ui
from PySide6 import QtWidgets, QtGui

import qdarktheme
from app.ui.core.proxy_style import ProxyStyle
from app.ui import macos_theme
from app.ui import mac_workflow


def create_app(argv):
    app = QtWidgets.QApplication(argv)
    app.setApplicationName('VisoMaster')
    app.setApplicationDisplayName('VisoMaster')
    app.setWindowIcon(QtGui.QIcon('app/ui/core/media/visomaster_small.png'))
    app.setStyle(ProxyStyle(QtWidgets.QStyleFactory.create('Fusion')))
    if macos_theme.IS_MACOS:
        macos_theme.apply(app)
    else:
        with open("app/ui/styles/dark_styles.qss", "r") as f:
            _style = f.read()
            _style = qdarktheme.load_stylesheet('dark')+'\n'+_style
            app.setStyleSheet(_style)
    window = main_ui.MainWindow()
    if macos_theme.IS_MACOS:
        macos_theme.setup_window(window)
        mac_workflow.setup(window)
        macos_theme.follow_system(app, lambda: window.control.get('ThemeSelection') == 'macOS')
    return app, window


def self_test(window) -> int:
    """Used by CI on the packaged app: first-use model download, Core ML inference, bundled ffmpeg."""
    import shutil
    import subprocess
    import numpy as np
    mp = window.models_processor
    print('provider', mp.provider_name, 'cwd', os.getcwd())
    session = mp.load_model('YoloFace8n')
    out = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 640, 640), np.float32)})
    print('YoloFace8n output', out[0].shape, 'on CPU' if getattr(session, 'on_cpu', False) else 'on Core ML')
    ffmpeg = shutil.which('ffmpeg')
    encoders = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    print('ffmpeg', ffmpeg, 'videotoolbox' if 'h264_videotoolbox' in encoders else 'NO videotoolbox')
    return 0 if 'h264_videotoolbox' in encoders else 1


if __name__=="__main__":
    # packaged app: child processes must not start a second copy of the UI
    import multiprocessing
    multiprocessing.freeze_support()
    if os.environ.get('VISOMASTER_SELFTEST'):
        from app.ui.widgets import widget_components
        widget_components.LoadLastWorkspaceDialog.exec = lambda self: 0
    app, window = create_app(sys.argv)
    if os.environ.get('VISOMASTER_SELFTEST'):
        code = self_test(window)
        sys.stdout.flush()
        os._exit(code)  # skip interpreter teardown with live Qt/Core ML objects
    window.show()
    app.exec()
