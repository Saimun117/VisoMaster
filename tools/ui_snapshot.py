"""
Renders the main window offscreen to a PNG, for checking UI changes without a display.

    .venv/bin/python tools/ui_snapshot.py out.png [light|dark] [workspace.json]
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PySide6 import QtWidgets, QtCore

import main as app_main
from app.ui.widgets import widget_components
from app.ui.widgets.actions import save_load_actions


def main():
    out = sys.argv[1]
    scheme = sys.argv[2] if len(sys.argv) > 2 else 'dark'
    workspace = sys.argv[3] if len(sys.argv) > 3 else None
    os.environ['VISO_FORCE_COLOR_SCHEME'] = scheme
    widget_components.LoadLastWorkspaceDialog.exec = lambda self: 0
    # never auto-save over the user's last_workspace.json when this tool exits
    save_load_actions.save_current_workspace = lambda *args, **kwargs: None
    app, window = app_main.create_app(sys.argv[:1])
    if workspace:
        save_load_actions.load_saved_workspace(window, workspace)
    width, height = (int(v) for v in os.environ.get('VISO_SNAPSHOT_SIZE', '1600x1000').split('x'))
    window.resize(width, height)
    window.show()
    # let switch animations (300 ms) finish
    deadline = QtCore.QDeadlineTimer(800)
    while not deadline.hasExpired():
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
    window.grab().save(out)
    print('saved', out)


if __name__ == '__main__':
    main()
