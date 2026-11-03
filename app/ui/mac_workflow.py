"""
macOS-style workflow for the main window:

- Media and face pickers with a section header and a "+" menu that opens the standard macOS
  open panel, adding to the list instead of replacing it, with file names on video thumbnails.
- A window that resizes freely: no hard minimum widths from the Designer file.
- A toolbar with Process Video -> Result preview -> Export…, replacing the record button:
  the whole video is processed first, the result plays in a native player with sound, and
  Export saves it wherever the user picks.
"""
import os
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets import ui_workers, widget_components
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import list_view_actions, video_control_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

_VIDEO_GLOBS = ' '.join(f'*{ext}' for ext in misc_helpers.video_extensions)
_IMAGE_GLOBS = ' '.join(f'*{ext}' for ext in misc_helpers.image_extensions)
MEDIA_FILTER = f'Videos and Images ({_VIDEO_GLOBS} {_IMAGE_GLOBS});;Videos ({_VIDEO_GLOBS});;Images ({_IMAGE_GLOBS})'
IMAGE_FILTER = f'Images ({_IMAGE_GLOBS})'


# --------------------------------------------------------------------------------------------
# Pickers
# --------------------------------------------------------------------------------------------

def _start_dir(path):
    return path if path and os.path.isdir(path) else os.path.expanduser('~/Movies')


def add_target_media(window: 'MainWindow', files):
    existing = {b.media_path for b in window.target_videos.values()}
    files = [f for f in files if (misc_helpers.is_video_file(f) or misc_helpers.is_image_file(f)) and f not in existing]
    if not files:
        return
    window.last_target_media_folder_path = os.path.dirname(files[0])
    worker = ui_workers.TargetMediaLoaderWorker(main_window=window, folder_name=False, files_list=files)
    worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_target_videos_list, window))
    worker.finished.connect(partial(_update_counts, window))
    window._mac_loaders.append(worker)
    worker.start()


def add_input_faces(window: 'MainWindow', files):
    existing = {b.media_path for b in window.input_faces.values()}
    files = [f for f in files if misc_helpers.is_image_file(f) and f not in existing]
    if not files:
        return
    window.last_input_media_folder_path = os.path.dirname(files[0])
    worker = ui_workers.InputFacesLoaderWorker(main_window=window, folder_name=False, files_list=files)
    worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_source_faces_list, window))
    worker.finished.connect(partial(_update_counts, window))
    window._mac_loaders.append(worker)
    worker.start()


def _files_in_folder(folder, images_only=False):
    files = misc_helpers.get_image_files(folder)
    if not images_only:
        files = misc_helpers.get_video_files(folder) + files
    return [f if os.path.isabs(f) else os.path.join(folder, f) for f in files]


def choose_media_files(window: 'MainWindow'):
    files, _ = QtWidgets.QFileDialog.getOpenFileNames(window, 'Add Videos and Images',
                                                      _start_dir(window.last_target_media_folder_path), MEDIA_FILTER)
    add_target_media(window, files)


def choose_media_folder(window: 'MainWindow'):
    folder = QtWidgets.QFileDialog.getExistingDirectory(window, 'Add Folder of Videos and Images',
                                                        _start_dir(window.last_target_media_folder_path))
    if folder:
        add_target_media(window, _files_in_folder(folder))


def choose_face_files(window: 'MainWindow'):
    files, _ = QtWidgets.QFileDialog.getOpenFileNames(window, 'Add Face Photos',
                                                      _start_dir(window.last_input_media_folder_path or os.path.expanduser('~/Pictures')),
                                                      IMAGE_FILTER)
    add_input_faces(window, files)


def choose_face_folder(window: 'MainWindow'):
    folder = QtWidgets.QFileDialog.getExistingDirectory(window, 'Add Folder of Face Photos',
                                                        _start_dir(window.last_input_media_folder_path or os.path.expanduser('~/Pictures')))
    if folder:
        add_input_faces(window, _files_in_folder(folder, images_only=True))


def _update_counts(window: 'MainWindow', *_):
    for label, items, noun in ((window._media_count_label, window.target_videos, 'item'),
                               (window._faces_count_label, window.input_faces, 'photo')):
        n = len(items)
        label.setText(f'{n} {noun}{"s" if n != 1 else ""}' if n else '')


def _section_header(window, title, actions):
    header = QtWidgets.QWidget()
    header.setObjectName('macSectionHeader')
    layout = QtWidgets.QHBoxLayout(header)
    layout.setContentsMargins(4, 6, 2, 2)
    title_label = QtWidgets.QLabel(title)
    title_label.setObjectName('macSectionTitle')
    count_label = QtWidgets.QLabel('')
    count_label.setObjectName('macSectionCount')
    add_button = QtWidgets.QToolButton()
    add_button.setObjectName('macAddButton')
    add_button.setText('+')
    add_button.setToolTip(f'Add {title.lower()}')
    add_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QtWidgets.QMenu(add_button)
    for text, shortcut, slot in actions:
        if text is None:
            menu.addSeparator()
            continue
        action = menu.addAction(text)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
            action.setShortcutVisibleInContextMenu(True)
        action.triggered.connect(slot)
    add_button.setMenu(menu)
    layout.addWidget(title_label)
    layout.addWidget(count_label)
    layout.addStretch(1)
    layout.addWidget(add_button)
    return header, count_label


class _EmptyListClick(QtCore.QObject):
    """Clicking an empty list opens the file picker (instead of a folder picker)."""

    def __init__(self, lists, parent=None):
        super().__init__(parent)
        self._lists = lists  # viewport -> (is_empty callable, open callable)

    def eventFilter(self, obj, event):
        entry = self._lists.get(obj)
        if entry and event.type() == QtCore.QEvent.Type.MouseButtonPress \
                and event.button() == QtCore.Qt.MouseButton.LeftButton and entry[0]():
            entry[1]()
            return True
        return False


def _paint_media_caption(original_paint):
    def paint(self, event):
        original_paint(self, event)
        name = os.path.basename(getattr(self, 'media_path', '') or '')
        if not name:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(3, 0, -3, -3)
        strip = QtCore.QRect(rect.left(), rect.bottom() - 16, rect.width(), 17)
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(strip), 4, 4)
        painter.fillPath(path, QtGui.QColor(0, 0, 0, 150))
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QtGui.QColor('#FFFFFF'))
        metrics = QtGui.QFontMetrics(font)
        text = metrics.elidedText(name, QtCore.Qt.TextElideMode.ElideMiddle, strip.width() - 6)
        painter.drawText(strip, QtCore.Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
    return paint


def setup_pickers(window: 'MainWindow'):
    window._mac_loaders = []

    media_header, window._media_count_label = _section_header(window, 'Videos', [
        ('Add Videos and Images…  ⌘O', None, lambda: choose_media_files(window)),
        ('Add Folder…  ⇧⌘O', None, lambda: choose_media_folder(window)),
    ])
    faces_header, window._faces_count_label = _section_header(window, 'Faces', [
        ('Add Face Photos…  ⌘I', None, lambda: choose_face_files(window)),
        ('Add Folder…  ⇧⌘I', None, lambda: choose_face_folder(window)),
    ])

    layout = window.vboxLayout
    for old_box, header in ((window.groupBox_TargetVideos_Select, media_header),
                            (window.groupBox_InputFaces_Select, faces_header)):
        index = layout.indexOf(old_box)
        old_box.hide()
        layout.insertWidget(index, header)

    # The File menu items open the same Mac panels
    for action in (window.actionOpen_Video_Files, window.actionOpen_Videos_Folder,
                   window.actionLoad_Source_Image_Files, window.actionLoad_Source_Images_Folder):
        action.triggered.disconnect()
    window.actionOpen_Video_Files.setText('Add Videos and Images…')
    window.actionOpen_Video_Files.triggered.connect(lambda: choose_media_files(window))
    window.actionOpen_Videos_Folder.setText('Add Folder of Videos…')
    window.actionOpen_Videos_Folder.triggered.connect(lambda: choose_media_folder(window))
    window.actionLoad_Source_Image_Files.setText('Add Face Photos…')
    window.actionLoad_Source_Image_Files.triggered.connect(lambda: choose_face_files(window))
    window.actionLoad_Source_Images_Folder.setText('Add Folder of Face Photos…')
    window.actionLoad_Source_Images_Folder.triggered.connect(lambda: choose_face_folder(window))

    window.targetVideosSearchBox.setPlaceholderText('Search')
    window.inputFacesSearchBox.setPlaceholderText('Search')
    window.targetVideosSearchBox.setClearButtonEnabled(True)
    window.inputFacesSearchBox.setClearButtonEnabled(True)

    for list_widget, text in ((window.targetVideosList, 'No Videos'), (window.inputFacesList, 'No Faces')):
        label = getattr(list_widget, 'placeholder_label', None)
        if label is not None:
            hint = 'videos or images' if list_widget is window.targetVideosList else 'face photos'
            label.setText(f"<div style='text-align:center'><p style='font-size:15px; font-weight:600'>{text}</p>"
                          f"<p style='font-size:12px'>Click + or drop {hint} here</p></div>")
            label.setStyleSheet('color: #8E8E93;')
        list_widget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    window._empty_click_filter = _EmptyListClick({
        window.targetVideosList.viewport(): (lambda: not window.target_videos, lambda: choose_media_files(window)),
        window.inputFacesList.viewport(): (lambda: not window.input_faces, lambda: choose_face_files(window)),
    }, window)
    # installed after the app's own filter, so it runs first
    window.targetVideosList.viewport().installEventFilter(window._empty_click_filter)
    window.inputFacesList.viewport().installEventFilter(window._empty_click_filter)

    # Captions drawn as an overlay strip; the card's own tiny 8 px name label is hidden
    card = widget_components.TargetMediaCardButton
    if not getattr(card, '_mac_caption', False):
        card.paintEvent = _paint_media_caption(card.paintEvent)
        original_init = card.__init__

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            for label in self.findChildren(QtWidgets.QLabel):
                label.hide()
        card.__init__ = init
        card._mac_caption = True

    timer = QtCore.QTimer(window)
    timer.timeout.connect(partial(_update_counts, window))
    timer.start(1500)
    window._count_timer = timer


# --------------------------------------------------------------------------------------------
# Responsive layout
# --------------------------------------------------------------------------------------------

class _ElideOnResize(QtCore.QObject):
    """Shortens a label's text with "…" to fit its width; the full text stays in a property."""

    def eventFilter(self, label, event):
        if event.type() == QtCore.QEvent.Type.Resize:
            full = label.property('_full_text')
            if full is not None:
                elided = label.fontMetrics().elidedText(full, QtCore.Qt.TextElideMode.ElideRight, label.width())
                if elided != label.text():
                    label.blockSignals(True)
                    QtWidgets.QLabel.setText(label, elided)
                    label.blockSignals(False)
        return False


_elide_filter = None


def _make_eliding(label: QtWidgets.QLabel):
    global _elide_filter
    if _elide_filter is None:
        _elide_filter = _ElideOnResize()
    text = label.text()
    if not text or '<' in text:
        return
    label.setProperty('_full_text', text)
    if not label.toolTip():
        label.setToolTip(text)
    label.setMinimumWidth(40)
    label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, label.sizePolicy().verticalPolicy())
    label.installEventFilter(_elide_filter)


def setup_responsive(window: 'MainWindow'):
    window.input_Target_DockWidget.setMinimumSize(200, 0)
    window.dockWidgetContents.setMinimumSize(0, 0)
    window.controlOptionsDockWidget.setMinimumSize(320, 0)
    window.inputEmbeddingsList.setMinimumSize(120, 90)
    window.graphicsViewFrame.setMinimumSize(240, 160)
    window.facesPanelGroupBox.setMinimumSize(0, 150)
    for widget in (window.mediaLayout, window.centralwidget):
        widget.setMinimumSize(0, 0)
    window.setMinimumSize(960, 600)
    window.setDockNestingEnabled(True)

    # Mac sidebars have no title bars; they're shown/hidden from the Window menu
    for dock in (window.input_Target_DockWidget, window.controlOptionsDockWidget):
        dock.setTitleBarWidget(QtWidgets.QWidget())

    # Parameter sliders stretch with the panel instead of a fixed 130 px
    for slider in window.findChildren(QtWidgets.QSlider):
        if isinstance(slider, (widget_components.ParameterSlider, widget_components.ParameterDecimalSlider)):
            slider.setMinimumWidth(70)
            slider.setMaximumWidth(200)
            slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)

    # Inspector labels wrap instead of forcing the panel wider; no sideways scrolling
    for scroll in window.controlOptionsDockWidget.findChildren(QtWidgets.QScrollArea):
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label in scroll.widget().findChildren(QtWidgets.QLabel):
            _make_eliding(label)
        # the groups' "Minimum" policy keeps them at their preferred width; let them shrink
        for group in scroll.widget().findChildren(QtWidgets.QGroupBox):
            group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, group.sizePolicy().verticalPolicy())
        for combo in scroll.widget().findChildren(QtWidgets.QComboBox):
            combo.setMinimumContentsLength(6)
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            # an explicit minimum overrides the long-item size hint (e.g. "InStyleSwapper256 Version A")
            combo.setMinimumWidth(90)
            combo.setMaximumWidth(220)

    # Panel visibility moves from a row of checkboxes to the View menu
    for checkbox, title, shortcut in ((window.mediaPanelCheckBox, 'Show Media Controls', None),
                                      (window.facesPanelCheckBox, 'Show Faces Panel', 'Ctrl+Alt+F'),
                                      (window.parametersPanelCheckBox, 'Show Parameters', None)):
        action = QtGui.QAction(title, window)
        action.setCheckable(True)
        action.setChecked(checkbox.isChecked())
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.toggled.connect(checkbox.setChecked)
        checkbox.toggled.connect(action.setChecked)
        window.menuView.addAction(action)
        checkbox.hide()
    # Sidebars take a sensible share of the window instead of fixed widths
    QtCore.QTimer.singleShot(0, lambda: _balance_docks(window))


def _balance_docks(window: 'MainWindow'):
    width = max(window.width(), 960)
    left = max(220, min(380, int(width * 0.2)))
    right = max(340, min(460, int(width * 0.26)))
    window.resizeDocks([window.input_Target_DockWidget, window.controlOptionsDockWidget], [left, right],
                       QtCore.Qt.Orientation.Horizontal)


# --------------------------------------------------------------------------------------------
# Result player
# --------------------------------------------------------------------------------------------

def _format_time(ms):
    seconds = max(0, int(ms // 1000))
    return f'{seconds // 60}:{seconds % 60:02d}'


class ResultPlayer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('resultPlayer')
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget(self)
        self.video.setMinimumSize(240, 160)
        self.video.setStyleSheet('background-color: black;')
        self.player.setVideoOutput(self.video)

        self.info = QtWidgets.QLabel('')
        self.info.setObjectName('resultInfo')

        self.play_button = QtWidgets.QToolButton()
        self.play_button.setObjectName('resultPlayButton')
        self.play_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle)
        self.position = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position.setObjectName('videoSeekSlider')
        self.position.sliderMoved.connect(self.player.setPosition)
        self.time_label = QtWidgets.QLabel('0:00 / 0:00')
        self.time_label.setMinimumWidth(90)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.position, 1)
        controls.addWidget(self.time_label)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.info)
        layout.addWidget(self.video, 1)
        layout.addLayout(controls)

        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._state_changed)
        self.player.mediaStatusChanged.connect(self._status_changed)
        self.video.videoSink().videoFrameChanged.connect(self._frame_arrived)
        self._priming = False

        shortcut = QtGui.QShortcut(QtGui.QKeySequence('Space'), self, self.toggle)
        shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def load(self, path, info_text):
        self.player.stop()
        self.player.setSource(QtCore.QUrl())
        self.player.setSource(QtCore.QUrl.fromLocalFile(path))
        self.info.setText(info_text)
        self._priming = True

    def _status_changed(self, status):
        # show the first frame right away, like QuickTime: play muted until a frame arrives
        if self._priming and status == QMediaPlayer.MediaStatus.LoadedMedia:
            self.audio.setMuted(True)
            self.player.play()

    def _frame_arrived(self, _frame):
        if self._priming:
            self._priming = False
            self.player.pause()
            self.player.setPosition(0)
            self.audio.setMuted(False)

    def _duration_changed(self, duration):
        self.position.setRange(0, duration)
        self.time_label.setText(f'{_format_time(self.player.position())} / {_format_time(duration)}')

    def unload(self):
        self.player.stop()
        self.player.setSource(QtCore.QUrl())

    def toggle(self):
        if self._priming:
            self._priming = False
            self.audio.setMuted(False)
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _position_changed(self, pos):
        if not self.position.isSliderDown():
            self.position.setValue(pos)
        self.time_label.setText(f'{_format_time(pos)} / {_format_time(self.player.duration())}')

    def _state_changed(self, state):
        icon = QtWidgets.QStyle.StandardPixmap.SP_MediaPause if state == QMediaPlayer.PlaybackState.PlayingState \
            else QtWidgets.QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))


# --------------------------------------------------------------------------------------------
# Process -> Result -> Export
# --------------------------------------------------------------------------------------------

class ProcessController(QtCore.QObject):
    def __init__(self, window: 'MainWindow'):
        super().__init__(window)
        self.window = window
        self.vp = window.video_processor
        self.result_path = None
        self.result_media_path = None
        self.started_at = 0.0
        self.cancelled = False

        self._build_toolbar()
        self._build_stack()

        self.vp.processing_finished_signal.connect(self._finished)
        self.progress_timer = QtCore.QTimer(self)
        self.progress_timer.timeout.connect(self._update_progress)
        self.media_timer = QtCore.QTimer(self)
        self.media_timer.timeout.connect(self._check_media)
        self.media_timer.start(1000)

        # The old record button (and its R shortcut) now starts/cancels processing
        record = window.buttonMediaRecord
        record.toggled.disconnect()  # was connected to the old record-to-folder action
        record.setCheckable(False)
        record.clicked.connect(self.start_or_cancel)
        record.hide()

    # ---- UI -------------------------------------------------------------------------------
    def _build_toolbar(self):
        w = self.window
        bar = QtWidgets.QToolBar('Toolbar', w)
        bar.setObjectName('macToolbar')
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.toggleViewAction().setVisible(False)
        w.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)

        self.process_button = QtWidgets.QPushButton('Process Video')
        self.process_button.setObjectName('processButton')
        self.process_button.setToolTip('Swap faces in the whole video (⌘R)')
        self.process_button.clicked.connect(self.start_or_cancel)
        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+R'), w, self.start_or_cancel)

        self.progress_box = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(self.progress_box)
        box.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName('processProgress')
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(180)
        self.progress_label = QtWidgets.QLabel('')
        self.progress_label.setObjectName('processLabel')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.cancel)
        box.addWidget(self.progress_bar)
        box.addWidget(self.progress_label)
        box.addWidget(self.cancel_button)

        self.status_label = QtWidgets.QLabel('')
        self.status_label.setObjectName('processLabel')
        self.reveal_button = QtWidgets.QPushButton('Show in Finder')
        self.reveal_button.setFlat(True)
        self.reveal_button.clicked.connect(self._reveal_export)
        self.last_export = None

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

        self.edit_button = QtWidgets.QPushButton('Edit')
        self.edit_button.setObjectName('segmentLeft')
        self.result_button = QtWidgets.QPushButton('Result')
        self.result_button.setObjectName('segmentRight')
        for b in (self.edit_button, self.result_button):
            b.setCheckable(True)
        self.edit_button.setChecked(True)
        self.result_button.setEnabled(False)
        self.result_button.setToolTip('Process the video to see the result')
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.edit_button)
        group.addButton(self.result_button)
        self.edit_button.clicked.connect(lambda: self.show_view('edit'))
        self.result_button.clicked.connect(lambda: self.show_view('result'))
        segment = QtWidgets.QWidget()
        seg_layout = QtWidgets.QHBoxLayout(segment)
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(0)
        seg_layout.addWidget(self.edit_button)
        seg_layout.addWidget(self.result_button)

        self.export_button = QtWidgets.QPushButton('Export…')
        self.export_button.setObjectName('exportButton')
        self.export_button.setToolTip('Save the processed video (⌘E)')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)
        export_action = QtGui.QAction('Export Video…', w)
        export_action.setShortcut(QtGui.QKeySequence('Ctrl+E'))
        export_action.triggered.connect(self.export)
        w.menuFile.insertAction(w.menuFile.actions()[0] if w.menuFile.actions() else None, export_action)
        process_action = QtGui.QAction('Process Video', w)
        process_action.triggered.connect(self.start_or_cancel)
        w.menuFile.insertAction(export_action, process_action)
        w.menuFile.insertSeparator(w.menuFile.actions()[2])
        self.export_action = export_action

        # toolbar widgets are shown/hidden through their actions
        self._actions = {}
        for name, widget in (('process', self.process_button), ('progress', self.progress_box),
                             ('status', self.status_label), ('reveal', self.reveal_button), ('spacer', spacer)):
            self._actions[name] = bar.addWidget(widget)
        self._actions['progress'].setVisible(False)
        self._actions['reveal'].setVisible(False)
        # Face compare / mask preview toggles live in the toolbar
        w.faceCompareCheckBox.setText('Compare')
        w.faceCompareCheckBox.setToolTip('Show the original and swapped faces side by side')
        w.faceMaskCheckBox.setText('Mask')
        w.faceMaskCheckBox.setToolTip('Show the face mask')
        bar.addWidget(w.faceCompareCheckBox)
        bar.addWidget(w.faceMaskCheckBox)
        bar.addSeparator()
        bar.addWidget(segment)
        bar.addSeparator()
        bar.addWidget(self.export_button)
        self.toolbar = bar

    def _build_stack(self):
        w = self.window
        layout = w.verticalLayout
        edit_page = QtWidgets.QWidget()
        edit_layout = QtWidgets.QVBoxLayout(edit_page)
        edit_layout.setContentsMargins(0, 0, 0, 0)
        layout.removeWidget(w.graphicsViewFrame)
        layout.removeItem(w.verticalLayoutMediaControls)
        edit_layout.addWidget(w.graphicsViewFrame, 1)
        edit_layout.addLayout(w.verticalLayoutMediaControls)

        self.player = ResultPlayer()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(edit_page)
        self.stack.addWidget(self.player)
        layout.insertWidget(1, self.stack, 1)

    def show_view(self, view):
        if view == 'result' and self.result_path:
            self.stack.setCurrentWidget(self.player)
            self.result_button.setChecked(True)
            self.window.facesPanelGroupBox.setVisible(False)
            self.player.setFocus()
        else:
            self.player.player.pause()
            self.stack.setCurrentIndex(0)
            self.edit_button.setChecked(True)
            self.window.facesPanelGroupBox.setVisible(self.window.facesPanelCheckBox.isChecked())

    # ---- processing ---------------------------------------------------------------------
    def start_or_cancel(self):
        if self.vp.process_to_preview and self.vp.processing:
            self.cancel()
        else:
            self.start()

    def start(self):
        w, vp = self.window, self.vp
        if vp.file_type != 'video' or not vp.media_capture:
            common_widget_actions.create_and_show_messagebox(
                w, 'No Video Selected', 'Select a video in the Videos list to process it. '
                'For images, use Save Image.', w)
            return
        if not misc_helpers.is_ffmpeg_in_path():
            common_widget_actions.create_and_show_messagebox(
                w, 'FFmpeg Not Found', 'Processing videos needs FFmpeg. Install it with: brew install ffmpeg', w)
            return
        if not (w.swapfacesButton.isChecked() or w.editFacesButton.isChecked()):
            answer = QtWidgets.QMessageBox.question(
                w, 'Swap Faces Is Off',
                'Swap Faces and Edit Faces are both off, so the video would come out unchanged.\n\n'
                'Turn on Swap Faces and process?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel)
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            w.swapfacesButton.setChecked(True)

        if vp.processing:
            vp.stop_processing()
        self.show_view('edit')
        self.player.unload()
        self.cancelled = False
        self.status_label.setText('')
        self._actions['reveal'].setVisible(False)

        w.videoSeekSlider.setValue(0)
        vp.current_frame_number = 0
        vp.recording = True
        vp.process_to_preview = True
        self.started_at = time.perf_counter()
        self._set_busy(True)
        w.buttonMediaPlay.setChecked(True)  # starts processing from the first frame
        if not vp.processing:
            # couldn't start (e.g. play state got out of sync): reset
            vp.recording = False
            vp.process_to_preview = False
            self._set_busy(False)

    def cancel(self):
        if not (self.vp.process_to_preview and self.vp.processing):
            return
        self.cancelled = True
        self.window.buttonMediaPlay.setChecked(False)

    def _set_busy(self, busy):
        self._actions['process'].setVisible(not busy)
        self._actions['progress'].setVisible(busy)
        self._actions['status'].setVisible(not busy)
        self.export_button.setEnabled(not busy and bool(self.result_path))
        self.result_button.setEnabled(not busy and bool(self.result_path))
        if busy:
            self.progress_bar.setRange(0, max(1, self.vp.max_frame_number + 1))
            self.progress_bar.setValue(0)
            self.progress_label.setText('Starting…')
            self.progress_timer.start(300)
        else:
            self.progress_timer.stop()

    def _update_progress(self):
        done = self.vp.next_frame_to_display
        total = self.vp.max_frame_number + 1
        self.progress_bar.setValue(min(done, total))
        elapsed = time.perf_counter() - self.started_at
        if done > 5:
            remaining = elapsed / done * (total - done)
            minutes, seconds = divmod(int(remaining), 60)
            left = f'about {minutes} min left' if minutes else f'{seconds} s left'
            self.progress_label.setText(f'{done:,} of {total:,} frames · {left}')
        else:
            self.progress_label.setText(f'{done:,} of {total:,} frames')

    @QtCore.Slot(str, bool)
    def _finished(self, path, completed):
        self._set_busy(False)
        if not completed or self.cancelled or not path or not os.path.isfile(path):
            if path and os.path.isfile(path):
                os.remove(path)
            self.status_label.setText('Processing cancelled')
            return
        self.result_path = path
        self.result_media_path = self.vp.media_path
        elapsed = time.perf_counter() - self.started_at
        minutes, seconds = divmod(int(elapsed), 60)
        self.status_label.setText(f'Processed in {minutes}:{seconds:02d}')
        self.result_button.setEnabled(True)
        self.result_button.setToolTip('')
        self.export_button.setEnabled(True)
        self.player.load(path, f'{Path(self.result_media_path).name} · processed')
        self.show_view('result')

    def _check_media(self):
        # a different video was selected: its result no longer applies
        if self.result_path and self.vp.media_path != self.result_media_path and not self.vp.processing:
            self.player.unload()
            self.result_path = None
            self.result_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.status_label.setText('')
            self._actions['reveal'].setVisible(False)
            if self.stack.currentWidget() is self.player:
                self.show_view('edit')

    # ---- export -----------------------------------------------------------------------
    def export(self):
        w = self.window
        if not self.result_path or not os.path.isfile(self.result_path):
            common_widget_actions.create_and_show_messagebox(
                w, 'Nothing to Export', 'Process the video first, then export the result.', w)
            return
        folder = w.control.get('OutputMediaFolder') or ''
        folder = folder if folder and os.path.isdir(folder) else os.path.expanduser('~/Movies')
        suggested = os.path.join(folder, f'{Path(self.result_media_path).stem}_swapped.mp4')
        target, _ = QtWidgets.QFileDialog.getSaveFileName(w, 'Export Video', suggested, 'MPEG-4 Video (*.mp4)')
        if not target:
            return
        if not target.lower().endswith('.mp4'):
            target += '.mp4'
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            shutil.copyfile(self.result_path, target)
        except OSError as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            common_widget_actions.create_and_show_messagebox(w, 'Export Failed', str(e), w)
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        w.control['OutputMediaFolder'] = os.path.dirname(target)
        self.last_export = target
        self.status_label.setText(f'Exported “{os.path.basename(target)}”')
        self._actions['reveal'].setVisible(True)

    def _reveal_export(self):
        if self.last_export and os.path.exists(self.last_export):
            subprocess.Popen(['open', '-R', self.last_export])


def setup(window: 'MainWindow'):
    setup_pickers(window)
    setup_responsive(window)
    window.process_controller = ProcessController(window)
