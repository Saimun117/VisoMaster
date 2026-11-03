"""
macOS look for VisoMaster: system font, system accent blue, follows macOS light/dark mode,
grouped "card" sections, pill tabs and rounded controls. The base widget styling comes from
qdarktheme, recolored to Apple's palette, with app-specific rules layered on top.
"""
import os
import sys

import qdarktheme
# white check mark icon, used on the accent-colored checkbox in both schemes
from qdarktheme.dist.dark import rc_icons  # noqa: F401
from PySide6 import QtCore, QtGui, QtWidgets

IS_MACOS = sys.platform == 'darwin'

# Apple system colors (https://developer.apple.com/design/human-interface-guidelines/color)
PALETTES = {
    'dark': {
        'accent': '#0A84FF', 'accent_hover': '#409CFF', 'accent_text': '#FFFFFF',
        'window': '#1E1E1E', 'sidebar': '#252526', 'card': '#2C2C2E', 'control': '#3A3A3C',
        'control_hover': '#48484A', 'border': '#3D3D40', 'separator': '#38383A',
        'text': '#F5F5F7', 'subtext': '#98989D', 'disabled': '#636366',
        'viewport': '#141414', 'track': '#48484A', 'switch_off': '#636366',
        'danger': '#FF453A', 'tooltip': '#3A3A3C',
    },
    'light': {
        'accent': '#007AFF', 'accent_hover': '#3395FF', 'accent_text': '#FFFFFF',
        'window': '#ECECEC', 'sidebar': '#E5E5E7', 'card': '#FFFFFF', 'control': '#FFFFFF',
        'control_hover': '#F2F2F7', 'border': '#D1D1D6', 'separator': '#D8D8DC',
        'text': '#1D1D1F', 'subtext': '#6E6E73', 'disabled': '#AEAEB2',
        'viewport': '#1C1C1E', 'track': '#D1D1D6', 'switch_off': '#D1D1D6',
        'danger': '#FF3B30', 'tooltip': '#FFFFFF',
    },
}

# qdarktheme colors -> macOS palette keys
_BASE_COLOR_MAP = {
    'dark': {
        'rgba(138.000, 180.000, 247.000, 1.000)': 'accent',
        'rgba(127.000, 166.000, 228.000, 1.000)': 'accent_hover',
        'rgba(32.000, 33.000, 36.000, 1.000)': 'window',
        'rgba(41.000, 42.000, 45.000, 1.000)': 'card',
        'rgba(63.000, 64.000, 66.000, 1.000)': 'border',
        'rgba(228.000, 231.000, 235.000, 1.000)': 'text',
        'rgba(0.000, 72.000, 117.000, 1.000)': 'accent',
        'rgba(30.000, 43.000, 60.000, 1.000)': 'control',
        'rgba(46.000, 70.000, 94.000, 1.000)': 'control_hover',
    },
    'light': {
        'rgba(0.000, 129.000, 219.000, 1.000)': 'accent',
        'rgba(76.000, 166.000, 229.000, 1.000)': 'accent_hover',
        'rgba(248.000, 249.000, 250.000, 1.000)': 'window',
        'rgba(218.000, 220.000, 224.000, 1.000)': 'border',
        'rgba(77.000, 81.000, 87.000, 1.000)': 'text',
        'rgba(226.000, 234.000, 251.000, 1.000)': 'control_hover',
    },
}

_APP_QSS = """
* {{
    font-family: "{font}";
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background-color: {window};
    color: {text};
}}
QWidget {{
    color: {text};
}}
QToolTip {{
    color: {text};
    background-color: {tooltip};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px 6px;
}}

/* Sidebars (dock widgets) */
QDockWidget {{
    color: {subtext};
    font-weight: 600;
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background-color: {sidebar};
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid {separator};
}}
QDockWidget > QWidget {{
    background-color: {sidebar};
}}

/* Grouped sections as rounded cards */
QGroupBox {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 10px;
    margin-top: 22px;
    padding: 10px 8px 6px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 6px;
    padding: 0 4px 2px 4px;
    color: {subtext};
    background-color: transparent;
}}

/* Pill-style tabs, like a segmented control */
QTabWidget::pane {{
    border: none;
    background: transparent;
    top: 4px;
}}
QTabWidget::tab-bar {{
    alignment: center;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: {control};
    color: {text};
    border: 1px solid {border};
    padding: 4px 14px;
    margin: 0px;
    font-weight: 500;
    min-width: 60px;
}}
QTabBar::tab:first {{
    border-top-left-radius: 7px;
    border-bottom-left-radius: 7px;
}}
QTabBar::tab:last {{
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
}}
QTabBar::tab:selected {{
    background: {accent};
    color: {accent_text};
    border-color: {accent};
}}
QTabBar::tab:hover:!selected {{
    background: {control_hover};
}}

QLabel {{
    background: transparent;
}}

/* Buttons */
QPushButton:flat {{
    background: transparent;
    border: none;
}}
QPushButton:flat:hover {{
    background-color: {control_hover};
}}
QPushButton {{
    background-color: {control};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 4px 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {control_hover};
}}
QPushButton:pressed {{
    background-color: {accent};
    color: {accent_text};
    border-color: {accent};
}}
QPushButton:checked {{
    background-color: {accent};
    color: {accent_text};
    border-color: {accent};
}}
/* Main face actions: real buttons, Swap/Edit light up in the accent color when on */
#findTargetFacesButton, #clearTargetFacesButton, #swapfacesButton, #editFacesButton {{
    background-color: {control};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 1px 12px;
    min-width: 96px;
}}
#findTargetFacesButton:hover, #clearTargetFacesButton:hover, #swapfacesButton:hover, #editFacesButton:hover {{
    background-color: {control_hover};
}}
#swapfacesButton:checked, #editFacesButton:checked {{
    background-color: {accent};
    color: {accent_text};
    border-color: {accent};
}}
QPushButton:disabled {{
    color: {disabled};
    background-color: {card};
}}

/* Text fields and drop-downs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {control};
    color: {text};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 3px 8px;
    selection-background-color: {accent};
    selection-color: {accent_text};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {accent};
    padding: 2px 7px;
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 6px;
    selection-background-color: {accent};
    selection-color: {accent_text};
    outline: none;
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid {border};
    background-color: {control};
}}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
    image: url(":qdarktheme/dist/dark/svg/check__icon-foreground__rotate-0.svg");
}}

/* Lists and the preview canvas */
QListView, QListWidget {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 10px;
    padding: 4px;
}}
QGraphicsView {{
    background-color: {viewport};
    border: 1px solid {border};
    border-radius: 10px;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}

/* Thin overlay-style scrollbars */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle {{
    background: {track};
    border-radius: 3px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{
    background: {subtext};
}}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
    border: none;
    width: 0px;
    height: 0px;
}}

/* Memory bar */
QProgressBar {{
    background-color: {control};
    border: 1px solid {border};
    border-radius: 6px;
    text-align: center;
    color: {text};
    font-size: 11px;
    min-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 5px;
}}

/* Media cards */
TargetMediaCardButton, TargetFaceCardButton, InputFaceCardButton, EmbeddingCardButton {{
    background-color: {control};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 6px;
}}
TargetMediaCardButton:hover, TargetFaceCardButton:hover, InputFaceCardButton:hover, EmbeddingCardButton:hover {{
    background-color: {control_hover};
    border-color: {accent};
}}
TargetMediaCardButton:checked, TargetFaceCardButton:checked, InputFaceCardButton:checked, EmbeddingCardButton:checked {{
    background-color: {control_hover};
    border: 2px solid {accent};
    color: {text};
}}

/* Video seek bar */
#videoSeekSlider::groove:horizontal {{
    height: 4px;
    border-radius: 2px;
    background-color: {track};
}}
#videoSeekSlider::sub-page:horizontal {{
    background-color: {accent};
    border-radius: 2px;
}}
#videoSeekSlider::add-page:horizontal {{
    background-color: {track};
    border-radius: 2px;
}}
#videoSeekSlider::handle:horizontal {{
    background-color: #FFFFFF;
    border: 1px solid {border};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

/* Parameter sliders: macOS-style thin track with round knob */
QSlider::groove:horizontal, ParameterSlider::groove:horizontal, ParameterDecimalSlider::groove:horizontal {{
    height: 4px;
    border-radius: 2px;
    background-color: {track};
    margin: 0px;
}}
QSlider::sub-page:horizontal, ParameterSlider::sub-page:horizontal, ParameterDecimalSlider::sub-page:horizontal {{
    background-color: {accent};
    border-radius: 2px;
}}
QSlider::add-page:horizontal, ParameterSlider::add-page:horizontal, ParameterDecimalSlider::add-page:horizontal {{
    background-color: {track};
    border-radius: 2px;
}}
QSlider::handle:horizontal, ParameterSlider::handle:horizontal, ParameterDecimalSlider::handle:horizontal {{
    background-color: #FFFFFF;
    border: 1px solid {border};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal:disabled, ParameterSlider::sub-page:horizontal:disabled, ParameterDecimalSlider::sub-page:horizontal:disabled {{
    background-color: {disabled};
}}

/* Sidebar section headers */
#macSectionTitle {{
    font-size: 13px;
    font-weight: 700;
    color: {text};
}}
#macSectionCount, #processLabel, #resultInfo {{
    color: {subtext};
    font-size: 12px;
}}
#macAddButton {{
    background: {control};
    border: 1px solid {border};
    border-radius: 6px;
    min-width: 22px;
    min-height: 22px;
    font-size: 16px;
    font-weight: 600;
    color: {text};
}}
#macAddButton:hover {{
    background: {control_hover};
}}
#macAddButton::menu-indicator {{
    image: none;
    width: 0px;
}}

/* Toolbar */
QToolBar#macToolbar {{
    background: {window};
    border: none;
    border-bottom: 1px solid {separator};
    padding: 6px 10px;
    spacing: 8px;
}}
#processButton {{
    background-color: {accent};
    color: {accent_text};
    border: 1px solid {accent};
    border-radius: 6px;
    padding: 4px 16px;
    font-weight: 600;
}}
#processButton:hover {{
    background-color: {accent_hover};
}}
#segmentLeft, #segmentRight {{
    padding: 3px 16px;
    min-width: 54px;
}}
#segmentLeft {{
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
}}
#segmentRight {{
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-left: none;
}}
#segmentLeft:checked, #segmentRight:checked {{
    background-color: {accent};
    color: {accent_text};
    border-color: {accent};
}}
#processProgress {{
    min-height: 6px;
    max-height: 6px;
    border-radius: 3px;
}}
#resultPlayButton {{
    background: transparent;
    border: none;
    padding: 4px;
}}

QMenuBar {{
    background-color: {window};
}}
QMenu {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 4px 18px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {accent};
    color: {accent_text};
}}
"""

_current_scheme = None


def system_font_family() -> str:
    if IS_MACOS:
        return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.GeneralFont).family()
    return QtWidgets.QApplication.font().family()


def system_color_scheme() -> str:
    forced = os.environ.get('VISO_FORCE_COLOR_SCHEME')
    if forced in ('light', 'dark'):
        return forced
    hints = QtGui.QGuiApplication.styleHints()
    if hasattr(hints, 'colorScheme') and hints.colorScheme() == QtCore.Qt.ColorScheme.Light:
        return 'light'
    return 'dark'


def stylesheet(scheme: str) -> str:
    colors = PALETTES[scheme]
    base = qdarktheme.load_stylesheet(scheme)
    for old, key in _BASE_COLOR_MAP[scheme].items():
        base = base.replace(old, colors[key])
    return base + '\n' + _APP_QSS.format(font=system_font_family(), **colors)


def apply(app: QtWidgets.QApplication, scheme: str | None = None):
    """Apply the macOS theme in the given ('light'/'dark') or the current system scheme."""
    global _current_scheme
    scheme = scheme or system_color_scheme()
    _current_scheme = scheme
    colors = PALETTES[scheme]

    font = QtGui.QFont(system_font_family())
    font.setPixelSize(13)
    app.setFont(font)
    # The Qt Designer file asks for Segoe UI, which doesn't exist on macOS
    QtGui.QFont.insertSubstitution('Segoe UI Semibold', system_font_family())
    QtGui.QFont.insertSubstitution('Segoe UI', system_font_family())

    palette = app.palette()
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(colors['accent']))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(colors['accent_text']))
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(colors['window']))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(colors['text']))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(colors['control']))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(colors['text']))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(colors['control']))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(colors['text']))
    palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor(colors['switch_off']))
    palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(colors['subtext']))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(scheme))
    for widget in app.allWidgets():
        if isinstance(widget, QtWidgets.QAbstractButton) and widget.objectName() in _TINTED_ICON_BUTTONS:
            _tint_icon(widget, QtGui.QColor(colors['text']))
        QtWidgets.QWidget.update(widget)  # item views overload update(index)


# Flat media buttons whose icons are white PNGs made for a dark background (play/record
# swap their icons at runtime and stay visible in both schemes, so they're left alone)
_TINTED_ICON_BUTTONS = {
    'frameRewindButton', 'frameAdvanceButton',
    'addMarkerButton', 'removeMarkerButton', 'previousMarkerButton', 'nextMarkerButton',
    'viewFullScreenButton',
}


def _tint_icon(button: QtWidgets.QAbstractButton, color: QtGui.QColor):
    original = button.property('_viso_original_icon')
    if original is None:
        original = button.icon()
        if original.isNull():
            return
        button.setProperty('_viso_original_icon', original)
    tinted = QtGui.QIcon()
    for size in original.availableSizes() or [button.iconSize()]:
        for mode in (QtGui.QIcon.Mode.Normal, QtGui.QIcon.Mode.Active):
            for state in (QtGui.QIcon.State.Off, QtGui.QIcon.State.On):
                pixmap = original.pixmap(size, mode, state)
                if pixmap.isNull():
                    continue
                painter = QtGui.QPainter(pixmap)
                painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceIn)
                painter.fillRect(pixmap.rect(), color)
                painter.end()
                tinted.addPixmap(pixmap, mode, state)
    button.setIcon(tinted)


def is_active() -> bool:
    return _current_scheme is not None


def deactivate():
    global _current_scheme
    _current_scheme = None


def follow_system(app: QtWidgets.QApplication, is_selected):
    """Re-apply the theme when macOS switches between light and dark, while the macOS theme
    is the selected one (is_selected is a callable)."""
    hints = QtGui.QGuiApplication.styleHints()
    if hasattr(hints, 'colorSchemeChanged'):
        hints.colorSchemeChanged.connect(lambda _scheme: is_selected() and apply(app))


def setup_window(window: QtWidgets.QMainWindow):
    """Mac window conventions: menu shortcuts, Settings (Cmd+,), Window and Help menus."""
    window.setUnifiedTitleAndToolBarOnMac(True)
    window.setWindowTitle('VisoMaster')
    window.clearMemoryButton.setText('Free Memory')
    window.clearMemoryButton.setToolTip('Unload all models to free memory')
    seq = QtGui.QKeySequence

    shortcuts = {
        window.actionOpen_Video_Files: seq('Ctrl+O'),
        window.actionOpen_Videos_Folder: seq('Ctrl+Shift+O'),
        window.actionLoad_Source_Image_Files: seq('Ctrl+I'),
        window.actionLoad_Source_Images_Folder: seq('Ctrl+Shift+I'),
        window.actionSave_CurrentWorkspace: seq('Ctrl+S'),
        window.actionLoad_SavedWorkspace: seq('Ctrl+Shift+L'),
    }
    for action, shortcut in shortcuts.items():
        action.setShortcut(shortcut)

    # Placeholder "Test" item from the Designer file
    window.actionTest_2.setVisible(False)
    window.menuEdit.menuAction().setVisible(False)

    window.actionView_Fullscreen_F11.setText('Enter Full Screen')
    window.actionView_Fullscreen_F11.setShortcut(seq('Ctrl+Meta+F'))

    settings_action = QtGui.QAction('Settings…', window)
    settings_action.setMenuRole(QtGui.QAction.MenuRole.PreferencesRole)
    settings_action.setShortcut(seq('Ctrl+,'))

    def open_settings():
        tabs = window.tabWidget
        for i in range(tabs.count()):
            if tabs.tabText(i).lower() == 'settings':
                window.controlOptionsDockWidget.show()
                tabs.setCurrentIndex(i)
                break
    settings_action.triggered.connect(open_settings)
    window.menuFile.addAction(settings_action)

    about_action = QtGui.QAction('About VisoMaster', window)
    about_action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
    about_action.triggered.connect(lambda: QtWidgets.QMessageBox.about(
        window, 'About VisoMaster',
        'VisoMaster 0.1.6 for macOS\n\nFace swapping and editing for images and videos.\n'
        'Models run on the Apple GPU through Core ML.'))
    window.menuFile.addAction(about_action)

    window_menu = window.topMenuBar.addMenu('Window')
    minimize = window_menu.addAction('Minimize')
    minimize.setShortcut(seq('Ctrl+M'))
    minimize.triggered.connect(window.showMinimized)
    zoom = window_menu.addAction('Zoom')
    zoom.triggered.connect(lambda: window.showNormal() if window.isMaximized() else window.showMaximized())
    window_menu.addSeparator()
    for title, dock in (('Media and Faces', window.input_Target_DockWidget),
                        ('Control Options', window.controlOptionsDockWidget)):
        toggle = dock.toggleViewAction()
        toggle.setText(f'Show {title}')
        window_menu.addAction(toggle)

    help_menu = window.topMenuBar.addMenu('Help')
    guide = help_menu.addAction('VisoMaster for Mac Guide')
    guide.triggered.connect(lambda: QtGui.QDesktopServices.openUrl(
        QtCore.QUrl.fromLocalFile(os.path.abspath('README_MACOS.md'))))
    shortcuts_help = help_menu.addAction('Keyboard Shortcuts')
    shortcuts_help.triggered.connect(lambda: QtWidgets.QMessageBox.information(
        window, 'Keyboard Shortcuts',
        'Space  Play / pause\nR  Record\nS  Swap faces on/off\n'
        'A / D  Back / forward 30 frames\nC / V  Back / forward 1 frame\nZ  Go to start\n'
        'F  Add marker    ⌥F  Remove marker\nQ / W  Previous / next marker\n'
        '⌘O  Open videos    ⌘I  Open source faces\n⌘S  Save workspace    ⌘,  Settings'))
