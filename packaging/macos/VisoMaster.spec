# PyInstaller spec for the standalone macOS app (Apple Silicon).
#   pyinstaller --noconfirm packaging/macos/VisoMaster.spec
# Models are not bundled: they download to ~/Library/Application Support/VisoMaster on first use.
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = os.path.abspath(os.path.join(SPECPATH, '..', '..'))
VERSION = os.environ.get('VISOMASTER_VERSION', '0.1.6')

datas = [
    # The whole app package as files too: styles, icons, the CLIP vocabulary etc. are read by path
    (os.path.join(ROOT, 'app'), 'app'),
    (os.path.join(ROOT, 'model_assets', 'meanshape_68.pkl'), 'model_assets'),
    (os.path.join(ROOT, 'model_assets', 'liveportrait_onnx', 'lip_array.pkl'), 'model_assets/liveportrait_onnx'),
    (os.path.join(ROOT, 'README_MACOS.md'), '.'),
    (os.path.join(ROOT, 'LICENSE'), '.'),
]
binaries = []
hiddenimports = ['PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets']

for package in ('onnxruntime', 'onnxsim', 'qdarktheme', 'qdarkstyle', 'pyqttoast', 'imageio_ffmpeg', 'kornia'):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
datas += collect_data_files('skimage')
binaries += collect_dynamic_libs('torch')

a = Analysis(
    [os.path.join(ROOT, 'main.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['tensorrt', 'pyvirtualcam', 'tkinter', 'matplotlib', 'IPython', 'jupyter', 'notebook',
              'torch.utils.tensorboard', 'tensorboard'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisoMaster',
    console=False,
    target_arch='arm64',
    codesign_identity=os.environ.get('CODESIGN_IDENTITY'),
)
coll = COLLECT(exe, a.binaries, a.datas, name='VisoMaster')
app = BUNDLE(
    coll,
    name='VisoMaster.app',
    icon=os.path.join(ROOT, 'build', 'VisoMaster.icns'),
    bundle_identifier='com.visomaster.app',
    version=VERSION,
    info_plist={
        'CFBundleDisplayName': 'VisoMaster',
        'CFBundleShortVersionString': VERSION,
        'CFBundleVersion': VERSION,
        'LSMinimumSystemVersion': '13.0',
        'LSArchitecturePriority': ['arm64'],
        'NSHighResolutionCapable': True,
        'NSCameraUsageDescription': 'VisoMaster uses the camera for live face swapping.',
        'NSRequiresAquaSystemAppearance': False,
    },
)
