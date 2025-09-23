# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('vestim', 'vestim'), ('hyperparams.json', '.'), ('USER_README.md', '.'), ('packaging/MODEL_DEPLOYMENT_GUIDE.md', '.'), ('requirements_cpu.txt', '.'), ('installer_assets', 'installer_assets')]
binaries = [('C:\\Users\\dehuryb\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\torch\\lib', 'torch/lib')]
hiddenimports = ['PyQt5.QtCore', 'PyQt5.QtWidgets', 'PyQt5.QtGui', 'pandas', 'numpy', 'matplotlib', 'matplotlib.backends.backend_qt5agg', 'sklearn', 'torch', 'optuna', 'seaborn', 'sklearn.utils._typedefs', 'scipy', 'h5py']
tmp_ret = collect_all('vestim')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['launch_gui_qt.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['packaging/runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PyBattML_2.0.1_2025_September_22_tvo_199_standalone_test',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon=['vestim\\gui\\resources\\icon.ico'],
)
