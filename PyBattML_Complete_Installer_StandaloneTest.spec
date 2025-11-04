# -*- mode: python ; coding: utf-8 -*-

import datetime
import os
import re
import subprocess


def _get_git_branch():
    """Return current git branch or 'unknown' if not available."""
    # Prefer an env var if provided by CI
    branch = os.environ.get('GIT_BRANCH') or os.environ.get('BRANCH_NAME')
    if branch:
        return branch
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=5,
        ).strip()
        return out or 'unknown'
    except Exception:
        return 'unknown'


def _sanitize(s: str) -> str:
    """Sanitize string for filesystem-friendly names."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)


# Version can be kept in sync with the app; adjust here if needed
_version = '2.0.1'
_date = datetime.datetime.now().strftime('%Y%m%d')
_branch = _sanitize(_get_git_branch())
_app_name = f'PyBattML_{_version}_{_date}_{_branch}'


a = Analysis(
    ['vestim_complete_installer.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Include main launch script
        ('launch_gui_qt.py', '.'),
        # Include core application source so installer can extract to install_dir
        ('vestim', 'vestim'),
        # Include default settings and templates
        ('defaults_templates', 'defaults_templates'),
        # Include sample data directory for first-run templates
        ('data', 'data'),
        # Include installer assets (demo data, readme, etc.) for first-run setup
        ('installer_assets', 'installer_assets'),
        # Include application icon so setup can copy it for shortcuts
        ('vestim\\gui\\resources\\PyBattML_icon.ico', 'vestim\\gui\\resources'),
        # Include sample data for testing and training (will be copied to project dir during install)
        # Include standalone test data file
        ('119_ReorderedUS06_n20C.csv', '.'),
        # Include documentation
        ('USER_README.md', '.'),
        ('packaging/NEW_PACKAGING_INSTRUCTIONS.txt', 'packaging'),
    ],
    hiddenimports=[
        'vestim',
        'vestim.config_manager',
        'vestim.logger_config',
        'vestim.gpu_setup',
        'vestim.smart_environment_setup',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'json',
        'subprocess',
        'sys',
        'os',
        'pathlib',
        'shutil',
        'urllib.request',
        'urllib.parse',
        'urllib.error',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML libraries - will be installed at runtime
        'torch',
        'torchvision', 
        'torchaudio',
        'pandas',
        'numpy',
        'scipy',
        'sklearn',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'optuna',
        'tensorboard',
        'tensorflow',
        'keras',
        'xgboost',
        'lightgbm',
        'catboost',
    ],
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
    name=_app_name,
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
    icon='vestim\\gui\\resources\\PyBattML_icon.ico',
)
