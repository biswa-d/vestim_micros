# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['vestim_complete_installer.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include core PyBattML modules
        ('vestim', 'vestim'),
        # Include main launch script
        ('launch_gui_qt.py', '.'),
        # Include default settings and templates
        ('defaults_templates', 'defaults_templates'),
        # Include sample data for testing and training (will be copied to project dir during install)
        ('data', 'data'),
        # Include standalone test data file
        ('119_ReorderedUS06_n20C.csv', '.'),
        # Include documentation
        ('USER_README.md', '.'),
        ('packaging/MODEL_DEPLOYMENT_GUIDE.md', '.'),
    ],
    hiddenimports=[
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
    name='PyBattML_2.0.1_2025_September_22_tvo_199_standalone_test',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='vestim\\gui\\resources\\PyBattML_icon.ico',
)
