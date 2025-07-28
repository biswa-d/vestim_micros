#!/usr/bin/env python3
"""
PyInstaller build script for Vestim
Creates a standalone executable bundle
"""

import PyInstaller.__main__
import os
import sys
import shutil
from pathlib import Path
import datetime
import subprocess

def prepare_demo_data():
    """Prepare demo data files and documentation for inclusion in the executable"""
    print("Preparing installer assets...")
    
    # Create installer assets directory
    assets_dir = Path("installer_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create demo data directory structure
    demo_data_dir = assets_dir / "demo_data"
    demo_data_dir.mkdir(exist_ok=True)
    
    train_dir = demo_data_dir / "train_data"
    val_dir = demo_data_dir / "val_data"
    test_dir = demo_data_dir / "test_data"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Source data directory
    source_data_dir = Path("data")
    
    if not source_data_dir.exists():
        print("Warning: No source data directory found. Installer will create empty demo structure.")
        return str(assets_dir)
    
    try:
        # Copy training data
        train_source = source_data_dir / "train_data" / "Combined_Training31-Aug-2023.csv"
        if train_source.exists():
            train_target = train_dir / "demo_train_data.csv"
            shutil.copy2(train_source, train_target)
            print(f"✓ Copied training data: {train_target.name}")
        
        # Copy validation data - try to get a file from val_data/raw_data
        val_raw_dir = source_data_dir / "val_data" / "raw_data"
        if val_raw_dir.exists():
            val_files = list(val_raw_dir.glob("*.csv"))
            if val_files:
                val_source = val_files[0]  # Take the first CSV file
                val_target = val_dir / "demo_validation_data.csv"
                shutil.copy2(val_source, val_target)
                print(f"✓ Copied validation data: {val_target.name}")
        
        # Copy test data - try to get a file from test_data/raw_data
        test_raw_dir = source_data_dir / "test_data" / "raw_data"
        if test_raw_dir.exists():
            test_files = list(test_raw_dir.glob("*.csv"))
            if test_files:
                test_source = test_files[0]  # Take the first CSV file
                test_target = test_dir / "demo_test_data.csv"
                shutil.copy2(test_source, test_target)
                print(f"✓ Copied test data: {test_target.name}")
        
        # Also copy the combined testing file as an alternative
        combined_test_source = source_data_dir / "Combined_Testing31-Aug-2023.csv"
        if combined_test_source.exists():
            combined_test_target = test_dir / "demo_combined_test_data.csv"
            shutil.copy2(combined_test_source, combined_test_target)
            print(f"✓ Copied combined test data: {combined_test_target.name}")
        
        print(f"✓ Demo data prepared in: {demo_data_dir}")
        
    except Exception as e:
        print(f"Warning: Could not prepare some demo files: {e}")
    
    # Create default settings template for installer
    create_default_settings_template(assets_dir)
    
    # Create user README for projects folder
    create_user_readme(assets_dir)
    
    return str(assets_dir)

def create_default_settings_template(assets_dir):
    """Create default settings template for the installer"""
    default_settings = {
        "last_used": {
            "train_folder": "{PROJECTS_DIR}\\data\\train_data",
            "val_folder": "{PROJECTS_DIR}\\data\\val_data",
            "test_folder": "{PROJECTS_DIR}\\data\\test_data",
            "file_format": "csv",
            "hyperparams": {
                "FEATURE_COLUMNS": ["SOC", "Current", "Temp"],
                "TARGET_COLUMN": "Voltage",
                "MODEL_TYPE": "LSTM",
                "LAYERS": "1",
                "HIDDEN_UNITS": "10",
                "TRAINING_METHOD": "Sequence-to-Sequence",
                "LOOKBACK": "400",
                "BATCH_TRAINING": True,
                "BATCH_SIZE": "200",
                "SCHEDULER_TYPE": "StepLR",
                "INITIAL_LR": "0.0001",
                "LR_PARAM": "0.1",
                "LR_PERIOD": "2",
                "PLATEAU_PATIENCE": "10",
                "PLATEAU_FACTOR": "0.1",
                "VALID_PATIENCE": "10",
                "VALID_FREQUENCY": "1",
                "MAX_EPOCHS": "5",
                "REPETITIONS": 1,
                "DEVICE_SELECTION": "CPU",
                "MAX_TRAINING_TIME_SECONDS": 0,
                "TRAIN_VAL_SPLIT": "0.8",
                "LR_DROP_FACTOR": "0.5",
                "LR_DROP_PERIOD": "1",
                "ValidFrequency": "1",
                "SEQUENCE_SPLIT_METHOD": "temporal",
                "MAX_TRAIN_HOURS": "0",
                "MAX_TRAIN_MINUTES": "30",
                "MAX_TRAIN_SECONDS": "0"
            }
        },
        "default_folders": {
            "train_folder": "{PROJECTS_DIR}\\data\\train_data",
            "val_folder": "{PROJECTS_DIR}\\data\\val_data",
            "test_folder": "{PROJECTS_DIR}\\data\\test_data"
        }
    }
    
    import json
    settings_template = assets_dir / "default_settings_template.json"
    with open(settings_template, 'w') as f:
        json.dump(default_settings, f, indent=4)
    print(f"✓ Created default settings template: {settings_template}")

def create_user_readme(assets_dir):
    """Copy comprehensive user README to installer assets"""
    try:
        # Use the comprehensive README we created
        source_readme = Path("USER_README.md")
        if source_readme.exists():
            target_readme = assets_dir / "USER_README.md"
            shutil.copy2(source_readme, target_readme)
            print(f"✓ Copied comprehensive user README: {target_readme}")
            return str(target_readme)
        else:
            print("Warning: USER_README.md not found, creating basic README")
            # Fallback to basic README
            readme_content = """# VEstim - Voltage Estimation Modeling Tool

Welcome to VEstim! This folder contains your Vestim projects and data files.

## Quick Start
1. Launch VEstim from Start Menu or Desktop shortcut
2. Use the demo data files included in the data/ folders to try your first model
3. Follow the 4-step workflow: Data Import → Preprocessing → Hyperparameters → Training

## Folder Structure
- **data/** - Your datasets (train_data/, val_data/, test_data/)
- **job_YYYYMMDD-HHMMSS/** - Training results and models
- **default_settings.json** - Your saved preferences

## Demo Data
Sample battery data files are included with columns: SOC, Current, Temp, Voltage

For detailed instructions, see the full documentation.
"""
            
            readme_path = assets_dir / "USER_README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print(f"✓ Created basic user README: {readme_path}")
            return str(readme_path)
            
    except Exception as e:
        print(f"Warning: Could not create user README: {e}")
        return None

def build_executable():
    """Build standalone executable using PyInstaller"""
    
    # Prepare installer assets first
    assets_dir = prepare_demo_data()
    
    # Get the main script path
    main_script = "vestim/gui/src/data_import_gui_qt.py"
    
    # Get icon path if it exists
    icon_path = "vestim/gui/resources/icon.ico"
    if not Path(icon_path).exists():
        icon_path = None
    
    # Get version, date, and branch for unique naming
    version = "2.0.0"
    build_date = datetime.datetime.now().strftime("%Y_%B_%d")
    try:
        branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    except Exception:
        branch_name = "unknown"
    exe_name = f"Vestim_{version}_{build_date}_{branch_name}"

    # PyInstaller arguments
    args = [
        main_script,
        f'--name={exe_name}',
        '--onefile',  # Create single executable
        # Note: Removed --windowed to show console window with logs
        '--add-data=vestim;vestim',  # Include entire vestim package
        '--add-data=hyperparams.json;.',  # Include config files
        '--add-data=USER_README.md;.',  # Include comprehensive user README
        f'--add-data={assets_dir};installer_assets',  # Include all installer assets
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtWidgets', 
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=matplotlib',
        '--hidden-import=sklearn',
        '--hidden-import=torch',
        '--hidden-import=scipy',
        '--collect-all=vestim',  # Ensure all vestim modules are included
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
    ]
    
    # Add icon if available
    if icon_path:
        args.append(f'--icon={icon_path}')
    
    # Add version info for Windows
    if sys.platform == "win32":
        version_info = create_version_file()
        if version_info:
            args.append(f'--version-file={version_info}')
    
    print("Building executable with PyInstaller...")
    print(f"Arguments: {' '.join(args)}")
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("✓ Executable built successfully!")
    print(f"✓ Output: dist/{exe_name}.exe")
    print(f"✓ Installer assets included from: {assets_dir}")

    # The uniquely named executable is now kept in the dist folder

def create_version_file():
    """Create version file for Windows executable"""
    version_content = '''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
# filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
# Set not needed items to zero 0.
filevers=(1,0,0,0),
prodvers=(2,0,0,0),
# Contains a bitmask that specifies the valid bits 'flags'r
mask=0x3f,
# Contains a bitmask that specifies the Boolean attributes of the file.
flags=0x0,
# The operating system for which this file was designed.
# 0x4 - NT and there is no need to change it.
OS=0x4,
# The general type of file.
# 0x1 - the file is an application.
fileType=0x1,
# The function of the file.
# 0x0 - the function is not defined for this fileType
subtype=0x0,
# Creation date and time stamp.
date=(0, 0)
),
  kids=[
StringFileInfo(
  [
  StringTable(
    u'040904B0',
    [StringStruct(u'CompanyName', u'Biswanath Dehury'),
    StringStruct(u'FileDescription', u'Vestim - Advanced battery modeling with separate train/valid/test datasets and Optuna optimization.'),
    StringStruct(u'FileVersion', u'2.0.0'),
    StringStruct(u'InternalName', u'Vestim'),
    StringStruct(u'LegalCopyright', u'Copyright (c) 2025 Biswanath Dehury'),
    StringStruct(u'OriginalFilename', u'Vestim.exe'),
    StringStruct(u'ProductName', u'Vestim'),
    StringStruct(u'ProductVersion', u'2.0.0')])
  ]), 
VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)'''
    
    version_file = Path("version_info.txt")
    version_file.write_text(version_content)
    return str(version_file)

if __name__ == "__main__":
    try:
        build_executable()
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)
