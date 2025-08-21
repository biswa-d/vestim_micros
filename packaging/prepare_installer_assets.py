#!/usr/bin/env python3
"""
Script to prepare installer assets for VEstim package
This script creates the default_data folder structure that will be bundled with the installer
"""

import os
import shutil
from pathlib import Path

def prepare_installer_assets():
    """Prepare installer assets with demo data files"""
    
    # Get paths
    repo_root = Path(__file__).parent
    source_data_dir = repo_root / "data"
    assets_dir = repo_root / "installer_assets"
    default_data_dir = assets_dir / "default_data"
    
    # Create directory structure
    print("Creating installer assets directory structure...")
    default_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    train_dir = default_data_dir / "train_data"
    val_dir = default_data_dir / "val_data"
    test_dir = default_data_dir / "test_data"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Copy demo files
    print("Copying demo data files...")
    
    # Copy training data
    train_source = source_data_dir / "train_data" / "Combined_Training31-Aug-2023.csv"
    if train_source.exists():
        train_target = train_dir / "demo_train_data.csv"
        shutil.copy2(train_source, train_target)
        print(f"✓ Copied training data: {train_target}")
    else:
        print(f"✗ Training data not found: {train_source}")
    
    # Copy validation data
    val_source = source_data_dir / "val_data" / "raw_data" / "105_UDDS_n20C.csv"
    if val_source.exists():
        val_target = val_dir / "demo_validation_data.csv"
        shutil.copy2(val_source, val_target)
        print(f"✓ Copied validation data: {val_target}")
    else:
        print(f"✗ Validation data not found: {val_source}")
    
    # Copy test data
    test_source = source_data_dir / "test_data" / "raw_data" / "107_LA92_n20C.csv"
    if test_source.exists():
        test_target = test_dir / "demo_test_data.csv"
        shutil.copy2(test_source, test_target)
        print(f"✓ Copied test data: {test_target}")
    else:
        print(f"✗ Test data not found: {test_source}")
    
    # Copy combined test data as additional option
    combined_test_source = source_data_dir / "Combined_Testing31-Aug-2023.csv"
    if combined_test_source.exists():
        combined_test_target = test_dir / "demo_combined_test_data.csv"
        shutil.copy2(combined_test_source, combined_test_target)
        print(f"✓ Copied combined test data: {combined_test_target}")
    else:
        print(f"✗ Combined test data not found: {combined_test_source}")
    
    # Create default settings file template for installer
    default_settings = {
        "last_used": {
            "train_folder": "${DATA_DIR}/train_data",
            "val_folder": "${DATA_DIR}/val_data",
            "test_folder": "${DATA_DIR}/test_data",
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
            "train_folder": "${DATA_DIR}/train_data",
            "val_folder": "${DATA_DIR}/val_data",
            "test_folder": "${DATA_DIR}/test_data"
        }
    }
    
    import json
    settings_template = assets_dir / "default_settings_template.json"
    with open(settings_template, 'w') as f:
        json.dump(default_settings, f, indent=4)
    print(f"✓ Created default settings template: {settings_template}")
    
    # Create README for installer assets
    readme_content = """# VEstim Installer Assets

This directory contains the default data files and settings that will be bundled with the VEstim installer.

## Structure:
- `default_data/` - Demo data files that will be copied to user's data directory
  - `train_data/demo_train_data.csv` - Sample training data
  - `val_data/demo_validation_data.csv` - Sample validation data  
  - `test_data/demo_test_data.csv` - Sample test data
  - `test_data/demo_combined_test_data.csv` - Additional test data option

- `default_settings_template.json` - Template for default settings (${DATA_DIR} will be replaced with actual data directory during installation)

## Usage:
1. Run `prepare_installer_assets.py` to update these files from the current data directory
2. Include the `installer_assets/default_data/` folder in your build process (PyInstaller/Inno Setup)
3. The installer should copy these files to the user-selected data directory during installation

## Customization:
You can replace any of the demo data files with different samples before building the installer.
Just ensure the column names match the default hyperparameters (SOC, Current, Temp, Voltage).
"""
    
    readme_path = assets_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"✓ Created README: {readme_path}")
    
    print(f"\n✅ Installer assets prepared successfully in: {assets_dir}")
    print("\nNext steps:")
    print("1. Review the demo data files in installer_assets/default_data/")
    print("2. Replace with different demo files if desired")
    print("3. Include installer_assets/default_data/ in your build process")
    print("4. Update your installer script to copy these files to the user's chosen data directory")

if __name__ == "__main__":
    prepare_installer_assets()
