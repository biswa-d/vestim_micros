# VEstim Packaging and Distribution Guide

## Overview

This document explains the complete packaging workflow for VEstim, which creates a professional Windows installer with embedded demo data and comprehensive user documentation.

## Build Process

### Single Command Build
```bash
build.bat
```

This script performs the complete packaging workflow:

1. **Dependency Installation** - Installs PyInstaller and build tools
2. **Asset Preparation** - Prepares demo data and documentation
3. **Executable Creation** - Creates standalone Vestim.exe
4. **Installer Creation** - Creates professional Windows installer

## What Gets Packaged

### Embedded in Executable
- Complete VEstim application
- Demo data files (train/val/test CSV files)
- Comprehensive user README (USER_README.md)
- Default hyperparameter settings
- All Python dependencies

### Installer Features
- User selects installation directory for Vestim.exe
- User selects projects folder location (default: Documents folder)
- Creates complete directory structure in projects folder
- Desktop and Start Menu shortcuts
- Professional uninstaller with cleanup

## User Installation Experience

### Step 1: Installation Directory
User selects where to install VEstim (e.g., `C:\Program Files\VEstim\`)

### Step 2: Projects Directory  
User selects where to create the `vestim_projects` folder (e.g., `C:\Users\Username\Documents\`)

### Step 3: Automatic Setup
Installer creates:
```
C:\Users\Username\Documents\vestim_projects\
├── data\
│   ├── train_data\
│   │   └── demo_train_data.csv
│   ├── val_data\
│   │   └── demo_validation_data.csv
│   └── test_data\
│       ├── demo_test_data.csv
│       └── demo_combined_test_data.csv
├── default_settings.json
└── README.md (comprehensive user guide)
```

### Step 4: Ready to Use
- User launches VEstim from Start Menu or Desktop
- Demo data is automatically loaded
- README.md provides complete usage instructions
- User can immediately start training models

## Configuration Files

### vestim_config.json (in installation directory)
```json
{
  "projects_directory": "C:\\Users\\Username\\Documents\\vestim_projects",
  "data_directory": "C:\\Users\\Username\\Documents\\vestim_projects\\data",
  "created_by_installer": true
}
```

### default_settings.json (in projects directory)
```json
{
  "last_used": {
    "train_folder": "C:\\Users\\Username\\Documents\\vestim_projects\\data\\train_data",
    "val_folder": "C:\\Users\\Username\\Documents\\vestim_projects\\data\\val_data",
    "test_folder": "C:\\Users\\Username\\Documents\\vestim_projects\\data\\test_data",
    "file_format": "csv"
  },
  "default_folders": {
    "train_folder": "C:\\Users\\Username\\Documents\\vestim_projects\\data\\train_data",
    "val_folder": "C:\\Users\\Username\\Documents\\vestim_projects\\data\\val_data",
    "test_folder": "C:\\Users\\Username\\Documents\\vestim_projects\\data\\test_data"
  }
}
```

## Demo Data Files

### Included Sample Data
- **demo_train_data.csv** - 762K+ battery measurement points
- **demo_validation_data.csv** - Validation dataset 
- **demo_test_data.csv** - Test dataset
- **demo_combined_test_data.csv** - Additional test data

### Data Format
All CSV files contain columns:
- **SOC** - State of Charge (0-1)
- **Current** - Battery current (A) 
- **Temp** - Temperature (°C)
- **Voltage** - Battery voltage (V)

## User Documentation

The comprehensive `USER_README.md` includes:

### Quick Start Guide
- Immediate setup with demo data
- 4-step workflow explanation
- First model training tutorial

### Complete GUI Walkthrough
- Data Import & Organization GUI
- Data Preprocessing & Augmentation GUI  
- Hyperparameter Selection GUI
- Model Training GUI
- Testing & Evaluation GUI

### Advanced Usage
- Using custom data
- Model optimization techniques
- Hyperparameter tuning strategies
- Troubleshooting common issues

### Machine Learning Concepts
- Time series prediction explanation
- Model types (LSTM/GRU/FNN) comparison
- Training process overview
- Validation and testing methodology

## Distribution

### Final Output
- **`installer_output\vestim-installer-1.0.0.exe`** - Professional Windows installer
- Completely standalone (no dependencies required)
- ~50-100MB size (includes Python runtime and ML libraries)

### User Requirements
- Windows 10/11 (64-bit)
- ~500MB disk space for installation
- ~1GB additional space for projects and models

### Testing Checklist
1. Test installer on clean Windows machine
2. Verify projects folder creation
3. Verify demo data loads correctly
4. Test complete workflow with demo data
5. Verify README.md displays properly
6. Test uninstaller cleanup

## Customization Before Packaging

### Updating Demo Data
Replace files in `data/` directory before running build:
- `data/train_data/Combined_Training31-Aug-2023.csv`
- `data/val_data/raw_data/*.csv`
- `data/test_data/raw_data/*.csv`
- `data/Combined_Testing31-Aug-2023.csv`

### Updating User Documentation
Edit `USER_README.md` before running build to customize user instructions.

### Updating Default Settings
Modify `config_manager.py` `_get_initial_default_hyperparams()` method to change default model settings.

## Technical Implementation

### PyInstaller Configuration
- Single executable (`--onefile`)
- Embedded data files (`--add-data`)
- All dependencies included (`--collect-all`)
- Console window enabled for debugging

### Inno Setup Configuration
- Modern wizard style
- User directory selection
- File association setup
- Registry cleanup on uninstall
- Professional metadata and icons

### Runtime Behavior
- Application detects if running from installer (checks for vestim_config.json)
- Automatically creates projects structure if missing
- Copies demo data from embedded assets on first run
- Saves user preferences in projects folder

## Maintenance

### Version Updates
1. Update version in `vestim_installer.iss`
2. Update version in `build_exe.py`
3. Update version info in USER_README.md
4. Run `build.bat` to create new installer

### Demo Data Updates
1. Replace files in `data/` directory
2. Verify column names match default hyperparameters
3. Test with actual training workflow
4. Run `build.bat` to rebuild with new data

This packaging system provides a complete, professional distribution solution that enables new users to immediately start using VEstim with working demo data and comprehensive documentation.
