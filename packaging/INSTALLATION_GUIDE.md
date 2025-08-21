# Vestim Installation and Default Data Setup

## Overview

When Vestim is installed, it automatically creates a complete working environment with demo data files so users can immediately test the tool and understand what to expect.

## Installation Behavior

### 1. Data Directory Creation

During installation, Vestim creates a data directory structure:

```
[Installation Directory]/
├── Vestim.exe
├── default_data/
│   ├── train_data/
│   │   └── demo_train_data.csv
│   ├── val_data/
│   │   └── demo_validation_data.csv
│   └── test_data/
│       ├── demo_test_data.csv
│       └── demo_combined_test_data.csv
└── default_settings.json
```

### 2. User Data Directory

On first run, Vestim creates a user data directory:

**For regular installation:**
- `C:\Users\[username]\Documents\vestim_data\`

**During development:**
- `[repo_root]\data\`

With the following structure:
```
vestim_data/
├── train_data/
│   └── demo_train_data.csv (copied from installation)
├── val_data/
│   └── demo_validation_data.csv (copied from installation)
└── test_data/
    ├── demo_test_data.csv (copied from installation)
    └── demo_combined_test_data.csv (copied from installation)
```

### 3. Default Settings

The tool ships with sensible defaults:

- **Default file format:** CSV
- **Default model:** LSTM with Sequence-to-Sequence training
- **Default hyperparameters:** Optimized for demo data
- **Default feature columns:** SOC, Current, Temp (based on demo data)
- **Default target column:** Voltage

## Demo Data Files

### Training Data (`demo_train_data.csv`)
- Source: `Combined_Training31-Aug-2023.csv`
- Contains battery training data with voltage, current, SOC, temperature

### Validation Data (`demo_validation_data.csv`)
- Source: First CSV file from `val_data/raw_data/`
- Used for model validation during training

### Test Data
- `demo_test_data.csv`: First CSV file from `test_data/raw_data/`
- `demo_combined_test_data.csv`: Combined testing dataset
- Used for final model evaluation

## User Experience

### First-Time Startup
1. User installs Vestim
2. User launches Vestim.exe
3. Tool automatically:
   - Creates user data directory
   - Copies demo files from installation
   - Sets up default folders pointing to demo data
   - Loads demo data into the interface

### Immediate Testing
Users can immediately:
1. See data loaded in train/validation/test sections
2. Proceed to hyperparameter selection (pre-filled with good defaults)
3. Start training with demo data
4. View results and understand the workflow

### Custom Data
Users can easily switch to their own data:
1. Replace demo files with their own CSV files
2. Select different folders in the data import GUI
3. Tool remembers their choices for future sessions

## Updating Demo Data

To include different demo data files before packaging:

1. Replace files in the `data/` directory structure:
   - `data/train_data/Combined_Training31-Aug-2023.csv`
   - `data/val_data/raw_data/[your_validation_file].csv`
   - `data/test_data/raw_data/[your_test_file].csv`
   - `data/Combined_Testing31-Aug-2023.csv`

2. Update default hyperparameters in `config_manager.py` if needed:
   ```python
   "FEATURE_COLUMNS": ["SOC", "Current", "Temp"],  # Match your data columns
   "TARGET_COLUMN": "Voltage",
   ```

3. Run the build process:
   ```bash
   python build_exe.py
   # or
   build.bat
   ```

## Build Process

The `build_exe.py` script automatically:

1. **Prepares demo data**: Copies files from `data/` to `installer_assets/default_data/`
2. **Builds executable**: Uses PyInstaller to create standalone .exe
3. **Includes assets**: Bundles demo data files into the executable
4. **Creates installer**: Packages everything into a Windows installer

## Benefits

- **Zero configuration**: Users can test immediately after installation
- **Clear expectations**: Demo data shows what file format and structure is expected
- **Professional experience**: Tool works out-of-the-box
- **Educational**: Users can learn the workflow with working data
- **Flexible**: Easy to switch to custom data once familiar

## Installation Directory vs User Directory

- **Installation directory**: Contains the application and bundled demo files (read-only)
- **User directory**: Contains working copies of demo data (user can modify)
- **Settings persistence**: User preferences saved in user directory, not installation directory

This separation ensures:
- Multiple users can use the same installation
- User data persists across updates
- Clean uninstallation (user data remains if desired)
