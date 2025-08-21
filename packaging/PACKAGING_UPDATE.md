# Packaging Update for Hyperparams Templates

## Changes Made

### 1. Directory Structure
- **Development**: `defaults_templates/` folder in repository root
- **Installed**: `default_hyperparams/` folder in user's project directory

### 2. ConfigManager Updates
- Added `get_defaults_directory()` method
- Auto-creates `default_hyperparams` directory during installation
- Copies template files from `defaults_templates` to user directory

### 3. Installer Updates (`vestim_installer.iss`)
- Added `defaults_templates` to [Files] section
- Creates `default_hyperparams` folder in user's project directory
- Copies template files during installation

### 4. GUI Updates
- "Load Parameters from File" dialog now opens `default_hyperparams` directory
- Auto-populated with template files on first use

## Project Structure After Installation

```
User's Projects Directory/
├── vestim_projects/
│   ├── data/
│   │   ├── train_data/       # Demo training files
│   │   ├── val_data/         # Demo validation files
│   │   └── test_data/        # Demo test files
│   ├── default_hyperparams/  # Template hyperparams files
│   │   ├── hyperparams_last_used.json
│   │   ├── optuna_hyperparams_lstm.json
│   │   ├── optuna_hyperparams_gru.json
│   │   ├── optuna_hyperparams_fnn.json
│   │   ├── grid_hyperparams_lstm.json
│   │   ├── grid_hyperparams_gru.json
│   │   └── grid_hyperparams_fnn.json
│   ├── output/               # Training job outputs
│   └── default_settings.json # User preferences
```

## User Experience
1. Install Vestim
2. Launch application
3. Click "Load Parameters from File"
4. Automatically opens `default_hyperparams` folder with ready-to-use templates
5. Select appropriate template (LSTM/GRU/FNN, Grid/Optuna)
6. Customize parameters as needed
7. Proceed with training

This ensures users have immediate access to properly formatted hyperparameter templates without needing to understand the format differences between Grid Search and Optuna modes.
