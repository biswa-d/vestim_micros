# Project File Driven Launch Architecture Proposal

## Goal

Enable users to open a shared project file and have PyBattML launch with preloaded dataset paths and guided defaults across initial workflow screens.

## Intent Captured

When a user double-clicks a shared project file:

1. PyBattML should open directly.
2. Dataset root and split folders (train/val/test) should be preloaded.
3. Data import screen should be prepopulated (including file format and selected files where possible).
4. Data augmentation screen should open with beginner-friendly defaults pre-checked.
5. User can still modify any preloaded values.

## Proposed Artifacts

### 1) Project file (portable entrypoint)

- Extension: `.pbmlproj`
- Format: JSON
- Purpose: points to config payload and optionally controls startup behavior.

Example:

```json
{
  "schema_version": "1.0",
  "project_name": "Battery_Aging_Demo",
  "config": {
    "relative_path": "./project_config"
  },
  "startup": {
    "flow": "training",
    "auto_open_data_import": true,
    "auto_continue_to_augmentation": false
  }
}
```

### 2) Config directory (shareable defaults)

- Folder: `project_config/` (next to `.pbmlproj`)
- Files:
  - `dataset_defaults.json`
  - `augmentation_defaults.json`
  - (optional) `hyperparams_defaults.json`

Example `dataset_defaults.json`:

```json
{
  "schema_version": "1.0",
  "dataset_root": "./data",
  "splits": {
    "train": "./data/train_data",
    "val": "./data/val_data",
    "test": "./data/test_data"
  },
  "file_format": "csv",
  "auto_select_all_files": true
}
```

Example `augmentation_defaults.json`:

```json
{
  "schema_version": "1.0",
  "filtering": {
    "enabled": true,
    "configs": []
  },
  "normalization": {
    "enabled": true
  },
  "resampling": {
    "enabled": false,
    "target_hz": null
  },
  "padding": {
    "enabled": false,
    "length": 0
  },
  "noise": {
    "enabled": false,
    "configs": []
  },
  "column_creation": {
    "enabled": false,
    "formulas": []
  }
}
```

## Startup Flow

1. OS file association passes `.pbmlproj` path to launcher args.
2. Launcher parses project file and resolves relative paths from project file directory.
3. Launcher seeds runtime project context (in-memory singleton + environment variable).
4. `WelcomeGUI` receives context and routes to `DataImportGUI` for training flow.
5. `DataImportGUI` loads dataset defaults first, then legacy persisted defaults as fallback.
6. `DataAugmentGUI` receives augmentation defaults and applies checkbox/field values.

## Code Touchpoints

1. `launch_gui_qt.py`
   - Parse optional argument: `<path>.pbmlproj`
   - Validate schema and construct project context.

2. `vestim/config_manager.py`
   - Add project context load/resolve helpers.
   - Priority order update:
     1) explicit project file context
     2) env var config
     3) installed config json
     4) fallback defaults

3. `vestim/gui/src/welcome_gui_qt.py`
   - Accept startup context.
   - Optional auto-transition to training flow when project file is present.

4. `vestim/gui/src/data_import_gui_qt.py`
   - Apply `dataset_defaults.json` overrides.
   - Keep existing UI editable.

5. `vestim/gui/src/data_augment_gui_qt.py`
   - Add optional defaults payload parameter.
   - Set checkbox/value defaults from project config.

## Compatibility and Safety

- Backward compatible: app works unchanged without `.pbmlproj`.
- Path portability: always resolve relative paths against project file directory.
- Validation: missing folders should show non-blocking warnings and allow manual override.
- Schema versioning: reject unsupported versions with clear message.

## Suggested Phase Plan

1. MVP
   - Parse `.pbmlproj` in launcher
   - Preload Data Import folders and file format

2. Guided augmentation
   - Apply augmentation defaults in Data Augmentation GUI

3. OS integration
   - Installer writes Windows file association for `.pbmlproj`

