# Building Vestim Standalone Installer

This guide explains how to create a standalone Windows installer for Vestim.

## Overview

The build process creates:
1. **Vestim.exe** - Standalone executable (no Python required)
2. **vestim-installer-1.0.0.exe** - Professional Windows installer

## Prerequisites

### Required Tools:
1. **Python 3.8+** with Vestim dependencies installed
2. **PyInstaller** for creating executables
3. **Inno Setup** for creating the installer (recommended)
   - Download from: https://jrsoftware.org/isinfo.php
   - Alternative: **NSIS** from https://nsis.sourceforge.io/

## Quick Build (Windows)

1. **Simple build** (double-click):
   ```
   build.bat
   ```

2. **Manual build**:
   ```cmd
   python -m pip install pyinstaller
   python build_exe.py
   iscc vestim_installer.iss
   ```

## What the Installer Does

When users run `vestim-installer-1.0.0.exe`:

1. **Installation Process**:
   - Installs to `C:\Program Files\Vestim\`
   - Creates desktop shortcut
   - Creates Start Menu entry
   - Sets up file associations

2. **User Experience**:
   - Double-click installer → Follow prompts → Done
   - Launch from desktop shortcut or Start Menu
   - Application asks for train/test folders
   - Creates job folders automatically

## Distribution

### For End Users:
- Share: `installer_output/vestim-installer-1.0.0.exe`
- Size: ~200MB (includes all dependencies)
- Requirements: Windows 10+ (64-bit)

### Testing:
```cmd
# Test the standalone exe:
dist\Vestim.exe

# Test the installer:
installer_output\vestim-installer-1.0.0.exe
```

## File Structure After Build

```
project/
├── dist/
│   └── Vestim.exe                    # Standalone executable
├── installer_output/
│   └── vestim-installer-1.0.0.exe    # Final installer
├── build/                            # PyInstaller build files
└── vestim_installer.iss              # Inno Setup script
```

## Customization

### Changing App Info:
Edit `vestim_installer.iss`:
```pascal
#define MyAppName "Vestim"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Name"
```

### Adding Files:
Edit the `[Files]` section in `vestim_installer.iss`:
```pascal
Source: "your_file.txt"; DestDir: "{app}"; Flags: ignoreversion
```

## Troubleshooting

### PyInstaller Issues:
- **Missing modules**: Add `--hidden-import=module_name` in build_exe.py
- **Large size**: Use `--exclude-module=unused_module`
- **Slow startup**: Consider `--onedir` instead of `--onefile`

### Installer Issues:
- **Icon missing**: Ensure `vestim/gui/resources/icon.ico` exists
- **File permissions**: Run Inno Setup as administrator if needed

### Runtime Issues:
- **Path problems**: The app automatically handles bundled execution
- **Missing resources**: Check that all files are included in PyInstaller spec

## Advanced Options

### Using NSIS instead of Inno Setup:
```cmd
makensis vestim_installer.nsi
```

### Creating portable version:
Edit `build_exe.py` and change to `--onedir` for a folder-based distribution.

## Support

For build issues, check:
1. Python and pip are up to date
2. All dependencies are installed
3. Build tools are in PATH
4. Sufficient disk space (2GB recommended)

The final installer is completely self-contained and can be distributed to any Windows 10+ machine.
