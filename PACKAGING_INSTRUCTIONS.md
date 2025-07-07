# Python Desktop Application Packaging Instructions

This document explains how to implement the complete packaging system used in Vestim for creating professional Windows installers from Python applications.

## 📦 Overview

The packaging system creates:
1. **Standalone executable** (.exe) using PyInstaller
2. **Professional Windows installer** (.exe) using Inno Setup
3. **User-configurable project directories** via registry storage
4. **Virtual environment isolation** for clean builds

## 🏗️ Required Project Structure

```
your_project/
├── build.bat                    # Main build script
├── build_exe.py                 # PyInstaller configuration
├── build_requirements.txt       # Build dependencies
├── requirements.txt             # Application dependencies
├── your_app_installer.iss       # Inno Setup script
├── your_app/                    # Main application package
│   ├── __init__.py
│   ├── config.py               # Registry-aware configuration
│   └── gui/
│       └── src/
│           └── main_gui.py     # Main GUI entry point
└── installer_output/            # Generated installer location
```

## 📋 Step-by-Step Implementation

### 1. Create Build Requirements File

**File: `build_requirements.txt`**
```txt
# Requirements for building the standalone installer
pyinstaller>=5.0
setuptools>=45.0
wheel>=0.37.0

# Optional installer tools (install these manually)
# Inno Setup: https://jrsoftware.org/isinfo.php
# NSIS: https://nsis.sourceforge.io/Download
```

### 2. Create PyInstaller Build Script

**File: `build_exe.py`**
```python
import PyInstaller.__main__
import os
import sys

# Add your main script path here
MAIN_SCRIPT = "your_app/gui/src/main_gui.py"  # Change this to your entry point
APP_NAME = "YourApp"  # Change this to your app name

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# PyInstaller arguments
args = [
    '--onefile',
    '--windowed',
    '--name', APP_NAME,
    '--distpath', os.path.join(script_dir, 'dist'),
    '--workpath', os.path.join(script_dir, 'build'),
    '--specpath', os.path.join(script_dir, 'build'),
    
    # Add your app icon (optional)
    # '--icon', 'your_app/gui/resources/icon.ico',
    
    # Add hidden imports if needed
    # '--hidden-import', 'your_module',
    
    # Add data files if needed
    # '--add-data', 'your_app/gui/resources;resources',
    
    # Clean build
    '--clean',
    
    # The main script
    MAIN_SCRIPT
]

print(f"Building {APP_NAME} with PyInstaller...")
print(f"Main script: {MAIN_SCRIPT}")
print(f"Build arguments: {' '.join(args)}")

# Run PyInstaller
PyInstaller.__main__.run(args)

print(f"\n✅ Build complete! Executable created at: dist/{APP_NAME}.exe")
```

### 3. Create Main Build Script

**File: `build.bat`**
```batch
@echo off
title YourApp Standalone Installer Builder
echo ========================================
echo YourApp Standalone Installer Builder  
echo ========================================
echo.
echo This script will create a standalone Windows
echo installer (.exe) for YourApp that users can
echo simply download and run.
echo.
pause

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Step 1: Installing build dependencies...
python -m pip install --upgrade pip pyinstaller

echo.
echo Step 2: Building standalone executable...
python build_exe.py

if not exist "dist\YourApp.exe" (
    echo Error: Failed to create executable
    pause
    exit /b 1
)

echo ✓ Executable created: dist\YourApp.exe

echo.
echo Step 3: Creating Windows installer...
echo.
echo NOTE: This requires Inno Setup to be installed.
echo Download from: https://jrsoftware.org/isinfo.php
echo.

REM Check if Inno Setup is available
where iscc >nul 2>&1
if errorlevel 1 (
    echo Inno Setup not found in PATH.
    echo.
    echo Manual steps:
    echo 1. Install Inno Setup from https://jrsoftware.org/isinfo.php
    echo 2. Open your_app_installer.iss in Inno Setup
    echo 3. Click Build to create the installer
    echo.
    pause
    exit /b 0
)

echo Inno Setup found! Creating installer...
iscc your_app_installer.iss

if exist "installer_output\your-app-installer-1.0.0.exe" (
    echo.
    echo ========================================
    echo SUCCESS! Installer created!
    echo ========================================
    echo.
    echo Installer location: installer_output\your-app-installer-1.0.0.exe
    echo.
    echo You can now:
    echo 1. Test the installer on this machine
    echo 2. Copy it to other Windows machines for testing
    echo 3. Distribute it to end users
    echo.
    echo The installer is completely standalone and includes
    echo everything needed to run YourApp.
) else (
    echo Error: Failed to create installer
    pause
    exit /b 1
)

echo.
pause
```

### 4. Create Registry-Aware Configuration

**File: `your_app/config.py`**
```python
import os
import sys
from pathlib import Path

def get_project_directory():
    """Get the configured project directory, checking registry on Windows if frozen executable"""
    if getattr(sys, 'frozen', False):
        # If bundled with PyInstaller, check Windows registry for configured path
        if sys.platform == 'win32':
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\YourApp")  # Change YourApp
                projects_dir, _ = winreg.QueryValueEx(key, "ProjectsDirectory")
                winreg.CloseKey(key)
                return Path(projects_dir)
            except (ImportError, FileNotFoundError, OSError):
                # Fall back to exe directory if registry not found
                pass
        
        # Default fallback: exe directory
        exe_dir = Path(sys.executable).parent
        return exe_dir / "YourApp_Projects"  # Change YourApp_Projects
    else:
        # If running from source, use the project root
        source_dir = Path(__file__).parent.parent
        return source_dir / "YourApp_Projects"  # Change YourApp_Projects

def get_root_dir():
    """Get the root directory, using current working directory for PyInstaller compatibility"""
    return os.getcwd()

def get_output_dir():
    """Get the output directory, dynamically based on current working directory"""
    return os.path.join(get_root_dir(), 'output')

# For backward compatibility, provide the variables
ROOT_DIR = get_root_dir()
OUTPUT_DIR = get_output_dir()
```

### 5. Create Inno Setup Script

**File: `your_app_installer.iss`**
```inno
; Inno Setup Script for YourApp
; This creates a professional Windows installer

#define MyAppName "YourApp"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Name"
#define MyAppURL "https://github.com/yourusername/yourapp"
#define MyAppExeName "YourApp.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Generate a new GUID for your application
AppId={{12345678-1234-1234-1234-123456789ABC}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Custom variables for project directory
UsedUserAreasWarning=no
LicenseFile=LICENSE
InfoBeforeFile=INSTALL_INFO.txt
OutputDir=installer_output
OutputBaseFilename=your-app-installer-{#MyAppVersion}
SetupIconFile=your_app\gui\resources\icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
Source: "dist\YourApp.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "your_app\gui\resources\*"; DestDir: "{app}\resources"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
; Note: We don't auto-delete the project directory as it contains user data

[Registry]
Root: HKCU; Subkey: "Software\{#MyAppName}"; ValueType: string; ValueName: "ProjectsDirectory"; ValueData: "{code:GetProjectDir}\YourApp_Projects"

[Code]
var
  ProjectDirPage: TInputDirWizardPage;

procedure InitializeWizard;
begin
  { Create a custom page for project directory selection }
  ProjectDirPage := CreateInputDirPage(wpSelectComponents,
    'Select Projects Directory', 'Where should YourApp store your projects?',
    'YourApp will create a "YourApp_Projects" folder in the directory you specify below. ' +
    'This is where all your data, models, and analysis results will be stored.' + #13#13 +
    'You can choose any location that has sufficient space (recommended: 5GB+).',
    False, '');
  ProjectDirPage.Add('');
  
  { Set default project directory to user's Documents folder }
  ProjectDirPage.Values[0] := ExpandConstant('{userdocs}');
end;

function GetProjectDir(Param: String): String;
begin
  Result := ProjectDirPage.Values[0];
end;

procedure CreateProjectDir();
var
  ProjectPath: String;
begin
  ProjectPath := ProjectDirPage.Values[0] + '\YourApp_Projects';
  if not DirExists(ProjectPath) then
  begin
    if not CreateDir(ProjectPath) then
      MsgBox('Warning: Could not create project directory at: ' + ProjectPath + #13#13 +
             'You may need to create it manually or choose a different location.', 
             mbInformation, MB_OK);
  end;
end;

function GetUninstallString(): String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade(): Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
  Result := 0;
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  end;
  
  if (CurStep=ssPostInstall) then
  begin
    { Create the project directory }
    CreateProjectDir();
    
    { Show completion message with project directory info }
    MsgBox('Installation Complete!' + #13#13 +
           'YourApp has been installed to: ' + ExpandConstant('{app}') + #13#13 +
           'Your projects will be stored in: ' + ProjectDirPage.Values[0] + '\YourApp_Projects' + #13#13 +
           'You can change the project directory later in YourApp''s settings.', 
           mbInformation, MB_OK);
  end;
end;
```

### 6. Update Main GUI to Use Configuration

**In your main GUI file (e.g., `your_app/gui/src/main_gui.py`):**
```python
import os
import sys
from pathlib import Path
import multiprocessing

def main():
    # Essential for PyInstaller multiprocessing support
    multiprocessing.freeze_support()
    
    # Get the configured project directory
    from your_app.config import get_project_directory
    project_dir = get_project_directory()
    
    # Create project directory if it doesn't exist
    project_dir.mkdir(exist_ok=True)
    
    # Change working directory to project directory
    os.chdir(project_dir)
    
    # Continue with your application initialization...
    # app = QApplication(sys.argv)
    # main_window = YourMainWindow()
    # main_window.show()
    # sys.exit(app.exec_())

if __name__ == '__main__':
    main()
```

### 7. Create Installation Info File

**File: `INSTALL_INFO.txt`**
```txt
Welcome to YourApp Setup
========================

YourApp is a [description of your application].

What this installer will do:
---------------------------

1. Install YourApp application to your computer
2. Create desktop shortcut (optional)
3. Create Start Menu entry
4. Allow you to choose where your projects are stored

After installation:
------------------

• Click the YourApp desktop icon or Start Menu entry to launch
• The application will use the project directory you selected
• YourApp will automatically create job folders to save your work
• No additional setup required - everything is self-contained

System Requirements:
-------------------

• Windows 10 or later (64-bit)
• At least 2GB of free disk space
• 4GB RAM recommended

For support, contact: your-email@example.com

Click Next to continue with the installation.
```

## 🔧 Customization Points

### Change These Values for Your Application:

1. **Application Name**: Replace `YourApp` throughout all files
2. **Project Directory**: Replace `YourApp_Projects` with your preferred folder name
3. **Main Script**: Update `MAIN_SCRIPT` in `build_exe.py` to your entry point
4. **App ID**: Generate new GUID for Inno Setup script
5. **Publisher Info**: Update publisher name, email, and URL
6. **Icon**: Add your application icon and update paths

### Generate New App ID:
```python
import uuid
print(str(uuid.uuid4()).upper())
```

## 🚀 Usage Instructions

### For the Developer:
```bash
# 1. Create virtual environment
python -m venv build_env
build_env\Scripts\activate

# 2. Install dependencies
pip install -r build_requirements.txt
pip install -r requirements.txt

# 3. Run build
build.bat
```

### For End Users:
1. Run `your-app-installer-1.0.0.exe`
2. Choose project directory during installation
3. Launch application from desktop shortcut

## 📋 Checklist

- [ ] Replace all `YourApp` placeholders with your app name
- [ ] Update main script path in `build_exe.py`
- [ ] Generate new GUID for Inno Setup
- [ ] Update publisher information
- [ ] Add your application icon
- [ ] Test build process in clean environment
- [ ] Test installer on different Windows versions
- [ ] Verify project directory selection works
- [ ] Confirm registry storage functions properly

## 🔍 Troubleshooting

- **PyInstaller Issues**: Check hidden imports, add `--debug` flag
- **Missing Files**: Use `--add-data` for additional resources
- **Registry Problems**: Ensure correct app name in registry key
- **Installer Fails**: Check Inno Setup syntax, verify file paths

This system provides a professional, user-friendly installation experience with configurable project directories and proper Windows integration.
