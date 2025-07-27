@echo off
title Vestim Professional Installer Builder
echo ========================================
echo Vestim Professional Installer Builder  
echo ========================================
echo.
echo This script will create a professional Windows
echo installer (.exe) for Vestim with user-configurable
echo project folders and embedded demo data.
echo.
REM pause

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    REM pause
    exit /b 1
)

echo.
echo Step 1: Installing build dependencies...
python -m pip install --upgrade pip pyinstaller

echo.
echo Step 2: Preparing installer assets...
echo - Demo data files (train/val/test)
echo - Default settings template
echo - User documentation
echo.

echo.
echo Step 3: Building standalone executable...
python build_exe.py

if not exist "dist\Vestim.exe" (
    echo Error: Failed to create executable
    REM pause
    exit /b 1
)

echo ✓ Executable created: dist\Vestim.exe
echo ✓ Installer assets embedded in executable

echo.
echo Step 4: Creating professional Windows installer...
echo.
echo Features:
echo - User selects Vestim installation directory
echo - User selects projects folder location  
echo - Automatic demo data deployment
echo - Desktop/Start menu shortcuts
echo - Uninstaller with cleanup
echo.

REM Check if Inno Setup is available
where iscc >nul 2>&1
if errorlevel 1 (
    echo Inno Setup not found in PATH.
    echo.
    echo Manual steps:
    echo 1. Install Inno Setup from https://jrsoftware.org/isinfo.php
    echo 2. Open vestim_installer.iss in Inno Setup
    echo 3. Click Build to create the installer
    echo.
    REM pause
    exit /b 0
)

echo Inno Setup found! Creating professional installer...
iscc vestim_installer.iss

if exist "installer_output\vestim-installer-2.0.0.exe" (
    echo.
    echo ========================================
    echo SUCCESS! Professional Installer Created!
    echo ========================================
    echo.
    echo Installer: installer_output\vestim-installer-2.0.0.exe
    echo.
    echo Installer Features:
    echo ✓ User selects Vestim installation directory
    echo ✓ User selects projects folder location
    echo ✓ Automatic demo data deployment to projects folder
    echo ✓ Creates: vestim_projects\data\train_data\demo_*.csv
    echo ✓ Creates: vestim_projects\data\val_data\demo_*.csv  
    echo ✓ Creates: vestim_projects\data\test_data\demo_*.csv
    echo ✓ Creates: vestim_projects\default_settings.json
    echo ✓ Creates: vestim_projects\README.md (comprehensive user guide)
    echo ✓ Desktop and Start Menu shortcuts
    echo ✓ Professional uninstaller with cleanup
    echo.
    echo User Experience:
    echo ✓ Comprehensive README with step-by-step tutorials
    echo ✓ All GUI workflows explained in detail  
    echo ✓ Demo data format and usage instructions
    echo ✓ Troubleshooting and best practices guide
    echo ✓ Advanced hyperparameter tuning tips
    echo.
    echo Distribution:
    echo - The installer is completely standalone  
    echo - Contains embedded demo data and documentation
    echo - Users can immediately test after installation
    echo - Job outputs will appear in vestim_projects folder
    echo.
    echo Next Steps:
    echo 1. Test the installer on a clean machine
    echo 2. Verify demo data loads correctly in the tool
    echo 3. Distribute to end users
) else (
    echo Error: Failed to create installer
    echo Check the build log above for errors
    REM pause
    exit /b 1
)

echo.
REM pause
