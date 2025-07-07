@echo off
title Vestim Standalone Installer Builder
echo ========================================
echo Vestim Standalone Installer Builder  
echo ========================================
echo.
echo This script will create a standalone Windows
echo installer (.exe) for Vestim that users can
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

if not exist "dist\Vestim.exe" (
    echo Error: Failed to create executable
    pause
    exit /b 1
)

echo ✓ Executable created: dist\Vestim.exe

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
    echo 2. Open vestim_installer.iss in Inno Setup
    echo 3. Click Build to create the installer
    echo.
    pause
    exit /b 0
)

echo Inno Setup found! Creating installer...
iscc vestim_installer.iss

if exist "installer_output\vestim-installer-1.0.0.exe" (
    echo.
    echo ========================================
    echo SUCCESS! Installer created!
    echo ========================================
    echo.
    echo Installer location: installer_output\vestim-installer-1.0.0.exe
    echo.
    echo You can now:
    echo 1. Test the installer on this machine
    echo 2. Copy it to other Windows machines for testing
    echo 3. Distribute it to end users
    echo.
    echo The installer is completely standalone and includes
    echo everything needed to run Vestim.
) else (
    echo Error: Failed to create installer
    pause
    exit /b 1
)

echo.
pause
