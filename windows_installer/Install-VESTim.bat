@echo off
:: VESTim Remote Launcher Setup
:: This batch file helps set up VESTim on a remote server with GUI forwarding
:: Author: Biswanath Dehury
:: Date: May 19, 2025

echo ==================================
echo  VESTim Remote Installation Tool
echo ==================================
echo.

:: Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell is required but not found.
    echo Please install PowerShell or use Windows 10/11 which includes it by default.
    pause
    exit /b 1
)

:: Launch PowerShell script with elevated privileges if needed
powershell -ExecutionPolicy Bypass -File "%~dp0Install-VESTim.ps1"

:: If PowerShell script exits successfully (0), offer to create desktop shortcut
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Installation completed successfully!
    echo.
    set /p CREATE_SHORTCUT="Would you like to create a desktop shortcut? (Y/N): "
    if /i "%CREATE_SHORTCUT%"=="Y" (
        powershell -ExecutionPolicy Bypass -Command "& {$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([System.Environment]::GetFolderPath('Desktop') + '\VESTim.lnk'); $Shortcut.TargetPath = '%~dp0Run-VESTim.bat'; $Shortcut.Save()}"
        echo Shortcut created on desktop.
    )
)

echo.
echo You can now launch VESTim by running Run-VESTim.bat
echo.
pause
