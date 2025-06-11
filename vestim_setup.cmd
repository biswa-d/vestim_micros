@echo off
REM Install and setup script for VEstim
REM This script installs VEstim in development mode and provides options to run components

echo VEstim Setup Script
echo =================

if "%1"=="" (
    goto :help
)

if "%1"=="install" (
    echo Installing VEstim in development mode...
    pip install -e .
    echo Installation complete.
    goto :eof
)

if "%1"=="server" (
    echo Starting VEstim server...
    vestim server
    goto :eof
)

if "%1"=="gui" (
    echo Starting VEstim GUI...
    vestim gui
    goto :eof
)

if "%1"=="all" (
    echo Starting VEstim (server and GUI)...
    vestim all
    goto :eof
)

if "%1"=="stop" (
    echo Stopping VEstim server...
    vestim stop
    goto :eof
)

if "%1"=="status" (
    echo Checking VEstim status...
    vestim status
    goto :eof
)

:help
echo Usage: vestim_setup.cmd [command]
echo Commands:
echo   install - Install VEstim in development mode
echo   server  - Start the VEstim server
echo   gui     - Start the VEstim GUI
echo   all     - Start both server and GUI
echo   stop    - Stop the running server
echo   status  - Check server status
echo.
echo Example: vestim_setup.cmd install

:eof
