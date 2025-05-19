@echo off
:: VESTim Remote Launcher
:: This batch file launches VESTim on a remote server with GUI forwarding

echo Starting VESTim...

:: Check if config file exists
if not exist "%~dp0server_config.txt" (
    echo Error: Server configuration not found.
    echo Please run Install-VESTim.bat first.
    pause
    exit /b 1
)

:: Read server configuration
for /f "tokens=1,* delims==" %%a in (%~dp0server_config.txt) do (
    if "%%a"=="Host" set SERVER_HOST=%%b
    if "%%a"=="Username" set SERVER_USER=%%b
    if "%%a"=="Port" set SERVER_PORT=%%b
)

:: Get local IP address
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set LOCAL_IP=%%a
    set LOCAL_IP=!LOCAL_IP:~1!
    goto :got_ip
)
:got_ip

:: Start X Server
echo Starting X Server...
start "" "%ProgramFiles%\VcXsrv\vcxsrv.exe" -multiwindow -clipboard -wgl -ac

:: Wait for X server to start
timeout /t 2 > nul

:: SSH connection with X11 forwarding
echo Connecting to %SERVER_HOST% as %SERVER_USER%...
set DISPLAY=%LOCAL_IP%:0.0
ssh -X -o "StrictHostKeyChecking=no" -p %SERVER_PORT% %SERVER_USER%@%SERVER_HOST% "export DISPLAY=%LOCAL_IP%:0.0 && vestim-gui"

:: Clean up
echo Closing X Server...
taskkill /f /im vcxsrv.exe > nul 2>&1

echo VESTim session ended.
pause
