@echo off
setlocal
cd /d "%~dp0"
set "VESTIM_PROJECT_DIR=%~dp0"
"%~dp0build_env\Scripts\python.exe" "%~dp0launch_gui_qt.py" %*
