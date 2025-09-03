e@echo on
set VESTIM_ROOT=%~1
set PYTHON_INSTALLER=python_installer\python-3.12.4-amd64.exe
set PYTHON_DIR=%VESTIM_ROOT%python
set REQUIREMENTS_FILE=requirements_cpu.txt

echo --- Vestim Environment Setup ---
echo VESTIM_ROOT is %VESTIM_ROOT%
echo PYTHON_INSTALLER is %PYTHON_INSTALLER%
echo PYTHON_DIR is %PYTHON_DIR%
echo REQUIREMENTS_FILE is %REQUIREMENTS_FILE%
echo.

echo Installing Python...
"%PYTHON_INSTALLER%" /passive InstallAllUsers=0 TargetDir="%PYTHON_DIR%" PrependPath=0 Include_test=0
echo Python installer finished with exit code: %errorlevel%
pause
echo.

if not exist "%PYTHON_DIR%\python.exe" (
    echo ERROR: Python.exe not found after installation.
    pause
    exit /b 1
)

echo Installing required packages...
"%PYTHON_DIR%\Scripts\pip.exe" install --no-cache-dir -r "%REQUIREMENTS_FILE%"
echo Pip install finished with exit code: %errorlevel%
pause
echo.

echo Environment setup complete.
pause
exit /b 0
