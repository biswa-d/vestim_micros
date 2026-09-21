@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Creating virtual environment ^(one-time setup^)...
    python -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install pandas numpy matplotlib scikit-learn scipy joblib
    .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo Setup done.
    echo.
)

echo [2/2] Running inference...
.venv\Scripts\python.exe run_offline_inference.py --output-dir inference_output %*

if %errorlevel% equ 0 (
    echo.
    echo Done. Results in inference_output\
) else (
    echo.
    echo FAILED.
    exit /b 1
)
