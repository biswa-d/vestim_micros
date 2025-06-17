@echo off
echo Starting VEstim application...

rem Use the local Python interpreter to start the application
python start_vestim.py

rem If the application exits with an error, keep the window open
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: VEstim exited with error code %ERRORLEVEL%
    echo Press any key to close this window...
    pause > nul
)
