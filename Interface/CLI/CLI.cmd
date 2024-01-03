@echo off
TITLE Pneumonia AI CLI
set DEBUG=0
set arg=%1
if "%arg%"=="-f" (
    goto :FAST_START)
REM Check Python version
echo Checking Python version...
FOR /F "tokens=2 delims=." %%i IN ('python --version 2^>^&1') DO set python_version=%%i
if %python_version% LSS 9 (
    echo Warning: Please update your Python version to 3.9.x or higher!
    pause
    exit /B
)

REM Verify required libraries
echo Checking the required packages...
for /F "usebackq delims==" %%i in ("Data\requirements.txt") do (
call :check_install %%i
)

echo Press any key to load the CLI...
pause > nul
:FAST_START
REM Print loading message
if "%arg%"=="-f" (
    echo Loading the CLI fast...
) else (
    echo Loading the CLI...
)
:restart
REM Clear the terminal
timeout /t 1 >nul
cls
REM Start the Python script
python "Data\CLI_main.py"

set /p restart="Do you want to restart the CLI or quit the CLI (y/n)? "
if /i "%restart%"=="y" (
    goto :restart
) else (
    goto :EOF
)


:check_install
pip show %1 >nul
if ERRORLEVEL 1 (
    echo Package %1 not found. Do you want to automatically install it? [Y/n]
)
set /p userinput="Answer: "
if /I "%userinput%"=="Y" (
    echo Installing package %1
    pip install %1
    if ERRORLEVEL 1 (
        echo Failed to install package %1.
        exit /B
    )
) else if "%DEBUG%"=="1" (
    echo Package %1 is already installed.
)
GOTO:EOF
