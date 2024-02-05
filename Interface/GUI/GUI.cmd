@echo off
REM Conf:
setlocal enabledelayedexpansion
TITLE Pneumonia-Detection-Ai-GUI
set python_min_VER=10
set DEBUG=0
set Full_Auto=1
set arg=%1
set PV_filepath="Data\\Python Ver.tmp"
set python_path=python
set pip_path=pip

REM Check if the fast start flag is used
if "%arg%"=="-f" (
    goto :FAST_START
)

REM Check if Python is installed
"%python_path%" --version 2>nul >nul
if errorlevel 1 goto :errorNoPython

@REM Geting the Python path and Python install time
for /f "delims=" %%p in ('where "%python_path%" 2^>^&1 ^| findstr /v "INFO:"') do (
    set "python_path_env=%%p"
)
for %%A in ("%python_path_env%") do (
    set Python_INSTALLTIME=%%~tA
)

REM Check if the Python version file exists and matches the current Python version
for /F "delims=" %%i IN ('"%python_path%" --version 2^>^&1') DO set current_python_version=%%i
set "current_python_version=%current_python_version%  %Python_INSTALLTIME%"
if not exist %PV_filepath% (
    goto :PASS_PVF_CHECK
)
set /p file_python_version=<%PV_filepath%
if "%file_python_version%"=="%current_python_version% " (
    goto :FAST_START
)

:PASS_PVF_CHECK
REM Write the current Python version to the file
echo Checking Python version...
REM Ensure Python version is %python_min_VER% or higher
for /F "tokens=2 delims=." %%i IN ('"%python_path%" --version 2^>^&1') DO set python_version_major=%%i
if %python_version_major% LSS %python_min_VER% (
    echo Warning: Please update your Python version to 3.%python_min_VER%.x or higher!
    pause
    exit /B
)

REM Check if the required packages are installed
echo Checking the required packages...
for /F "usebackq delims==" %%i in ("Data\requirements.txt") do (
    call :check_install %%i
)
REM Write the current Python version + Python install time to the file
echo %current_python_version% > %PV_filepath%
@REM Pause for user input
echo Press any key to load the GUI...
pause > nul

:FAST_START
REM Print the appropriate loading message
if "%arg%"=="-f" (
    echo Loading the GUI fast...
) else (
    echo Loading the GUI...
)

:restart
REM Clear the terminal and start the Python GUI script
timeout /t 1 >nul
cls
"%python_path%" "Data\GUI_main.py"

goto :EOF

REM errorNoPython
:errorNoPython
echo Error: Python is not installed
pause
goto :EOF

:check_install
REM Check if a package is installed and offer to install it if not
set userinput=Y
"%pip_path%" show %1 >nul
if ERRORLEVEL 1 (
    if not if "%Full_Auto%"=="1" (
    echo Package %1 not found. Do you want to automatically install it? [Y/n]
    set /p userinput="Answer: "
    )
    if /I "%userinput%"=="Y" or "%Full_Auto%"=="1"(
        echo Installing package %1
        "%pip_path%" install %1
        if ERRORLEVEL 1 (
            echo Failed to install package %1.
            exit /B
        )
    )
) else if "%DEBUG%"=="1" (
    echo Package %1 is already installed.
)
GOTO:EOF
