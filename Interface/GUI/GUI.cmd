@echo off
REM Conf:
setlocal enabledelayedexpansion
TITLE Pneumonia-Detection-Ai-GUI-Launcher
set python_min_VER=10
set python_max_VER=10
set DEBUG=0
set Full_Auto=1
set arg=%1
set PV_filepath="Data\\Python Ver.tmp"
set python_path=python
set file_path="Data\GUI_main"
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
    echo Warning: Please uninstall python and install Python 3.%python_max_VER%.x ^(Ver too low^)
    pause
    exit /B
) else if %python_version_major% GTR %python_max_VER% (
    echo Warning: Please uninstall python and install Python 3.%python_max_VER%.x ^(Ver too high^)
    pause
    exit /B
)

REM Check if the required packages are installed
echo Checking the required packages...
for /F "usebackq delims=" %%i in ("Data\requirements.txt") do (
    call :check_install %%i
)
REM Write the current Python version + Python install time to the file
echo %current_python_version% > %PV_filepath%
@REM Pause for user input
if not "%Full_Auto%"=="1" (
    echo Press any key to load the GUI...
    pause > nul
)

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
if exist %file_path%.py (
    echo Python file found, attempting to run...
    %python_path% %file_path%.py
    if %errorlevel% neq 0 (
        echo Error running Python file.
        pause
    )
) else if exist %file_path%.pyc (
    echo Compiled Python file found, attempting to run...
    %python_path% %file_path%.pyc
    if %errorlevel% neq 0 (
        echo Error running compiled Python file.
        pause
    )
) else (
    echo Neither Python nor compiled Python file found.
    pause
)

goto :EOF

REM errorNoPython
:errorNoPython
echo Error: Python is not installed
pause
goto :EOF

:check_install
REM Check if a package is installed and offer to install it if not
set userinput=Y
set "P_name=%~1==%~2"
"%pip_path%" show %~1 >nul 2>&1
if ERRORLEVEL 1 (
    if not "%Full_Auto%"=="1" (
        echo Package %P_name% not found. Do you want to automatically install it? [Y/n]
        set /p userinput="Answer: "
    )
    if /I "%userinput%"=="Y" (
        echo Installing package %P_name%
        "%pip_path%" install %P_name%
        if ERRORLEVEL 1 (
            echo Failed to install package %P_name%.
            Pause
            goto :EOF
        )
    )
) else if "%DEBUG%"=="1" (
    echo Package %P_name% is already installed.
)
GOTO:EOF