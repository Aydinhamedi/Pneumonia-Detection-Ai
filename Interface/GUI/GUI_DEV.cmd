@echo off
:restart
REM Clear the terminal and start the Python CLI script
cls

cd "C:\Users\aydin\Desktop\Pneumonia AI Dev\Interface\GUI"

python "Data\GUI_main.py"

REM Prompt to restart or quit the CLI
set /p restart="Do you want to restart the GUI or quit the GUI (y/n)? "
if /i "%restart%"=="y" (
    goto :restart
) else (
    goto :EOF
)
