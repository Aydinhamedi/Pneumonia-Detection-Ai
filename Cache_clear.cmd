@echo off
setlocal

set "folder=cache"

for /d %%a in ("%folder%\*") do (
    rd /s /q "%%~fa"
)

for %%a in ("%folder%\*") do (
    del /f /q "%%~fa"
)

endlocal
