@echo off
del requirements.txt >nul 2>&1
echo Y | pigar -l ERROR generate

