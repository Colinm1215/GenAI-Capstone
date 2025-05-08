@echo off

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "PATH=%CUDA_PATH%\bin;%PATH%"

set "VENV=C:\Users\Colin\PycharmProjects\pythonProject\.venv"

cd /d "%~dp0%"

set PYTHONPATH=%~dp0%

call "%VENV%\Scripts\activate.bat"

python testing\evaluation.py
