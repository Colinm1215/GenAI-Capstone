@echo off

set CUDA_LAUNCH_BLOCKING=1
set TORCH_USE_CUDA_DSA=1


start cmd /k python backend.py
start cmd /k streamlit run frontend.py

pause