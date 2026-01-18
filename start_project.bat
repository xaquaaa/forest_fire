@echo off
echo ==============================
echo Starting Forest Fire Project
echo ==============================

REM ---- Set Conda ----
call C:\Users\ALIENWARE\anaconda3\condabin\conda.bat activate base

echo Starting Phase 1 backend...
start cmd /k "call C:\Users\ALIENWARE\anaconda3\condabin\conda.bat activate base && python app.py"

echo Starting Phase 1 frontend...
start cmd /k "call C:\Users\ALIENWARE\anaconda3\condabin\conda.bat activate base && python -m http.server 8000"

echo Starting Phase 2 Simulation backend...
start cmd /k "call C:\Users\ALIENWARE\anaconda3\condabin\conda.bat activate base && python simulation4_react+flask/backend/app.py"

echo ==============================
echo All services started
echo ==============================
pause
