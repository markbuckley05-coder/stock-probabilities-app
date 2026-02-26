@echo off
setlocal
cd /d "%~dp0"

REM Create venv if missing
if not exist "stockapp_env\Scripts\python.exe" (
  python -m venv stockapp_env
)

call "stockapp_env\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

echo.
echo Environment ready.
pause
endlocal