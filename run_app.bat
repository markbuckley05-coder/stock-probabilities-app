@echo off
setlocal
cd /d "%~dp0"

if not exist "stockapp_env\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run create_env.bat first.
  pause
  exit /b 1
)

call "stockapp_env\Scripts\activate.bat"

python "app\pro_stock_probabilities.py"

pause
endlocal