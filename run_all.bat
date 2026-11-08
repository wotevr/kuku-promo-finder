@echo off
REM ==== Double-click me: runs the whole Kuku promo finder ====
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=python"
if exist "%USERPROFILE%\anacondaa\python.exe" set "PY=%USERPROFILE%\anacondaa\python.exe"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
echo Using Python: %PY%

if not exist .env (
  copy .env.example .env >nul
  echo.
  echo Notepad will open the file .env
  echo Paste your Gemini API key right after  LLM_API_KEY=   then SAVE and CLOSE Notepad.
  notepad .env
)

echo.
echo === Installing packages (only needed once) ===
"%PY%" -m pip install -q -r requirements.txt

echo.
echo === Checking your key and model (1 request) ===
"%PY%" check_setup.py
echo.
echo If the test call above WORKED: press any key to start the full run (about 5-10 minutes).
echo If it FAILED: close this window, fix .env (key or LLM_MODEL), and double-click again.
pause >nul

echo.
echo === Full run on all 50 episodes ===
"%PY%" run.py --input ./data --out outputs/
echo.
echo === Consistency check (2 more requests) ===
"%PY%" consistency.py --input ./data --out outputs --episodes 1-7
echo.
echo === Writing the top 20 clips for you to rate ===
"%PY%" show_clips.py --input ./data --out outputs
echo.
echo ================================================================
echo DONE. If you saw "STOPPED: quota", change LLM_MODEL in .env and double-click again
echo (finished work is reused). Otherwise go back to Claude and say: run finished
echo ================================================================
pause
