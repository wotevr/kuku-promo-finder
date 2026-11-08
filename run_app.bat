@echo off
REM Opens the promo finder as a web page in your browser (runs on this laptop, nothing is uploaded online)
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
echo Installing the web page package (first time only)...
"%PY%" -m pip install -q streamlit
echo.
echo Opening http://localhost:8501 in your browser. Keep this window open while you use the page.
"%PY%" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
