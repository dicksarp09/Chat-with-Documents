@echo off
echo Starting Document Intelligence Engine...
echo.

cd /d "%~dp0"

set GROQ_API_KEY=%GROQ_API_KEY%
if "%GROQ_API_KEY%"=="" (
    echo WARNING: GROQ_API_KEY environment variable not set.
    echo Set it with: set GROQ_API_KEY=your_api_key
    echo.
)

echo Installing dependencies if needed...
pip install -r requirements.txt -q

echo.
echo Starting FastAPI server...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
