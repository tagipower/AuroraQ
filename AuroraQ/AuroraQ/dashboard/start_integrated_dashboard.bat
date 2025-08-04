@echo off
echo ========================================
echo AuroraQ Integrated Monitoring Dashboard
echo ========================================
echo.

echo [1/3] Checking Python environment...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

echo [2/3] Installing required packages...
pip install fastapi uvicorn websockets psutil scikit-learn numpy

echo [3/3] Starting Integrated Dashboard...
echo.
echo Dashboard will be available at:
echo   Web Interface: http://localhost:8080
echo   API Docs: http://localhost:8080/docs
echo.
echo Press Ctrl+C to stop the dashboard
echo.

python integrated_monitoring_system.py

pause