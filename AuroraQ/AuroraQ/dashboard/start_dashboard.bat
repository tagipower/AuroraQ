@echo off
echo ====================================================
echo   AuroraQ Dashboard - Starting
echo ====================================================

echo Checking API connection...
curl -s http://localhost:8001/onnx/metrics >nul
if errorlevel 1 (
  echo WARNING: Metrics API not responding on port 8001
  echo Make sure to start the API server first:
  echo   cd ../sentiment-service
  echo   python -m uvicorn api.metrics_router:app --host 0.0.0.0 --port 8001
  pause
)

echo Starting AuroraQ Dashboard...
python aurora_dashboard_final.py
pause