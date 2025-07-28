@echo off
echo ====================================
echo    AuroraQ 백테스트 시스템 v2
echo ====================================

:: Python 경로 확인
python --version
if %errorlevel% neq 0 (
    echo Python이 설치되지 않았거나 PATH에 없습니다.
    pause
    exit /b 1
)

:: 현재 디렉토리를 백테스트 디렉토리로 변경
cd /d "%~dp0"

:: 데이터 파일 확인
set PRICE_DATA=..\..\data\btc_5m_sample.csv
set SENTIMENT_DATA=..\..\data\sentiment_sample.csv

echo 데이터 파일 확인 중...
if not exist "%PRICE_DATA%" (
    echo 가격 데이터 파일이 없습니다: %PRICE_DATA%
    echo 샘플 데이터를 생성하거나 실제 데이터 경로를 지정하세요.
    pause
    exit /b 1
)

echo.
echo 백테스트 시작...
echo.

:: 백테스트 실행
python run_backtest.py ^
    --price-data "%PRICE_DATA%" ^
    --sentiment-data "%SENTIMENT_DATA%" ^
    --initial-capital 1000000 ^
    --window-size 100 ^
    --mode normal ^
    --enable-multiframe ^
    --output-dir "reports/backtest" ^
    --indicators sma_20 sma_50 ema_12 ema_26 rsi macd macd_line macd_signal macd_hist bbands bollinger bb_upper bb_middle bb_lower atr adx volatility

if %errorlevel% equ 0 (
    echo.
    echo ====================================
    echo     백테스트가 성공적으로 완료되었습니다!
    echo ====================================
    echo.
    echo 결과 보고서를 확인하세요: reports/backtest/
    echo.
) else (
    echo.
    echo ====================================
    echo        백테스트 실행 중 오류 발생
    echo ====================================
    echo.
)

pause