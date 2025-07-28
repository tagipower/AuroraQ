#!/bin/bash

echo "===================================="
echo "   AuroraQ 백테스트 시스템 v2"
echo "===================================="

# Python 확인
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Python이 설치되지 않았습니다."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "Python 버전:"
$PYTHON_CMD --version

# 백테스트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 데이터 파일 경로
PRICE_DATA="../../data/btc_5m_sample.csv"
SENTIMENT_DATA="../../data/sentiment_sample.csv"

echo "데이터 파일 확인 중..."
if [ ! -f "$PRICE_DATA" ]; then
    echo "가격 데이터 파일이 없습니다: $PRICE_DATA"
    echo "샘플 데이터를 생성하거나 실제 데이터 경로를 지정하세요."
    exit 1
fi

echo ""
echo "백테스트 시작..."
echo ""

# 백테스트 실행
$PYTHON_CMD run_backtest.py \
    --price-data "$PRICE_DATA" \
    --sentiment-data "$SENTIMENT_DATA" \
    --initial-capital 1000000 \
    --window-size 100 \
    --mode normal \
    --enable-multiframe \
    --output-dir "reports/backtest" \
    --indicators sma_20 sma_50 ema_12 ema_26 rsi macd macd_line macd_signal macd_hist bbands bollinger bb_upper bb_middle bb_lower atr adx volatility

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================="
    echo "   백테스트가 성공적으로 완료되었습니다!"
    echo "===================================="
    echo ""
    echo "결과 보고서를 확인하세요: reports/backtest/"
    echo ""
else
    echo ""
    echo "===================================="
    echo "      백테스트 실행 중 오류 발생"
    echo "===================================="
    echo ""
fi

read -p "아무 키나 누르세요..."