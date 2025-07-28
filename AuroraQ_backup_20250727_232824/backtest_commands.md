# AuroraQ 백테스트 실행 명령어 가이드

## 🚀 기본 사용법

### 1. 단일 백테스트 실행
```bash
# 기본 실행
python run_backtest.py --price-data data/price_data.csv

# 전체 옵션 포함
python run_backtest.py \
  --name "my_backtest" \
  --price-data data/price_data.csv \
  --sentiment-data data/sentiment_data.csv \
  --capital 1000000 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --window-size 100 \
  --indicators "sma_20,sma_50,rsi,macd" \
  --exploration
```

### 2. 설정 파일 사용
```bash
# 단일 백테스트
python run_backtest.py --config config/backtest_configs.json

# 다중 백테스트
python run_backtest.py --config config/backtest_configs.json --multiple

# 워크포워드 분석
python run_backtest.py --config config/backtest_configs.json --walk-forward
```

## 📊 실행 모드별 명령어

### 일반 모드
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --mode normal \
  --capital 1000000
```

### 탐색 모드 (다양한 전략 시도)
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --mode exploration \
  --exploration \
  --capital 1000000
```

### 검증 모드 (엄격한 조건)
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --mode validation \
  --disable-ppo \
  --capital 1000000
```

## 🔧 고급 설정

### 다중 타임프레임 비활성화
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --disable-multiframe
```

### 특정 지표만 사용
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --indicators "sma_20,rsi,volatility"
```

### 캐시 크기 조정
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --cache-size 2000
```

## 📈 워크포워드 분석

### 기본 워크포워드
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --walk-forward \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### 커스텀 워크포워드
```bash
python run_backtest.py \
  --price-data data/price_data.csv \
  --walk-forward \
  --wf-windows 15 \
  --wf-train-ratio 0.7 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

## 🔄 다중 백테스트

### 병렬 실행 (기본)
```bash
python run_backtest.py \
  --config config/backtest_configs.json \
  --multiple
```

### 순차 실행
```bash
python run_backtest.py \
  --config config/backtest_configs.json \
  --multiple \
  --no-parallel
```

## 📁 파일 구조

```
AuroraQ/
├── run_backtest.py          # 메인 실행기
├── config/
│   └── backtest_configs.json # 설정 파일 예시
├── data/
│   ├── price_data.csv       # 가격 데이터
│   └── sentiment_data.csv   # 감정 데이터
└── reports/
    └── backtest/            # 결과 저장 디렉토리
```

## 📊 결과 확인

백테스트 완료 후 `reports/backtest/` 디렉토리에서 결과 확인:
- `{name}_{timestamp}_result.json`: 상세 결과
- 콘솔 출력: 실시간 진행 상황 및 요약

## ⚡ 빠른 테스트 명령어

### 5분 테스트
```bash
python run_backtest.py \
  --price-data data/test/simple_price.csv \
  --window-size 20 \
  --indicators "sma_20,rsi"
```

### 탐색 모드 테스트
```bash
python run_backtest.py \
  --price-data data/test/simple_price.csv \
  --exploration \
  --window-size 10
```

### 워크포워드 테스트
```bash
python run_backtest.py \
  --price-data data/test/simple_price.csv \
  --walk-forward \
  --wf-windows 5
```

## 🛠 파라미터 설명

- `--price-data`: 가격 데이터 파일 (필수)
- `--sentiment-data`: 감정 데이터 파일 (선택)
- `--capital`: 초기 자본 (기본: 1,000,000)
- `--mode`: 백테스트 모드 (normal/exploration/validation/walk_forward)
- `--window-size`: 데이터 윈도우 크기 (기본: 100)
- `--indicators`: 사용할 지표 목록 (콤마 구분)
- `--exploration`: 탐색 모드 활성화
- `--disable-multiframe`: 다중 타임프레임 비활성화
- `--disable-ppo`: PPO 비활성화
- `--walk-forward`: 워크포워드 분석 실행
- `--multiple`: 다중 백테스트 실행

이제 실제 데이터로 백테스트를 진행할 수 있습니다!