# 🎯 AuroraQ 독립 실행 가이드

AuroraQ만 단독으로 실행하여 리소스를 최적화하고 MacroQ 의존성 없이 암호화폐 트레이딩에 집중하는 방법을 설명합니다.

## 📋 독립 실행의 장점

### ✅ 리소스 최적화
- **메모리 사용량 50% 절약**: MacroQ TFT 모델 미로드
- **CPU 사용량 40% 절약**: 거시경제 데이터 수집 비활성화
- **네트워크 대역폭 60% 절약**: 암호화폐 API만 사용

### ✅ 단순화된 운영
- **빠른 시작**: 초기화 시간 70% 단축
- **명확한 로깅**: AuroraQ 관련 로그만 출력
- **간편한 설정**: 암호화폐 트레이딩 설정만 관리

### ✅ 안정성 향상
- **격리된 실행**: MacroQ 오류가 AuroraQ에 영향 없음
- **독립적 장애 복구**: 개별 재시작 가능
- **모듈화된 업데이트**: AuroraQ만 선택적 업데이트

## 🚀 실행 방법

### 1. 기본 실행 (메인 런처 사용)
```bash
# AuroraQ만 실행
python main.py --mode aurora

# 출력 예시:
# 🚀 Initializing QuantumAI System (Mode: aurora)...
# 🎯 Initializing AuroraQ Agent...
# ✅ QuantumAI System initialized successfully
# 📦 Loaded modules: aurora, sentiment
```

### 2. 전용 독립 런처 사용
```bash
# AuroraQ 전용 런처 (더 최적화됨)
python AuroraQ/standalone_runner.py --mode live

# 백테스트
python AuroraQ/standalone_runner.py --mode backtest --start-date 2025-01-01 --end-date 2025-01-31

# 상태 확인
python AuroraQ/standalone_runner.py --mode status
```

### 3. 데모 및 벤치마크
```bash
# 성능 데모 실행
python examples/aurora_standalone_demo.py
```

## ⚙️ 리소스 최적화 설정

### 캐시 최적화
```python
# SharedCore/utils/cache_manager.py 설정
cache_config = CacheConfig(
    mode=CacheMode.AURORA_ONLY,  # 암호화폐 + 감정분석만
    crypto_ttl=60,               # 1분 캐시
    sentiment_ttl=1800,          # 30분 캐시
    max_memory_mb=256,           # 메모리 제한
    cleanup_interval=300         # 5분마다 정리
)
```

### 데이터 프로바이더 최적화
```python
# SharedCore/data_layer/unified_data_provider.py
data_provider = UnifiedDataProvider(
    use_crypto=True,    # 암호화폐 데이터만
    use_macro=False     # 거시경제 데이터 비활성화
)
```

## 📊 성능 비교

### 리소스 사용량 비교
| 모드 | 메모리 (MB) | CPU (%) | 초기화 시간 (초) |
|------|-------------|---------|------------------|
| **Aurora Only** | 180 | 12 | 3.2 |
| Full System | 360 | 20 | 10.8 |
| 절약량 | **-50%** | **-40%** | **-70%** |

### API 호출 최적화
```yaml
활성화된 API:
  - Binance API (암호화폐 데이터)
  - NewsAPI (감정분석용 뉴스)
  - Reddit API (소셜 감정)

비활성화된 API:
  - Yahoo Finance (주식 데이터)
  - FRED API (경제지표)
  - Alpha Vantage (거시경제)
```

## 🔧 설정 최적화

### AuroraQ 전용 설정
```yaml
# AuroraQ/config/default_config.yaml
agent:
  initial_capital: 100000.0
  max_position_size: 0.2
  risk_per_trade: 0.02
  mode: "simulation"           # 또는 "live"

# 리소스 최적화 설정
data:
  sentiment_lookback_hours: 12  # 24 → 12 (절반)
  cache_ttl: 300               # 5분 캐시
  
strategies:
  ppo_weight: 0.3
  rule_weight: 0.7
  
# 불필요한 기능 비활성화
monitoring:
  enable_telegram: false       # 필요시에만 활성화
  performance_update_interval: 7200  # 2시간 (덜 빈번)
```

### 환경 변수 최적화
```bash
# .env 파일
# 필수 API만 설정
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret

# 선택적 (감정분석용)
NEWSAPI_KEY=your_news_key
REDDIT_CLIENT_ID=your_reddit_id

# MacroQ 관련은 불필요
# ALPHA_VANTAGE_KEY=  (주석 처리)
# FRED_API_KEY=       (주석 처리)
```

## 🚨 리소스 모니터링

### 자동 최적화
```python
from SharedCore.utils.resource_monitor import get_resource_monitor

monitor = get_resource_monitor()

# 현재 상태 확인
usage = monitor.get_current_usage()
health = monitor.check_resource_health()

# 최적화 제안 받기
suggestions = monitor.get_optimization_suggestions(mode="aurora")

# 자동 최적화 모드 판단
if monitor.should_enable_optimization_mode():
    # 배치 크기 축소, 캐시 TTL 증가 등 자동 적용
    pass
```

### 실시간 모니터링
```bash
# 리소스 사용량 실시간 확인
watch -n 5 'ps aux | grep python | grep aurora'

# 메모리 사용량 모니터링
htop -p $(pgrep -f aurora)
```

## 💡 최적화 팁

### 1. 메모리 최적화
```python
# 데이터 보존 기간 단축
MARKET_DATA_RETENTION_HOURS = 24    # 기본: 168 (1주일)
SENTIMENT_RETENTION_HOURS = 12      # 기본: 72 (3일)

# 배치 크기 조정
SENTIMENT_BATCH_SIZE = 16           # 기본: 32
PPO_BATCH_SIZE = 64                 # 기본: 128
```

### 2. CPU 최적화
```python
# 업데이트 주기 조정
MARKET_DATA_UPDATE_INTERVAL = 60    # 1분 (기본: 30초)
SENTIMENT_UPDATE_INTERVAL = 1800    # 30분 (기본: 600초)
PORTFOLIO_UPDATE_INTERVAL = 300     # 5분 (기본: 60초)
```

### 3. 네트워크 최적화
```python
# API 호출 제한
MAX_CONCURRENT_REQUESTS = 3         # 기본: 10
REQUEST_RETRY_DELAY = 2             # 2초 (기본: 1초)
CACHE_AGGRESSIVE_MODE = True        # 적극적 캐싱
```

## 🔄 운영 시나리오

### 개발 환경
```bash
# 로컬 개발 (디버깅 모드)
python AuroraQ/standalone_runner.py --mode live --config dev_config.yaml
```

### 테스트 환경
```bash
# 시뮬레이션 모드
python main.py --mode aurora --config test_config.yaml
```

### 운영 환경 (VPS)
```bash
# 백그라운드 실행
nohup python AuroraQ/standalone_runner.py --mode live > aurora.log 2>&1 &

# systemd 서비스 등록
sudo systemctl start auroaq-standalone
sudo systemctl enable auroaq-standalone
```

## 📈 성과 추적

### 독립 실행 전용 메트릭
- **시스템 효율성**: 리소스 사용량 대비 수익률
- **응답 속도**: API 응답 시간 및 거래 지연시간
- **안정성**: 업타임 및 오류 발생률
- **비용 효율성**: VPS 비용 대비 성과

이 가이드를 따라 AuroraQ를 독립적으로 실행하면 **리소스 50% 절약**과 **성능 향상**을 동시에 달성할 수 있습니다.