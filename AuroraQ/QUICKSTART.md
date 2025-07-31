# 🚀 QuantumAI - 빠른 시작 가이드

## 📋 준비사항

### 시스템 요구사항
- **Python 3.8+**
- **RAM 4GB+** (권장 8GB)
- **디스크 10GB+**
- **Redis** (선택사항, 캐싱용)

### API 키 준비
- **Binance API** (암호화폐 데이터)
- **GitHub Token** (MCP 연동, 선택사항)

## ⚡ 1분 설치

### 1. 프로젝트 클론
```bash
git clone https://github.com/yourusername/QuantumAI.git
cd QuantumAI
```

### 2. 가상환경 생성
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. 의존성 설치
```bash
# 기본 패키지
pip install -r requirements/base.txt

# MacroQ 사용시 추가
pip install -r requirements/macro.txt
```

### 4. 환경 설정
```bash
# .env 파일 생성 (이미 존재)
# 필요한 값들을 입력하세요

# .env 파일 내용:
GITHUB_TOKEN=your_github_token_here
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
```

## 🎯 실행 방법

### AuroraQ만 실행 (단기 트레이딩)
```bash
python main.py --mode aurora
```

### MacroQ만 실행 (중장기 포트폴리오)
```bash
python main.py --mode macro
```

### 둘 다 실행
```bash
python main.py --mode both
```

### 백테스트 실행
```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
```

## 📊 실행 결과 예시

### AuroraQ 실행시
```
🚀 Initializing QuantumAI System...
✅ QuantumAI System initialized successfully
🎯 Starting AuroraQ Agent...
INFO - AuroraQ Agent initialized with config: AuroraQConfig(...)
INFO - Starting AuroraQ trading...
INFO - Market data collected: BTC/USDT
INFO - Sentiment score: 0.65 (Positive)
INFO - SIMULATION: {'action': 'buy', 'symbol': 'BTC/USDT', 'size': 0.02}
```

### MacroQ 실행시
```
📊 Starting MacroQ System...
INFO - Loading TFT model...
INFO - Portfolio optimizer ready
INFO - MacroQ TFT model ready for predictions
INFO - MacroQ: Portfolio optimization cycle...
```

## 🔧 설정 커스터마이징

### AuroraQ 설정 수정
```python
# AuroraQ/config/default_config.yaml
agent:
  initial_capital: 50000.0    # 초기 자본 (기본: 100k)
  risk_per_trade: 0.01        # 거래당 리스크 (기본: 2%)
  mode: "live"                # 실제 거래 (기본: simulation)

strategies:
  ppo_weight: 0.5             # PPO 가중치 (기본: 0.3)
  rule_weight: 0.5            # Rule 가중치 (기본: 0.7)
```

### MacroQ 자산 목록 수정
```yaml
# MacroQ/config/assets.yaml
assets:
  - symbol: "SPY"             # S&P 500 ETF
    type: "etf"
    weight: 0.3
  - symbol: "QQQ"             # NASDAQ ETF  
    type: "etf"
    weight: 0.2
  - symbol: "BTC"             # Bitcoin
    type: "crypto"
    weight: 0.1
```

## 🐛 트러블슈팅

### 일반적인 문제

#### 1. Redis 연결 오류
```bash
# Redis가 없어도 동작하지만, 성능 향상을 위해 설치 권장
sudo apt-get install redis-server  # Ubuntu
brew install redis                  # macOS

# Redis 없이 실행하려면 메모리 캐시 사용
```

#### 2. 메모리 부족
```bash
# 배치 크기 축소
export BATCH_SIZE=16  # 기본: 32

# 또는 스왑 메모리 추가
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile  
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. GPU 오류 (MacroQ)
```python
# CPU만 사용하도록 강제
import torch
torch.cuda.is_available = lambda: False
```

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/aurora_q.log

# 오류 로그만 확인
grep ERROR logs/aurora_q.log
```

### 성능 모니터링
```bash
# 시스템 리소스 사용량
htop

# 프로세스별 메모리 사용량
ps aux | grep python
```

## 📈 다음 단계

### 1. 백테스트 분석
```bash
# 다양한 기간으로 백테스트
python main.py --mode backtest --start-date 2022-01-01 --end-date 2023-12-31
python main.py --mode backtest --start-date 2023-06-01 --end-date 2024-06-01
```

### 2. 실제 거래 전환
```yaml
# AuroraQ/config/default_config.yaml
agent:
  mode: "live"  # simulation → live

binance:
  testnet: false  # 실제 거래
```

### 3. 모니터링 설정
```yaml
# 텔레그램 알림 활성화
monitoring:
  enable_telegram: true
  telegram_bot_token: "your_telegram_bot_token"
  telegram_chat_id: "YOUR_CHAT_ID"
```

### 4. 자동 시작 설정 (Linux)
```bash
# 시스템 서비스로 등록
sudo cp deployment/systemd/quantumai.service /etc/systemd/system/
sudo systemctl enable quantumai
sudo systemctl start quantumai
```

## 🆘 도움이 필요하다면

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **Documentation**: `ARCHITECTURE_V2.md` 참고
- **설정 가이드**: `configs/` 폴더의 예제 파일들

---

**⚠️ 중요**: 실제 거래 전에 반드시 시뮬레이션 모드에서 충분히 테스트하세요!