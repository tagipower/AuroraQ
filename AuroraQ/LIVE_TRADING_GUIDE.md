# 🔥 AuroraQ 실전 거래 연결 가이드

AuroraQ를 실제 Binance API 및 Feedly 뉴스와 연결하여 실전 데이터로 운영하는 완전한 가이드입니다.

## 🎯 연결된 실전 시스템

### ✅ 구현된 실전 연결
- **Binance API**: 실시간 암호화폐 시장 데이터 및 거래 실행
- **Feedly API**: 실시간 암호화폐 뉴스 수집 및 감정분석
- **통합 데이터 레이어**: 실전 데이터와 센티멘트 통합 제공
- **리소스 최적화**: 캐싱 및 API 제한 관리

### 🔄 데이터 플로우
```
Binance API → 실시간 OHLCV 데이터 → UnifiedDataProvider
     ↓                                        ↓
Feedly API → 뉴스 수집 → 감정분석 → SentimentAggregator
     ↓                                        ↓
     → AuroraQ Agent → PPO + Rules → 거래 결정
```

## 🔑 필수 API 키 설정

### 1. Binance API 설정 (필수)

#### 테스트넷 계정 생성 (권장)
```bash
# 1. https://testnet.binance.vision/ 접속
# 2. GitHub/Google 계정으로 로그인
# 3. API Key 생성
# 4. .env 파일에 추가
```

#### .env 파일 설정
```bash
# Binance API (테스트넷)
BINANCE_API_KEY=your_actual_testnet_api_key
BINANCE_API_SECRET=your_actual_testnet_secret
BINANCE_TESTNET=true

# 실제 거래시 (주의!)
# BINANCE_API_KEY=your_real_api_key
# BINANCE_API_SECRET=your_real_secret  
# BINANCE_TESTNET=false
```

### 2. Feedly API 설정 (선택사항)

#### 무료 토큰 발급
```bash
# 1. https://developer.feedly.com/ 방문
# 2. 무료 계정 생성
# 3. Developer Console에서 토큰 발급
# 4. .env 파일에 추가
```

```bash
# Feedly API (뉴스 감정분석)
FEEDLY_ACCESS_TOKEN=your_feedly_access_token

# 토큰 없이도 작동하지만 제한됨
# FEEDLY_ACCESS_TOKEN=
```

### 3. 기타 API (선택사항)
```bash
# NewsAPI (추가 뉴스 소스)
NEWSAPI_KEY=your_newsapi_key

# Telegram 알림
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🚀 실전 연결 테스트

### 1. 환경 설정 확인
```bash
# 의존성 설치
pip install python-binance aiohttp python-dotenv

# 환경 변수 로드 확인
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('BINANCE_API_KEY')[:10] + '...' if os.getenv('BINANCE_API_KEY') else 'Not set')"
```

### 2. 연결 테스트 실행
```bash
# 실전 API 연결 테스트
python tests/test_real_connections.py

# 예상 출력:
# 🔗 Testing Binance API Connection...
#   ✅ Binance connection successful (testnet)
#      Account: SPOT
#      BTC/USDT Price: $43,250.00
#
# 📰 Testing Feedly API Connection...
#   ✅ Feedly connection successful  
#      Articles collected: 15
#      Overall sentiment: 0.62
```

### 3. AuroraQ 실전 모드 실행
```bash
# 실전 데이터로 AuroraQ 실행
python main.py --mode aurora

# 또는 독립 실행
python AuroraQ/standalone_runner.py --mode live
```

## 📊 실전 데이터 검증

### Binance 실시간 데이터 확인
```python
# 실시간 가격 확인
from SharedCore.data_layer.unified_data_provider import UnifiedDataProvider

provider = UnifiedDataProvider(use_crypto=True, use_macro=False)
await provider.connect()

# BTC/USDT 1시간 데이터
data = await provider.get_market_data("crypto", "BTC/USDT", "1h")
print(f"Latest BTC price: ${data['close'].iloc[-1]:,.2f}")
print(f"24h volume: {data['volume'].sum():,.0f}")
```

### Feedly 뉴스 감정분석 확인
```python
# 실시간 센티멘트 확인
from SharedCore.sentiment_engine.sentiment_aggregator import SentimentAggregator

aggregator = SentimentAggregator()
sentiment = await aggregator.get_real_time_sentiment("BTC")

print(f"BTC Sentiment: {sentiment['sentiment_score']:.2f}")
print(f"Confidence: {sentiment['confidence']:.2f}")
print(f"Articles: {sentiment['article_count']}")
```

## ⚙️ 실전 모드 설정

### AuroraQ 실전 거래 설정
```yaml
# AuroraQ/config/default_config.yaml
agent:
  initial_capital: 1000.0      # 실제 투자 금액
  max_position_size: 0.1       # 10% (보수적)
  risk_per_trade: 0.01         # 1% (보수적)  
  mode: "live"                 # 실제 거래 모드

# 테스트넷에서 충분히 검증 후 사용!
```

### 리스크 관리 설정
```yaml
# 일일 손실 한도
daily_loss_limit: 0.02         # 2%

# 최대 포지션 수
max_concurrent_positions: 3

# 강제 손절 설정
stop_loss_percent: 0.05        # 5%
take_profit_percent: 0.10      # 10%
```

## 🔍 실시간 모니터링

### 1. 거래 로그 모니터링
```bash
# 실시간 로그 확인
tail -f logs/aurora_q.log

# 거래 관련 로그만 필터링
grep "TRADE\|ORDER\|POSITION" logs/aurora_q.log
```

### 2. 성과 대시보드
```python
# 실시간 성과 확인
from AuroraQ.standalone_runner import AuroraQStandalone

runner = AuroraQStandalone()
await runner.initialize()

status = await runner.get_status()
print(f"Portfolio Value: ${status['portfolio']['total_value']:,.2f}")
print(f"Today's P&L: {status['performance']['daily_return']:.2%}")
print(f"Total Return: {status['performance']['total_return']:.2%}")
```

### 3. Telegram 알림 설정
```bash
# .env 파일에 봇 설정
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=123456789

# 알림 활성화
monitoring:
  enable_telegram: true
  alert_on_trades: true
  alert_on_errors: true
```

## 🚨 안전 수칙 및 주의사항

### ⚠️ 필수 안전 수칙
1. **테스트넷 우선**: 실제 돈 투자 전에 테스트넷에서 충분히 검증
2. **소액 시작**: 처음에는 최소 금액으로 시작
3. **손실 한도**: 감당할 수 있는 금액만 투자
4. **지속적 모니터링**: 봇 운영 중에는 정기적으로 상태 확인
5. **백업 계획**: 시스템 장애시 수동 대응 방안 준비

### 🔒 보안 권장사항
```bash
# API 키 권한 최소화
- Spot Trading: 활성화
- Futures: 비활성화 (필요시에만)
- Withdraw: 비활성화 (절대 활성화 금지)

# IP 제한 설정
- VPS IP만 허용
- 개발 PC IP 추가 (필요시)
```

### 💰 리스크 관리
```python
# 자동 안전장치 설정
safety_limits = {
    'max_daily_loss': 0.02,        # 일일 2% 손실시 정지
    'max_drawdown': 0.15,          # 15% 낙폭시 정지  
    'max_consecutive_losses': 5,   # 연속 5회 손실시 정지
    'min_account_balance': 100.0   # 최소 잔고 유지
}
```

## 📈 성과 추적 및 최적화

### 주요 지표 모니터링
- **샤프 비율**: 위험 대비 수익률
- **승률**: 수익 거래 비율  
- **평균 수익/손실**: 거래당 평균 손익
- **최대 낙폭**: 최고점 대비 최대 하락
- **센티멘트 정확도**: 뉴스 감정과 가격 상관관계

### 백테스트 검증
```bash
# 실전 전 백테스트 필수
python main.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31

# 다양한 시장 상황 테스트
python main.py --mode backtest --start-date 2022-05-01 --end-date 2022-07-31  # 하락장
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-04-30  # 상승장
```

## 🎯 실전 운영 체크리스트

### 실행 전 체크리스트
- [ ] Binance 테스트넷 API 키 설정
- [ ] Feedly API 토큰 설정 (선택)
- [ ] 연결 테스트 성공 확인
- [ ] 백테스트 성과 검증
- [ ] 리스크 한도 설정
- [ ] 모니터링 시스템 준비
- [ ] 수동 개입 계획 수립

### 일일 운영 체크리스트  
- [ ] 시스템 상태 확인
- [ ] 포트폴리오 현황 점검
- [ ] 뉴스/이벤트 확인
- [ ] 성과 지표 분석
- [ ] 로그 에러 체크

### 주간 운영 체크리스트
- [ ] 전략 성과 분석
- [ ] 파라미터 최적화 검토
- [ ] 시장 변화 대응
- [ ] 리스크 한도 재검토

## 🆘 문제 해결

### 일반적인 문제
1. **API 연결 실패**: 키 유효성, 네트워크, 권한 확인
2. **감정분석 오류**: Feedly 토큰, 네트워크, 뉴스 소스 확인  
3. **거래 실행 실패**: 잔고, 시장 상황, API 제한 확인
4. **성과 부진**: 시장 상황, 전략 파라미터, 데이터 품질 점검

### 긴급 대응
```python
# 모든 포지션 강제 청산 (긴급시만)
from SharedCore.data_layer.market_data.binance_collector import create_binance_collector

collector = create_binance_collector(api_key, api_secret, testnet=True)
await collector.connect()

# 미체결 주문 모두 취소
open_orders = await collector.get_open_orders()
for order in open_orders:
    await collector.cancel_order(order['symbol'], order['orderId'])
```

이 가이드를 따라하면 AuroraQ를 **안전하고 효과적으로** 실전 환경에서 운영할 수 있습니다. 항상 **테스트넷부터 시작**하고, **소액으로 검증** 후에 본격적인 투자를 진행하세요!