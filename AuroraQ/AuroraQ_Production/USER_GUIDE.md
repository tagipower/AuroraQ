# AuroraQ Production 사용자 가이드

## 🎯 빠른 시작

### 1분 만에 시작하기

```bash
# 1. 데모 모드로 시스템 체험 (2분간)
python main.py --mode demo --duration 2

# 2. 테스트 모드로 안전하게 테스트 (5분간)
python main.py --mode test --duration 5

# 3. 실거래 모드 (주의: 실제 자금 필요)
python main.py --mode live
```

## 📖 상세 사용법

### 기본 명령어

#### 데모 모드
```bash
# 기본 데모 (2분)
python main.py --mode demo

# 시간 지정 데모
python main.py --mode demo --duration 5

# 센티멘트 분석 포함 데모
python main.py --mode demo --sentiment

# 설정 파일 지정
python main.py --mode demo --config my_config.yaml
```

#### 테스트 모드
```bash
# 기본 테스트 (5분)
python main.py --mode test

# 장시간 테스트 (30분)
python main.py --mode test --duration 30

# 디버그 로그 포함 테스트
python main.py --mode test --log-level DEBUG
```

#### 실거래 모드
```bash
# 실거래 시작 (무제한)
python main.py --mode live

# 시간 제한 실거래 (60분)
python main.py --mode live --duration 60

# 고급 설정으로 실거래
python main.py --mode live --config production_config.yaml --sentiment
```

### 설정 옵션

#### 명령행 옵션
| 옵션 | 설명 | 기본값 | 예시 |
|------|------|--------|------|
| `--mode` | 실행 모드 | live | demo, test, live |
| `--config` | 설정 파일 | config.yaml | my_config.yaml |
| `--duration` | 실행 시간(분) | 0(무제한) | 30 |
| `--sentiment` | 센티멘트 분석 | False | --sentiment |
| `--log-level` | 로그 레벨 | INFO | DEBUG, WARNING |

#### 설정 파일 (config.yaml)

**거래 설정**
```yaml
trading:
  max_position_size: 0.1          # 최대 포지션 크기 (10%)
  emergency_stop_loss: 0.05       # 긴급 손절선 (5%)
  max_daily_trades: 10            # 일일 최대 거래 횟수
  min_data_points: 50             # 최소 데이터 포인트
```

**전략 설정**
```yaml
strategy:
  rule_strategies:                # 사용할 Rule 전략
    - "RuleStrategyA"             # 트렌드 추종 전략
    - "RuleStrategyB"             # 평균 회귀 전략
    - "RuleStrategyC"             # 볼린저 밴드 전략
  enable_ppo: true                # PPO 강화학습 활성화
  hybrid_mode: "ensemble"         # ensemble/consensus/competition
  execution_strategy: "market"    # market/limit/smart
  ppo_weight: 0.3                # PPO 가중치 (30%)
```

**리스크 관리**
```yaml
risk:
  max_drawdown: 0.15              # 최대 낙폭 허용 (15%)
  max_portfolio_risk: 0.02        # 포트폴리오 리스크 (2%)
  risk_tolerance: "moderate"      # conservative/moderate/aggressive
```

## 🔧 기능별 사용법

### 1. 하이브리드 전략 모드

#### Ensemble 모드 (기본)
- **특징**: 모든 전략의 가중평균
- **장점**: 안정적, 분산 효과
- **설정**: `hybrid_mode: "ensemble"`

#### Consensus 모드  
- **특징**: 2/3 이상 전략이 동의할 때만 실행
- **장점**: 높은 정확도
- **설정**: `hybrid_mode: "consensus"`

#### Competition 모드
- **특징**: 가장 높은 신뢰도 전략만 사용
- **장점**: 최고 성과 전략 활용
- **설정**: `hybrid_mode: "competition"`

### 2. 센티멘트 분석

#### 활성화 방법
```bash
# 명령행에서 활성화
python main.py --sentiment

# 또는 설정 파일에서
sentiment:
  enable_sentiment: true
  news_sources:
    - "coindesk"
    - "yahoo_finance"
  sentiment_weight: 0.2
```

#### 지원 뉴스 소스
- **CoinDesk**: 암호화폐 전문 뉴스
- **Yahoo Finance**: 금융 뉴스
- **CoinTelegraph**: 블록체인 뉴스
- **Reuters**: 국제 금융 뉴스

### 3. 알림 시스템

#### 콘솔 알림
```yaml
notifications:
  enable_notifications: true
  channels:
    - "console"                   # 터미널 출력
```

#### 파일 로그
```yaml
notifications:
  channels:
    - "file"                      # logs/notifications.log
```

#### 이메일 알림 (고급)
```yaml
notifications:
  channels:
    - "email"
  email_recipients:
    - "trader@company.com"
```

## 📊 결과 모니터링

### 실시간 상태 확인

시스템 실행 중 1분마다 다음 정보가 출력됩니다:

```
=== 실시간 거래 시스템 상태 (14:30:15) ===
현재 가격: 50,125.34
Position: LONG 0.0250 @ 49,980.12 (PnL: +0.29%)
총 신호: 45, 실행된 거래: 8 (실행률: 17.8%)
일일 거래 수: 3/10
데이터 버퍼: 85/100
```

### 최종 성과 리포트

시스템 종료 시 자동으로 생성되는 리포트:

```
🎯 최종 성과 리포트
========================
📊 신호 생성: 156개
⚡ 실행된 거래: 28개  
📈 신호 실행률: 17.9%
✅ 완료된 거래: 25개
🏆 승률: 68.0%
💰 평균 수익률: +1.24%
📍 현재 포지션: 없음
========================
```

## ⚠️ 리스크 관리

### 기본 안전장치

1. **긴급 손절**: 5% 손실 시 자동 청산
2. **일일 한도**: 하루 최대 10거래 제한
3. **포지션 크기**: 자본의 10% 이하 권장
4. **중복 방지**: 동일 방향 포지션 중복 차단

### 추천 설정

#### 보수적 설정
```yaml
trading:
  max_position_size: 0.05         # 5%
  emergency_stop_loss: 0.03       # 3%
  max_daily_trades: 5
risk:
  risk_tolerance: "conservative"
```

#### 공격적 설정  
```yaml
trading:
  max_position_size: 0.15         # 15%
  emergency_stop_loss: 0.08       # 8%
  max_daily_trades: 20
risk:
  risk_tolerance: "aggressive"
```

## 🔍 문제 해결

### 일반적인 문제

#### 1. 신호가 생성되지 않음
```bash
# 최소 데이터 포인트 줄이기
min_data_points: 20  # 기본값: 50

# 또는 더 오래 대기
python main.py --mode demo --duration 10
```

#### 2. 거래가 실행되지 않음
```bash
# 신뢰도 임계값 낮추기
strategy:
  min_confidence: 0.4  # 기본값: 0.6

# 하이브리드 모드 변경
hybrid_mode: "ensemble"  # consensus 대신
```

#### 3. 메모리 부족
```bash
# 데이터 버퍼 크기 줄이기
trading:
  lookback_periods: 50  # 기본값: 100
```

### 로그 확인

#### 로그 레벨 변경
```bash
# 상세 로그 보기
python main.py --log-level DEBUG

# 오류만 보기  
python main.py --log-level ERROR
```

#### 로그 파일 위치
- **메인 로그**: `logs/auroraQ.log`
- **거래 로그**: `logs/trading.log`
- **알림 로그**: `logs/notifications.log`

## 📈 성과 향상 팁

### 1. 전략 조합 최적화
```bash
# 최적화 실행 후 설정 자동 적용
python optimization/optimal_combination_recommender.py
```

### 2. 센티멘트 활용
```bash
# 뉴스 기반 거래 활성화
python main.py --sentiment --mode live
```

### 3. 백테스트 기반 조정
```bash
# 과거 데이터로 전략 검증
python tests/test_strategies.py
```

## 🚀 고급 사용법

### 사용자 정의 전략 추가
1. `strategies/` 폴더에 새 전략 파일 생성
2. `BaseRuleStrategy` 클래스 상속
3. `score()` 메서드 구현
4. `strategy_registry.py`에 등록

### API 연동
- Binance API
- Upbit API  
- 기타 거래소 API

### 클라우드 배포
- AWS EC2
- Google Cloud
- Docker 컨테이너