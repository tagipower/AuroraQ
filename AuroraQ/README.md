# AuroraQ - 고급 암호화폐 자동거래 시스템

## 📊 시스템 개요

AuroraQ는 AI/ML 기반의 암호화폐 자동거래 시스템으로, PPO(Proximal Policy Optimization) 강화학습과 룰 기반 전략을 결합하여 바이낸스에서 자동거래를 수행합니다.

## 🏗️ 프로젝트 구조

```
AuroraQ/
├── 📁 AuroraQ/                     # 메인 프로젝트 디렉토리
│   ├── 📁 config/                  # 시스템 설정
│   │   ├── env_loader.py           # 환경변수 로더
│   │   ├── env_validator.py        # 환경변수 검증
│   │   ├── onnx_settings.py       # ONNX 모델 설정
│   │   └── metrics_integration.yaml
│   │
│   ├── 📁 core/                    # 핵심 시스템 모듈
│   │   ├── 📁 error_recovery/      # 오류 복구 시스템
│   │   ├── 📁 event_management/    # 이벤트 TTL 관리
│   │   ├── 📁 logging_management/  # 로그 관리 시스템
│   │   ├── 📁 model_management/    # AI 모델 관리
│   │   ├── 📁 performance/         # 성능 최적화
│   │   ├── 📁 resource_management/ # 리소스 관리
│   │   └── 📁 strategy_protection/ # 전략 보호 시스템
│   │
│   ├── 📁 dashboard/               # 실시간 대시보드
│   │   ├── aurora_dashboard_final.py # 고급 대시보드 (모드 전환 포함)
│   │   └── start_dashboard.bat
│   │
│   ├── 📁 deployment/              # 배포 시스템
│   │   ├── api_system.py           # API 시스템
│   │   ├── vps_standalone_runner.py # 독립 실행기
│   │   ├── docker-compose-production.yml
│   │   └── deploy_vps.sh
│   │
│   ├── 📁 infrastructure/          # 인프라 관리
│   │   ├── 📁 monitoring/          # 모니터링 시스템
│   │   ├── 📁 logs/               # 로그 저장소
│   │   └── 📁 vps_logging/        # VPS 로깅 시스템
│   │
│   ├── 📁 scripts/                 # 운영 스크립트
│   │   ├── deploy_trading_system.sh
│   │   ├── monitor_trading_system.sh
│   │   └── restart_trading_system.sh
│   │
│   ├── 📁 sentiment/               # 감정 분석 서비스
│   │   ├── 📁 api/                # API 라우터
│   │   ├── 📁 collectors/         # 뉴스/데이터 수집
│   │   ├── 📁 processors/         # 감정 분석 처리
│   │   └── 📁 schedulers/         # 배치 스케줄러
│   │
│   ├── 📁 trade/                   # 거래 시스템
│   │   └── 📁 trading/            # 거래 로직
│   │       ├── ppo_agent.py       # PPO 에이전트
│   │       ├── ppo_strategy.py    # PPO 전략
│   │       ├── rule_strategies.py # 룰 기반 전략
│   │       ├── vps_market_data.py # 시장 데이터
│   │       ├── vps_order_manager.py # 주문 관리
│   │       └── vps_realtime_system.py # 실시간 시스템
│   │
│   └── 📁 utils/                   # 유틸리티
│       ├── debug_system.py         # 디버그 시스템
│       ├── security_system.py      # 보안 시스템
│       └── validate_deployment.py  # 배포 검증
│
├── 📁 sentiment-service/           # 독립 감정분석 서비스
├── 📁 docs/                       # 문서
│
├── improved_binance_verification.py # 바이낸스 통합 검증
├── setup_binance_testnet.py       # 테스트넷 설정 도우미
└── README.md                      # 이 파일
```

## 🚀 핵심 기능

### 1. 🤖 AI 기반 거래 전략
- **PPO 강화학습**: 시장 데이터 학습을 통한 자동 전략 최적화
- **룰 기반 전략**: 5개의 독립적인 룰 기반 거래 전략
- **통합 신호 시스템**: 다중 전략 신호 융합

### 2. 📈 실시간 모니터링
- **고급 대시보드**: Rich TUI 기반 실시간 대시보드
- **모드 전환**: Paper/Live/Backtest 모드 실시간 전환
- **성과 추적**: 실시간 PnL, 승률, Sharpe ratio 모니터링

### 3. 🔒 위험 관리
- **포지션 관리**: 자동 포지션 크기 조절
- **손절매/익절**: 동적 위험 관리
- **전략 보호**: 비정상 신호 감지 및 차단

### 4. 📊 감정 분석
- **뉴스 분석**: FinBERT 기반 뉴스 감정 분석
- **거시지표**: 경제 지표 통합 분석
- **이벤트 관리**: TTL 기반 이벤트 생명주기 관리

## ⚙️ 설치 및 설정

### 1. 환경 요구사항
```bash
Python 3.9+
Node.js (대시보드용)
Docker (선택사항)
```

### 2. 의존성 설치
```bash
pip install -r AuroraQ/deployment/requirements.txt
```

### 3. API 키 설정
```bash
# 테스트넷 설정 (권장)
python setup_binance_testnet.py

# 또는 수동으로 .env 파일 생성
echo "TRADING_MODE=paper" > .env
echo "BINANCE_TESTNET_API_KEY=your_testnet_key" >> .env
echo "BINANCE_TESTNET_API_SECRET=your_testnet_secret" >> .env
```

### 4. 시스템 검증
```bash
python improved_binance_verification.py
```

## 🎮 사용법

### 1. 대시보드 실행
```bash
cd AuroraQ/dashboard
python aurora_dashboard_final.py
```

**대시보드 단축키:**
- `M`: 모드 선택 메뉴
- `P`: Paper Trading 모드
- `L`: Live Trading 모드 (API 설정 필요)
- `1-8`: 메뉴 탐색
- `Q`: 종료

### 2. 독립 실행
```bash
cd AuroraQ/deployment
python vps_standalone_runner.py
```

### 3. Docker 배포
```bash
docker-compose -f AuroraQ/deployment/docker-compose-production.yml up -d
```

## 📊 거래 모드

### 1. Paper Trading (시뮬레이션)
- 실제 자금 없이 안전한 테스트
- 테스트넷 API 사용 (권장)
- 전략 검증 및 학습

### 2. Live Trading (실거래)
- 실제 자금으로 거래 실행
- 메인넷 API 필수
- 철저한 테스트 후 사용 권장

### 3. Backtest Mode
- 과거 데이터로 전략 백테스팅
- API 연결 불필요
- 전략 성과 분석

### 4. Dry Run Mode
- 거래 신호만 확인
- 실제 주문 없음
- 전략 모니터링용

## 🔧 주요 설정

### 환경변수 (.env)
```env
# 거래 모드
TRADING_MODE=paper

# 바이낸스 API (테스트넷)
BINANCE_TESTNET_API_KEY=your_testnet_key
BINANCE_TESTNET_API_SECRET=your_testnet_secret
BINANCE_TESTNET=true

# 바이낸스 API (실거래)
BINANCE_API_KEY=your_mainnet_key
BINANCE_API_SECRET=your_mainnet_secret

# 거래 설정
EXCHANGE=binance
SYMBOL=BTCUSDT
MAX_POSITION_SIZE=0.05
STOP_LOSS_PERCENTAGE=0.02
TAKE_PROFIT_PERCENTAGE=0.04

# VPS 최적화
VPS_MEMORY_LIMIT=3G
VPS_CPU_LIMIT=2
LOG_LEVEL=INFO
```

## 📈 성능 최적화

### 1. 메모리 관리
- 동적 배치 관리
- 자동 가비지 컬렉션
- TTL 기반 이벤트 정리

### 2. 성능 모니터링
- Prometheus 메트릭
- 실시간 성능 추적
- 자동 알림 시스템

### 3. 오류 복구
- API 연결 자동 복구
- 전략 이상 감지
- 자동 재시작 메커니즘

## 🛡️ 보안 기능

### 1. API 보안
- IP 제한 권장
- 최소 권한 원칙
- 키 로테이션 지원

### 2. 전략 보호
- 비정상 신호 차단
- 위험 한도 제한
- 자동 안전 정지

### 3. 데이터 보안
- 로그 민감정보 마스킹
- 암호화된 설정 저장
- 감사 추적 완전성

## 📊 모니터링 및 로깅

### 1. 실시간 대시보드
- 시스템 상태 모니터링
- 거래 성과 추적
- 리소스 사용량 확인

### 2. 로그 관리
- 통합 로그 시스템
- 자동 아카이브
- 로그 보존 정책

### 3. 알림 시스템
- 이상 상황 알림
- 성과 리포트
- 시스템 상태 알림

## 🔍 문제 해결

### 1. API 연결 문제
```bash
# 연결 상태 확인
python improved_binance_verification.py

# 테스트넷 설정
python setup_binance_testnet.py
```

### 2. 메모리 부족
```bash
# 메모리 사용량 확인
python AuroraQ/utils/system_validator.py

# 설정 조정: VPS_MEMORY_LIMIT=2G
```

### 3. 성능 저하
```bash
# 성능 분석
python AuroraQ/core/performance/performance_optimizer.py
```

## 📚 문서

- [대시보드 가이드](AuroraQ/DASHBOARD_GUIDE.md)
- [SSH 접속 가이드](AuroraQ/SSH_DASHBOARD_GUIDE.md)
- [모바일 접속 가이드](AuroraQ/MOBILE_ACCESS_GUIDE.md)
- [폴더 구조](AuroraQ/FOLDER_STRUCTURE_FINAL.md)
- [감정분석 서비스](AuroraQ/sentiment/SENTIMENT_SERVICE_ANALYSIS.md)

## 🤝 기여 및 지원

이 프로젝트는 개인 거래 시스템으로 개발되었습니다. 사용 시 다음 사항을 주의하세요:

- **투자 위험**: 암호화폐 거래는 높은 위험을 수반합니다
- **테스트 우선**: 실거래 전 충분한 테스트를 권장합니다
- **자기 책임**: 모든 거래 결과에 대한 책임은 사용자에게 있습니다

## 📄 라이선스

이 프로젝트는 개인 사용을 위한 것입니다. 상업적 사용 시 별도 협의가 필요합니다.

---

**⚠️ 주의사항**: 이 시스템은 교육 및 연구 목적으로 개발되었습니다. 실제 거래 시에는 충분한 테스트와 검증 후 사용하시기 바랍니다.