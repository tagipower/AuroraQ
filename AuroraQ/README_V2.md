# 🚀 QuantumAI - AI Agent 기반 통합 자산 운용 시스템

> **AuroraQ** (단기 트레이딩) + **MacroQ** (중장기 포트폴리오) = 완전한 AI 자산 운용 솔루션

## 📌 프로젝트 개요

QuantumAI는 단기 트레이딩과 중장기 포트폴리오 관리를 통합한 AI 기반 자산 운용 시스템입니다.

### 🎯 핵심 특징

- **이중 AI Agent 구조**: 독립적이면서 협조적인 두 AI 에이전트
- **통합 데이터 레이어**: 뉴스 감정, 거시 이벤트, 다자산 시장 데이터 일원화
- **리소스 효율적**: VPS 환경에서 실행 가능한 경량화 설계
- **설명 가능한 AI**: TFT 기반 예측 결과 해석 가능

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     SharedCore (공통 데이터 레이어)         │
├─────────────────────────────────────────────────────────────┤
│  📊 Market Data    │  📰 Sentiment     │  📅 Events        │
│  • Multi-asset     │  • FinBERT       │  • FOMC/CPI       │
│  • Real-time cache │  • Social media  │  • Economic cal   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
            ┌───────▼───────┐   ┌───────▼───────┐
            │   AuroraQ     │   │   MacroQ      │
            │ (단기 매매)   │   │ (중장기 투자) │
            └───────────────┘   └───────────────┘
```

## 🚀 주요 구성 요소

### 1. **SharedCore** - 공통 데이터 인프라
- **통합 데이터 제공자**: 모든 시장 데이터를 읽기 전용으로 제공
- **감정 분석 엔진**: 배치 처리 기반 효율적 뉴스 감정 분석
- **이벤트 캘린더**: 거시경제 이벤트 자동 수집 및 정량화
- **리스크 관리**: 통합 리스크 모니터링 및 제어

### 2. **AuroraQ** - 단기 트레이딩 AI Agent
- **전략**: PPO 강화학습 + Rule 기반 전략 (A~E)
- **타임프레임**: 1분 ~ 1시간
- **대상**: 암호화폐 (Binance)
- **특징**: 실시간 뉴스 감정 기반 진입/청산

### 3. **MacroQ** - 중장기 포트폴리오 AI Agent
- **전략**: TFT 기반 멀티호라이즌 예측
- **타임프레임**: 1일 ~ 3개월
- **대상**: 다자산 (주식, ETF, 채권, 암호화폐)
- **특징**: 리스크 패리티 기반 포트폴리오 최적화

## 🛠️ 기술 스택

### Core Technologies
- **Python 3.8+**: 메인 언어
- **PyTorch**: 딥러닝 프레임워크 (TFT, PPO)
- **Redis**: 고성능 캐싱
- **Docker**: 컨테이너화 배포

### AI/ML Libraries
- **Transformers**: FinBERT 감정 분석
- **CVXPY**: 포트폴리오 최적화
- **Gymnasium**: 강화학습 환경

### Data Sources
- **Binance API**: 암호화폐 데이터
- **Yahoo Finance**: 주식/ETF 데이터
- **FRED API**: 거시경제 지표
- **Feedly/Reddit**: 뉴스 및 소셜 감정

## 📦 설치 가이드

### 1. 기본 요구사항
```bash
# Python 3.8 이상
python --version

# Redis 설치 (선택사항)
sudo apt-get install redis-server  # Linux
brew install redis                  # macOS
```

### 2. 프로젝트 클론 및 의존성 설치
```bash
git clone https://github.com/yourusername/QuantumAI.git
cd QuantumAI

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# 의존성 설치
pip install -r requirements/base.txt
```

### 3. 환경 설정
```bash
# .env 파일 생성
cp configs/environment/.env.template .env

# 필수 설정 입력
# - BINANCE_API_KEY
# - BINANCE_API_SECRET
# - GITHUB_TOKEN (옵션)
```

## 🚀 빠른 시작

### AuroraQ 실행 (단기 트레이딩)
```python
from AuroraQ import AuroraQAgent
from SharedCore import UnifiedDataProvider

# 데이터 제공자 초기화
data_provider = UnifiedDataProvider()
await data_provider.connect()

# AuroraQ 에이전트 시작
agent = AuroraQAgent(data_provider=data_provider)
await agent.start_trading()
```

### MacroQ 실행 (중장기 포트폴리오)
```python
from MacroQ import MacroQAgent
from SharedCore import UnifiedDataProvider

# MacroQ 에이전트 시작
macro_agent = MacroQAgent(data_provider=data_provider)

# 포트폴리오 최적화 실행
optimal_weights = await macro_agent.optimize_portfolio()
print(f"Optimal allocation: {optimal_weights}")
```

## 📊 성능 및 리소스

### 리소스 요구사항
- **최소**: 2 CPU cores, 4GB RAM
- **권장**: 4 CPU cores, 8GB RAM
- **스토리지**: 10GB+ (데이터 캐싱)

### 성능 지표
- **AuroraQ**: 초당 100+ 틱 처리
- **MacroQ**: 5개 자산 TFT 예측 < 1초
- **감정 분석**: 배치당 32개 뉴스 < 2초

## 🔧 고급 설정

### 1. 학습/운영 환경 분리
```yaml
# 학습 환경 (로컬/클라우드)
training:
  device: "cuda"  # GPU 사용
  batch_size: 64
  epochs: 100

# 운영 환경 (VPS)
production:
  device: "cpu"  # CPU만 사용
  model: "quantized"  # 압축 모델
  cache_ttl: 300  # 5분 캐시
```

### 2. 리스크 관리 설정
```python
# 통합 리스크 한도
TOTAL_RISK_LIMIT = 100000  # $100k
MAX_POSITION_SIZE = 0.2    # 20% per position
MAX_DRAWDOWN = 0.15        # 15% 최대 낙폭
```

## 📈 로드맵

### Phase 1 (현재 - 2개월)
- [x] SharedCore 데이터 레이어 구축
- [x] AuroraQ 통합
- [x] MacroQ 기본 구조
- [ ] 배치 감정 분석 시스템

### Phase 2 (2-4개월)
- [ ] TFT 모델 학습 파이프라인
- [ ] 포트폴리오 최적화 고도화
- [ ] 실시간 모니터링 대시보드

### Phase 3 (4-6개월)
- [ ] 분산 처리 (Ray/Dask)
- [ ] 자동 하이퍼파라미터 튜닝
- [ ] 프로덕션 배포 자동화

## 🤝 기여 가이드

프로젝트에 기여하고 싶으시다면:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

- **프로젝트 관리자**: QuantumAI Team
- **이메일**: contact@quantumai.ai
- **GitHub**: [https://github.com/quantumai](https://github.com/quantumai)

---

**⚠️ 투자 경고**: 이 시스템은 교육 및 연구 목적으로 개발되었습니다. 실제 투자에 사용하기 전에 충분한 테스트와 검증을 거쳐야 하며, 모든 투자 결정은 사용자의 책임입니다.