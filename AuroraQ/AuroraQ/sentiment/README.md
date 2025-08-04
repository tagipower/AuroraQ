# AuroraQ Sentiment Analysis Service

**Version 2.0.0** - VPS 최적화 센티먼트 분석 서비스  
**Real-time financial sentiment analysis with ONNX FinBERT integration**

---

## 📋 Overview

AuroraQ Sentiment Service는 암호화폐 및 금융 시장을 위한 실시간 센티먼트 분석 시스템입니다. VPS 환경에 최적화되어 제한된 리소스에서도 안정적으로 작동하며, 다양한 뉴스 소스로부터 데이터를 수집하여 ONNX 기반 FinBERT 모델로 분석합니다.

### 🎯 주요 기능
- **실시간 뉴스 수집**: Google News, Yahoo Finance, NewsAPI, Finnhub 연동
- **AI 기반 감정 분석**: ONNX FinBERT 모델 + 키워드 기반 폴백
- **VPS 최적화**: 메모리 3GB 제한, 동적 배치 크기 조정
- **이벤트 감지**: 주요 경제/암호화폐 이벤트 자동 감지
- **성능 모니터링**: 실시간 시스템 리소스 추적
- **RESTful API**: 거래 시스템과의 쉬운 연동

---

## 🗂️ 폴더 구조

```
AuroraQ/sentiment/
├── 📁 api/                           # API 엔드포인트
│   ├── __init__.py
│   └── metrics_router.py              # 메트릭 API 라우터
│
├── 📁 collectors/                     # 데이터 수집기
│   ├── enhanced_news_collector_v2.py  # 향상된 뉴스 수집기 (v2)
│   └── macro_indicator_collector.py   # 거시경제 지표 수집기
│
├── 📁 config/                         # 설정 관리
│   └── sentiment_service_config.py    # 서비스 설정 클래스
│
├── 📁 deployment/                     # 배포 스크립트
│   ├── deploy.sh                      # VPS 배포 스크립트
│   └── service_runner.py              # 서비스 메인 실행기
│
├── 📁 models/                         # AI 모델 및 분석기
│   ├── __init__.py
│   ├── advanced_keyword_scorer_vps.py # VPS 최적화 키워드 스코어러
│   └── keyword_scorer.py              # 통합 키워드 스코어링 시스템
│
├── 📁 monitors/                       # 모니터링 시스템
│   └── option_expiry_monitor.py       # 옵션 만료 모니터
│
├── 📁 processors/                     # 데이터 처리기
│   ├── __init__.py
│   ├── advanced_fusion_manager_vps.py # 고급 융합 매니저 (VPS)
│   ├── big_event_detector_v2.py       # 대형 이벤트 감지기 (v2)
│   ├── event_impact_manager.py        # 이벤트 영향도 매니저
│   ├── finbert_batch_processor_v2.py  # FinBERT 배치 프로세서 (v2)
│   ├── scheduled_event_fusion.py      # 스케줄된 이벤트 융합
│   └── sentiment_fusion_manager_v2.py # 센티먼트 융합 매니저 (v2)
│
├── 📁 schedulers/                     # 작업 스케줄러
│   ├── batch_scheduler_v2.py          # 배치 스케줄러 (v2)
│   └── event_schedule_loader.py       # 이벤트 스케줄 로더
│
├── 📄 .env                            # 환경변수 설정 파일
├── 📄 __init__.py                     # 패키지 초기화
├── 📄 README.md                       # 기본 문서
├── 📄 README_COMPREHENSIVE.md         # 상세 문서 (이 파일)
└── 📄 SENTIMENT_SERVICE_ANALYSIS.md   # 서비스 분석 보고서
```

---

## 🏗️ 핵심 구성요소

### 1. **Data Collectors** (데이터 수집기)

#### **EnhancedNewsCollectorV2**
```python
# 주요 기능
- Google News RSS 수집
- Yahoo Finance RSS 수집  
- NewsAPI 통합
- Finnhub 금융 뉴스 수집
- Rate limiting 및 중복 제거
- VPS 최적화 (동시 요청 3개 제한)
```

#### **MacroIndicatorCollector**
```python
# 수집 데이터
- VIX 지수 (공포 지수)
- DXY (달러 지수)
- 금 가격
- 국채 수익률
- S&P 500 지수
```

### 2. **AI Processors** (AI 처리기)

#### **FinBERTBatchProcessorV2**
```python
# ONNX FinBERT 통합
- ProsusAI/finbert 모델 사용
- 동적 배치 크기 조정 (2-12개)
- 메모리 효율적 처리
- CPU 최적화 (2코어 제한)
- 키워드 기반 폴백 시스템
```

#### **SentimentFusionManagerV2**
```python
# 다중 소스 융합
- FinBERT 결과 (가중치 0.6)
- 키워드 스코어 (가중치 0.4)  
- 기술적 지표 (가중치 0.3)
- 뉴스 소스 신뢰도 조정
- 이상치 감지 및 필터링
```

### 3. **Event Detection** (이벤트 감지)

#### **BigEventDetectorV2**
```python
# 감지 이벤트
- FOMC 회의 및 금리 결정
- CPI/PPI 발표
- 고용 지표 발표
- 암호화폐 규제 뉴스
- 거래소 해킹/문제
- 주요 기업 암호화폐 채택
```

#### **EventImpactManager**
```python
# 영향도 분석
- 즉시 (immediate): 0초 지연
- 높음 (high): 5분 지연
- 보통 (normal): 15분 지연  
- 낮음 (low): 30분 지연
```

### 4. **Configuration System** (설정 시스템)

#### **환경변수 기반 설정**
```bash
# 서비스 기본 설정
SERVICE_NAME=auroaq-sentiment-service
SERVICE_VERSION=2.0.0
DEPLOYMENT_MODE=vps

# VPS 리소스 제한
MAX_MEMORY_MB=3072
MAX_CONCURRENT_REQUESTS=3
MAX_BATCH_SIZE=12

# API 키
NEWSAPI_KEY=your_newsapi_key
FINNHUB_KEY=your_finnhub_key
```

---

## 🚀 Quick Start

### 1. **환경 설정**
```bash
# 의존성 설치
pip install aiohttp backoff beautifulsoup4 transformers torch psutil

# 환경변수 설정
cp .env.example .env
# .env 파일에서 API 키 설정

# 데이터베이스 설정 (선택사항)
# PostgreSQL 및 Redis 설정
```

### 2. **서비스 실행**
```bash
# 개발 모드
python deployment/service_runner.py

# VPS 배포
chmod +x deployment/deploy.sh
sudo ./deployment/deploy.sh
```

### 3. **API 사용 예시**
```python
import asyncio
from sentiment.models.advanced_keyword_scorer_vps import analyze_sentiment_vps

async def test_sentiment():
    result = await analyze_sentiment_vps(
        "Bitcoin surges to new highs amid institutional adoption"
    )
    print(f"Sentiment: {result['sentiment']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Label: {result['label']}")

asyncio.run(test_sentiment())
```

---

## 📊 성능 및 리소스

### **VPS 최적화 사양**
| 구분 | 사양 | 설정값 |
|------|------|---------|
| **메모리** | 최대 사용량 | 3,072MB |
| **CPU** | 스레드 제한 | 2 스레드 |
| **동시 요청** | API 호출 제한 | 3개 |
| **배치 크기** | 동적 조정 | 2-12개 |
| **처리 간격** | 뉴스 수집 | 5분 |
| **처리 간격** | FinBERT 분석 | 15분 |

### **성능 메트릭**
- **뉴스 처리**: 100+ 기사/분
- **센티먼트 분석**: 500+ 텍스트/분 (배치)
- **응답 시간**: <200ms (API 호출)
- **정확도**: >85% (금융 텍스트)
- **가동 시간**: >99.5% 목표

---

## 🔧 API Reference

### **센티먼트 분석 API**
```python
# 단일 텍스트 분석
POST /api/v1/sentiment/analyze
{
    "text": "Bitcoin price surges amid bullish market sentiment",
    "metadata": {"source": "news", "symbol": "BTC"}
}

# 응답
{
    "sentiment": 0.85,
    "confidence": 0.92,
    "label": "positive",
    "keywords": ["bitcoin", "surges", "bullish"],
    "processing_time": 0.045
}
```

### **배치 분석 API**
```python
# 여러 텍스트 동시 분석
POST /api/v1/sentiment/batch
{
    "texts": [
        "Bitcoin rallies strongly",
        "Market crash fears grow",
        "Stable trading conditions"
    ],
    "batch_size": 8
}
```

### **시스템 메트릭 API**
```python
# 성능 통계 조회
GET /api/v1/metrics/performance
{
    "cpu_usage": 15.2,
    "memory_usage": 2048.5,
    "processed_items": 1250,
    "success_rate": 98.7
}
```

---

## 🔍 모니터링 및 디버깅

### **로그 시스템**
```bash
# 서비스 로그
journalctl -u auroaq-sentiment-service -f

# 파일 로그
tail -f /var/log/auroaq/sentiment_service.log

# 에러 로그만 필터링
grep "ERROR" /var/log/auroaq/sentiment_service.log
```

### **헬스 체크**
```bash
# 서비스 상태 확인
curl http://localhost:8080/health

# 상세 메트릭
curl http://localhost:8081/metrics
```

### **성능 분석**
```python
# 프로파일링 활성화
ENABLE_PROFILING=true
PROFILING_OUTPUT_PATH=./profiling/

# 메모리 사용량 분석
from sentiment.models.advanced_keyword_scorer_vps import get_vps_performance_stats
stats = await get_vps_performance_stats()
```

---

## 🚨 트러블슈팅

### **일반적인 문제**

#### **1. 메모리 부족 오류**
```bash
# 증상: OutOfMemoryError, 프로세스 강제 종료
# 해결책:
export MAX_MEMORY_MB=2048
export MAX_BATCH_SIZE=6
sudo systemctl restart auroaq-sentiment-service
```

#### **2. API 레이트 리미트**
```bash
# 증상: HTTP 429 오류, API 응답 실패
# 해결책:
export NEWSAPI_REQUESTS_PER_HOUR=50
export FINNHUB_REQUESTS_PER_HOUR=30
```

#### **3. FinBERT 모델 로딩 실패**
```bash
# 증상: 모델 다운로드/로딩 오류
# 해결책:
pip install --upgrade transformers torch
# 또는 키워드 기반 폴백 사용
export ENABLE_ONNX_OPTIMIZATION=false
```

### **로그 분석**
```bash
# 에러 패턴 분석
grep -E "(ERROR|CRITICAL)" /var/log/auroaq/sentiment_service.log | tail -20

# 성능 이슈 확인
grep "processing_time" /var/log/auroaq/sentiment_service.log | tail -10

# API 실패 확인
grep "failed" /var/log/auroraQ/sentiment_service.log | tail -10
```

---

## 🔐 보안 고려사항

### **API 키 보안**
- ✅ 환경변수 사용 (하드코딩 금지)
- ✅ `.env` 파일 Git 제외
- ✅ API 키 정기 로테이션
- ✅ 최소 권한 원칙 적용

### **네트워크 보안**
- ✅ HTTPS 사용 강제
- ✅ CORS 설정 제한
- ✅ Rate limiting 적용
- ✅ 입력 데이터 검증

### **시스템 보안**
- ✅ 로그 민감정보 마스킹
- ✅ 프로세스 권한 최소화
- ✅ 정기 보안 업데이트
- ✅ 시스템 리소스 모니터링

---

## 📈 성능 최적화 팁

### **VPS 환경 최적화**
1. **메모리 관리**
   ```python
   # 가비지 컬렉션 주기 조정
   FINBERT_GC_INTERVAL=300
   
   # 배치 크기 자동 조정
   BATCH_SIZE_AUTO_ADJUSTMENT=true
   ```

2. **CPU 최적화**
   ```python
   # 스레드 수 제한
   ONNX_THREAD_COUNT=2
   THREAD_POOL_WORKERS=2
   ```

3. **네트워크 최적화**
   ```python
   # 연결 풀 설정
   HTTP_POOL_MAXSIZE=10
   HTTP_RETRIES=3
   HTTP_BACKOFF_FACTOR=0.3
   ```

### **캐시 전략**
```python
# 결과 캐싱 활성화
ENABLE_RESULT_CACHING=true
CACHE_TTL=300

# 사전 로딩
CACHE_PRELOAD_PATTERNS=BTCUSDT,ETHUSDT
```

---

## 🤝 기여 가이드라인

### **개발 환경 설정**
```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# 테스트 실행
python -m pytest tests/

# 코드 품질 검사
flake8 sentiment/
black sentiment/
```

### **커밋 컨벤션**
```bash
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 업데이트
perf: 성능 개선
refactor: 코드 리팩토링
test: 테스트 추가/수정
```

---

## 📞 지원 및 문의

### **기술 지원**
- 📧 이메일: support@auroaq.com
- 📚 문서: [AuroraQ Documentation](https://docs.auroaq.com)
- 🐛 버그 리포트: [GitHub Issues](https://github.com/auroaq/sentiment/issues)

### **커뮤니티**
- 💬 Discord: [AuroraQ Community](https://discord.gg/auroaq)
- 📱 Telegram: [@AuroraQSupport](https://t.me/AuroraQSupport)

---

## 📄 라이센스

이 프로젝트는 AuroraQ 독점 라이센스 하에 있습니다.  
상업적 사용 및 배포에 대한 문의는 legal@auroaq.com으로 연락해 주세요.

---

**© 2024 AuroraQ Team. All rights reserved.**