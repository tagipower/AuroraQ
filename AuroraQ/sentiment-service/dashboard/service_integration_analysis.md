# AuroraQ 센티멘트 서비스 ↔ 대시보드 통합 분석

## 📋 개요

현재 AuroraQ 센티멘트 서비스와 터미널 대시보드 간의 통합 상태를 분석하고, 데이터 흐름과 점수 계산 전체 구조를 문서화합니다.

## 🏗️ 서비스 아키텍처 구조

### 1. 센티멘트 서비스 핵심 컴포넌트

#### **A. 데이터 수집 레이어**
```
📰 Enhanced News Collector
├─ RSS/API 기반 뉴스 수집
├─ 필터링 및 중복 제거
└─ 메타데이터 추출

📱 Reddit Collector  
├─ Reddit API 연동
├─ 서브레딧별 수집
└─ 실시간 감정 추출

🔍 Power Search Engine
├─ 다중 소스 검색
├─ 키워드 기반 수집
└─ 관련성 스코어링
```

#### **B. 감정 분석 레이어**
```
⚡ Keyword Scorer (실시간)
├─ 0.5초 내 응답
├─ 4개 카테고리 분석
│   ├─ Price (40%): surge, rally, crash, dump
│   ├─ Institutional (30%): approval, ban, ETF
│   ├─ Sentiment (20%): optimistic, fear, panic
│   └─ Technical (10%): golden cross, death cross
└─ -1.0 ~ 1.0 점수 출력

🤖 FinBERT Batch Processor
├─ 딥러닝 기반 정확한 분석
├─ 배치 처리 (느림, 정확함)
├─ 0.0 ~ 1.0 점수 출력
└─ 캐시를 통한 결과 저장

🔄 Sentiment Fusion Manager
├─ 실시간 + 배치 결과 융합
├─ 적응형 가중치 시스템
├─ 이상치 제거 (Z-score)
├─ 품질/신뢰성 점수 계산
└─ 최종 융합 점수 생성
```

#### **C. API 서비스 레이어**
```
🌐 FastAPI 서버 (포트 8080)
├─ /api/v1/sentiment/* - 기본 감정 분석
├─ /api/v1/fusion/* - 융합 감정 분석
├─ /api/v1/events/* - 이벤트 감지
├─ /api/v1/trading/* - 트레이딩 신호
├─ /api/v1/scheduler/* - 배치 작업
└─ /health - 헬스체크

📊 메트릭스 & 모니터링
├─ Prometheus 메트릭
├─ 구조화된 로깅
├─ Redis 캐싱
└─ 성능 모니터링
```

## 🔄 데이터 흐름 및 점수 계산 과정

### **Phase 1: 데이터 수집**
```
External Sources → Collectors → Content Cache
     ↓
1. 뉴스/소셜 미디어에서 원시 데이터 수집
2. 중복 제거 및 관련성 필터링
3. 메타데이터 추출 (키워드, 엔티티)
4. Redis 캐시에 저장 (content_hash 기준)
```

### **Phase 2: 실시간 감정 분석**
```
Text Input → KeywordScorer → SentimentSignal
     ↓
1. 텍스트 전처리 (소문자, 특수문자 제거)
2. 4개 카테고리별 키워드 매칭:
   - Price (0.4): surge(+0.8), crash(-0.9)
   - Institutional (0.3): approval(+0.8), ban(-0.9)  
   - Sentiment (0.2): optimistic(+0.6), fear(-0.7)
   - Technical (0.1): golden(+0.7), death(-0.8)
3. 카테고리별 점수 계산 후 가중 평균
4. 최종 점수: -1.0 ~ 1.0
```

### **Phase 3: 배치 감정 분석**
```
Cached Content → FinBERT → Processed Results
     ↓
1. 배치 스케줄러가 큐된 컨텐츠 처리
2. FinBERT 모델로 정확한 감정 분석
3. 0.0 ~ 1.0 점수 생성 
4. 캐시에 결과 저장 (content_hash 매핑)
```

### **Phase 4: 감정 융합**
```
KeywordScore + FinBERT → Fusion Manager → Final Score
     ↓
1. 신호 수집:
   - 실시간 키워드 점수 (-1~1)
   - FinBERT 배치 결과 (0~1 → -1~1 변환)
2. 융합 방법 선택:
   - ADAPTIVE: 신호 품질에 따라 자동 선택
   - CONFIDENCE_BASED: 신뢰도 기반 가중치
   - WEIGHTED_AVERAGE: 단순 가중 평균
3. 이상치 제거 (Z-score > 3.0)
4. 최종 점수 계산:
   - 가중 평균: Σ(score_i × weight_i)
   - 신뢰도: Σ(confidence_i × weight_i)
5. 품질/신뢰성 점수 계산
```

## 🎯 현재 대시보드 통합 상태

### **✅ 구현된 부분**

#### **1. 대시보드 구조**
- 6개 패널 레이아웃 (2x3)
- Claude Code 스타일 타이핑 효과
- 실시간 데이터 변화 감지
- ANSI 컬러 및 ASCII 박스

#### **2. 패널 구성**
```
📊 Sentiment Panel    📅 Events Panel      📈 Strategy Panel
🔗 API Status Panel   💻 VPS Resources     🚨 System Alerts
```

### **❌ 통합 갭 분석**

#### **1. 데이터 연결 부재**
현재 대시보드는 **Mock 데이터**를 사용하고 있음:
```python
# 현재 (Mock)
sentiment_data = {
    'fusion_score': getattr(self, '_mock_sentiment', 0.0),
    'components': {
        'news': np.random.uniform(-1, 1),
        'reddit': np.random.uniform(-1, 1),
        # ...
    }
}

# 필요한 실제 연결
async def get_real_sentiment_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('http://localhost:8080/api/v1/fusion/fuse') as resp:
            return await resp.json()
```

#### **2. API 엔드포인트 매핑**
| 대시보드 패널 | 필요한 API | 현재 상태 |
|-------------|-----------|----------|
| 📊 Sentiment | `/api/v1/fusion/fuse` | ❌ Mock |
| 📅 Events | `/api/v1/events/timeline` | ❌ Mock |
| 📈 Strategy | `/api/v1/trading/performance` | ❌ Mock |
| 🔗 API Status | `/health`, `/metrics` | ❌ Mock |
| 💻 VPS Resources | `psutil` (실제 구현됨) | ✅ Real |
| 🚨 System Alerts | 서비스 상태 API | ❌ Mock |

#### **3. 데이터 형식 불일치**
```python
# 서비스 응답 (FusionResponse)
{
    "fused_score": 0.742,        # 0.0 ~ 1.0
    "confidence": 0.85,
    "trend": "strong_bullish",
    "raw_scores": {...},
    "weights_used": {...}
}

# 대시보드 기대 형식
{
    "fusion_score": 0.484,       # -1.0 ~ 1.0
    "components": {
        "news": 0.6,
        "reddit": 0.3,
        "technical": 0.1,
        "market": 0.0
    }
}
```

## 🔧 통합을 위한 필요 작업

### **1. 즉시 필요한 작업**
```python
# A. HTTP 클라이언트 추가
self.session = aiohttp.ClientSession()
self.api_base_url = "http://localhost:8080"

# B. 실제 API 호출 함수
async def fetch_sentiment_data(self):
    async with self.session.get(f"{self.api_base_url}/api/v1/fusion/statistics/BTCUSDT") as resp:
        return await resp.json()

# C. 데이터 형식 변환
def convert_fusion_response(self, api_response):
    # 0~1 → -1~1 변환
    fusion_score = api_response['fused_score'] * 2 - 1
    return {
        'fusion_score': fusion_score,
        'components': api_response.get('raw_scores', {}),
        'confidence': api_response['confidence']
    }
```

### **2. 미들웨어 레이어 추가**
```python
class SentimentServiceClient:
    """센티멘트 서비스 클라이언트"""
    
    async def get_fusion_sentiment(self, symbol: str = "BTCUSDT"):
        """융합 감정 점수 조회"""
        
    async def get_event_timeline(self):
        """이벤트 타임라인 조회"""
        
    async def get_trading_performance(self):
        """트레이딩 성과 조회"""
        
    async def get_service_health(self):
        """서비스 헬스 상태 조회"""
```

### **3. 에러 처리 및 폴백**
```python
async def get_sentiment_with_fallback(self):
    try:
        # 실제 API 호출
        return await self.fetch_real_sentiment()
    except Exception as e:
        logger.warning(f"API call failed: {e}, using mock data")
        # Mock 데이터로 폴백
        return self.generate_mock_sentiment()
```

## 📊 점수 계산 상세 흐름

### **최종 Fusion Score 계산식**
```
1. Keyword Score (-1~1) × Weight_keyword
2. FinBERT Score (0~1 → -1~1) × Weight_finbert  
3. Technical Score (-1~1) × Weight_technical
4. Social Score (-1~1) × Weight_social

Final_Score = Σ(Score_i × Weight_i) / Σ(Weight_i)

예시:
- Keyword: +0.6 × 0.4 = +0.24
- FinBERT: 0.8 → +0.6 × 0.6 = +0.36
- Result: (+0.24 + 0.36) / 1.0 = +0.60
```

### **신뢰도 및 품질 계산**
```
Confidence = Σ(Individual_Confidence_i × Weight_i)
Quality = (Signal_Count×0.3 + Avg_Confidence×0.4 + Consistency×0.2 + Diversity×0.1)
Reliability = (Avg_Confidence + Time_Factor + Source_Factor + Metadata_Factor) / 4
```

## 🚀 통합 로드맵

### **Phase 1: 기본 연결 (1-2일)**
1. HTTP 클라이언트 구현
2. 주요 API 엔드포인트 연결
3. 데이터 형식 변환 로직
4. 기본 에러 처리

### **Phase 2: 완전한 통합 (3-5일)**
1. 모든 패널 실제 데이터 연결
2. 실시간 업데이트 시스템
3. 캐싱 및 성능 최적화
4. 포괄적인 에러 처리

### **Phase 3: 고도화 (1주일)**
1. 자동 재연결 시스템
2. 데이터 검증 및 품질 관리
3. 성능 모니터링 대시보드
4. 알림 시스템 완성

현재 상태: **Phase 0 (Mock Data)** → Phase 1로 이동 필요