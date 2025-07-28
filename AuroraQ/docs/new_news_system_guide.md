# 새로운 무료 뉴스 수집 시스템 가이드

## 🎯 개요

Feedly 비용 문제로 인해 새로운 무료 뉴스 수집 시스템을 구축했습니다. 다음과 같은 5개의 무료 데이터 소스를 활용합니다:

### 🆓 완전 무료 (API 키 불필요)
1. **Google News RSS** - 헤드라인 + 키워드 뉴스
2. **Yahoo Finance RSS** - 금융 및 자산 관련 뉴스  
3. **Reddit API** - 커뮤니티 감정과 트렌드 감지

### 🔑 무료 티어 (API 키 필요)
4. **NewsAPI.org** - 글로벌 뉴스 (속보, 주요 인물, 이벤트)
5. **Finnhub** - 경제 캘린더 (FOMC, CPI) + 금융 뉴스

## 🚀 시스템 구조

```
새로운 뉴스 수집 시스템
├── Google News RSS ──┐
├── Yahoo Finance ────┤
├── Reddit API ───────┼──→ 통합 수집기 ──→ 감정 분석 ──→ 점수화
├── NewsAPI.org ──────┤
└── Finnhub ─────────┘
```

## 📋 주요 기능

### 1. 뉴스 카테고리
- **헤드라인**: 주요 헤드라인 기사
- **암호화폐**: BTC, ETH, DeFi, 규제 뉴스
- **금융**: 주식, 시장, 경제 뉴스
- **거시경제**: FOMC, CPI, NFP, GDP 이벤트
- **속보**: 실시간 긴급 뉴스
- **주요 인물**: CEO, 정치인, 경제 인사

### 2. 감정 분석
- 키워드 기반 감정 점수화
- 커뮤니티 감정 반영 (Reddit)
- 시장 영향도 평가
- 실시간 트렌드 분석

### 3. 데이터 집계
- 중복 제거 및 관련성 점수
- 시간별 트렌드 분석
- 소스별 신뢰도 가중치
- 개인화된 뉴스 피드

## 🔧 설정 방법

### 1. 환경 변수 설정

`.env` 파일에 다음 API 키 추가 (선택사항):

```bash
# 필수 아님 - 무료 티어 API 키들 (선택적으로 추가하면 더 많은 데이터 수집 가능)

# NewsAPI (무료: 100 requests/day)
NEWSAPI_KEY=your_newsapi_key_here

# Finnhub (무료: 60 requests/minute)  
FINNHUB_API_KEY=your_finnhub_key_here
```

### 2. 무료 API 키 발급

#### NewsAPI.org
1. https://newsapi.org/register 방문
2. 무료 계정 생성
3. API 키 복사
4. 무료 티어: **100 requests/day**

#### Finnhub
1. https://finnhub.io/register 방문  
2. 무료 계정 생성
3. API 키 복사
4. 무료 티어: **60 calls/minute**

**참고**: API 키가 없어도 Google News, Yahoo Finance, Reddit으로 충분한 뉴스 수집이 가능합니다!

## 💻 사용법

### 1. 기본 사용 (기존 코드와 호환)

```python
from SharedCore.sentiment_engine.news_collectors.news_collector import NewsCollector

# 기존 방식과 동일하게 사용 가능
collector = NewsCollector()
await collector.connect()

# 암호화폐 뉴스 수집
crypto_news = await collector.get_latest_crypto_news(hours_back=24)

# 감정 분석
sentiment_summary = await collector.get_sentiment_summary(crypto_news)

# 속보 확인
breaking_news = await collector.get_breaking_news(minutes=30)
```

### 2. 새로운 고급 기능

```python
from SharedCore.data_collection.news_aggregation_system import AuroraQNewsAggregator
from SharedCore.data_collection.base_collector import NewsCategory

aggregator = AuroraQNewsAggregator()

# 포괄적 뉴스 수집
news_data = await aggregator.collect_comprehensive_news(
    categories=[NewsCategory.CRYPTO, NewsCategory.FINANCE, NewsCategory.MACRO],
    hours_back=6,
    articles_per_category=20
)

# 시장 영향 뉴스 분석
market_news = await aggregator.get_market_moving_news(minutes=30)

# 감정 트렌드 분석
sentiment_trends = await aggregator.analyze_sentiment_trends(hours=24)

# 개인화된 뉴스 피드
personalized_feed = await aggregator.get_personalized_feed(
    interests=["bitcoin", "federal reserve", "tesla"],
    count=50
)
```

## 📊 데이터 소스별 특징

### Google News RSS
- **장점**: 무료, 안정적, 다량의 헤드라인
- **수집량**: 제한 없음
- **업데이트**: 실시간
- **특화**: 전 세계 주요 뉴스

### Yahoo Finance  
- **장점**: 금융 특화, 티커별 뉴스
- **수집량**: 제한 없음
- **업데이트**: 실시간
- **특화**: 주식, 암호화폐, 경제 데이터

### Reddit
- **장점**: 커뮤니티 감정, 실시간 트렌드
- **수집량**: 제한적 (rate limit)
- **업데이트**: 실시간
- **특화**: 소셜 센티멘트, 밈 추적

### NewsAPI.org (무료 티어)
- **장점**: 글로벌 소스, 검색 기능
- **수집량**: 100 requests/day
- **업데이트**: 15분 지연
- **특화**: 국제 뉴스, 속보

### Finnhub (무료 티어)
- **장점**: 경제 캘린더, 금융 데이터
- **수집량**: 60 calls/minute  
- **업데이트**: 실시간
- **특화**: FOMC, CPI, 경제 지표

## 🔍 수집 가능한 뉴스 유형

### 주요 헤드라인
- 전 세계 주요 뉴스 (Google News)
- 국가별/지역별 헤드라인
- 카테고리별 분류 (정치, 경제, 기술)

### 암호화폐 뉴스
- BTC, ETH 가격 및 시장 분석
- DeFi, NFT 트렌드
- 규제 뉴스 (SEC, 각국 정부)
- 거래소 및 기업 뉴스

### 거시경제 이벤트
- **FOMC**: 연준 회의 및 정책 결정
- **CPI**: 인플레이션 데이터
- **NFP**: 고용 통계
- **GDP**: 경제 성장률
- **중앙은행**: ECB, BOJ, PBoC 정책

### 금융 시장
- 주식 시장 동향
- 기업 실적 발표
- IPO 및 M&A 뉴스
- 원자재 가격 동향

### 속보 및 긴급 뉴스
- 시장 급변 상황
- 정책 발표
- 기업 공시
- 지정학적 이벤트

## 📈 성능 특징

### 수집 속도
- **API 키 없음**: 분당 ~200 기사
- **API 키 있음**: 분당 ~500 기사
- **병렬 처리**: 5개 소스 동시 수집

### 데이터 품질
- **중복 제거**: URL 기반 자동 필터링
- **관련성 점수**: 키워드 매칭 알고리즘
- **신뢰도 가중치**: 소스별 차등 적용

### 시스템 안정성
- **무료 소스 우선**: API 장애에도 작동
- **자동 폴백**: 오류 시 대체 소스 활용
- **캐싱**: 5-15분 결과 캐시

## 🚨 주의사항 및 제한사항

### API 제한사항
- **NewsAPI**: 무료 티어는 개발용만 (상용 제한)
- **Finnhub**: 분당 60 요청 제한
- **Reddit**: 비공식 API 사용으로 안정성 제한

### 권장사항
1. **API 키 선택적 사용**: 필수는 아니지만 있으면 더 풍부한 데이터
2. **캐싱 활용**: 동일한 쿼리 반복 방지
3. **로그 모니터링**: 수집 실패 시 알림 설정
4. **백업 전략**: 주요 소스 장애 시 대체 방안

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 뉴스 수집이 안됨
```bash
# 시스템 상태 확인
python -c "
import asyncio
from SharedCore.data_collection.news_aggregation_system import AuroraQNewsAggregator

async def check():
    agg = AuroraQNewsAggregator()
    health = await agg.get_system_health()
    print(f'Status: {health[\"status\"]}')
    print(f'Active collectors: {health[\"active_collectors\"]}/{health[\"total_collectors\"]}')
    await agg.close_all()

asyncio.run(check())
"
```

#### 2. API 키 오류
- `.env` 파일에 올바른 키가 있는지 확인
- 키 없이도 Google News, Yahoo Finance, Reddit으로 작동

#### 3. 속도 문제
- 캐시 TTL을 늘려서 반복 요청 줄이기
- 필요한 카테고리만 수집하도록 설정

## 📊 모니터링 및 통계

### 수집 통계 확인
```python
# 시스템 상태 모니터링
health = await aggregator.get_system_health()
print(f"전체 상태: {health['status']}")
print(f"활성 수집기: {health['active_collectors']}")
print(f"수집된 기사: {health['collection_stats']['total_articles']}")

# 카테고리별 통계
for category, count in health['collection_stats']['by_category'].items():
    print(f"{category}: {count}개 기사")
```

### 성능 모니터링
```python
# 각 수집기 개별 상태
for name, collector in aggregator.collectors.items():
    stats = collector.get_stats()
    print(f"{name}: {stats['requests_made']} 요청, {stats['articles_collected']} 기사")
```

## 🎯 다음 개발 계획

1. **AI 기반 감정 분석**: 키워드 기반에서 LLM 기반으로 업그레이드
2. **실시간 알림**: 중요 뉴스 발생 시 Telegram 알림
3. **트렌드 예측**: 뉴스 패턴 기반 시장 예측
4. **다국어 지원**: 한국어, 일본어 뉴스 소스 추가
5. **웹 대시보드**: 뉴스 수집 현황 실시간 모니터링

---

## 🚀 시작하기

1. **의존성 설치**: 기존 requirements.txt 사용
2. **API 키 설정**: 선택적으로 NewsAPI, Finnhub 키 추가
3. **테스트 실행**: `python SharedCore/sentiment_engine/news_collectors/news_collector.py`
4. **기존 코드**: 수정 없이 바로 사용 가능!

새로운 시스템은 기존 코드와 100% 호환되면서도 더 안정적이고 비용 효율적입니다. 🎉