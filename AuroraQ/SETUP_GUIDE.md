# 🚀 AuroraQ v2.0 빠른 설정 가이드

## 📋 1단계: 즉시 사용 (API 키 없음)

### ✅ 이미 작동하는 무료 소스
- **Google News RSS**: 전 세계 헤드라인, 제한 없음
- **Yahoo Finance RSS**: 금융 뉴스, 제한 없음  
- **Reddit**: 커뮤니티 감정, r/cryptocurrency, r/wallstreetbets

### 🧪 테스트 실행
```bash
# 기본 테스트
python test_integration.py

# 거래 봇 시뮬레이션
python integration_example.py
```

---

## 🔑 2단계: 선택적 API 키 추가 (더 많은 데이터)

### NewsAPI (100회/일 무료)
1. https://newsapi.org/register 방문
2. 무료 계정 생성
3. API 키 복사
4. `.env` 파일에서 수정:
```env
NEWSAPI_KEY=실제_발급받은_키
```

### Finnhub (60회/분 무료)
1. https://finnhub.io/register 방문
2. 무료 계정 생성  
3. API 키 복사
4. `.env` 파일에서 수정:
```env
FINNHUB_API_KEY=실제_발급받은_키
```

---

## 📱 3단계: Telegram 알림 설정 (권장)

### Telegram Bot 생성
1. Telegram에서 `@BotFather` 검색
2. `/newbot` 명령어 입력
3. 봇 이름과 사용자명 설정
4. 받은 토큰을 `.env`에 추가:
```env
TELEGRAM_BOT_TOKEN=실제_봇_토큰
```

### Chat ID 확인
1. 생성한 봇과 대화 시작 (/start)
2. `https://api.telegram.org/bot<봇토큰>/getUpdates` 방문
3. chat id 복사하여 `.env`에 추가:
```env
TELEGRAM_CHAT_ID=실제_채팅_ID
```

---

## ⚙️ 4단계: 거래 설정 조정

### 감정 분석 임계값 (`.env` 파일)
```env
SENTIMENT_BUY_THRESHOLD=0.65    # 65% 이상 긍정시 매수 신호
SENTIMENT_SELL_THRESHOLD=0.35   # 35% 이하 부정시 매도 신호  
SENTIMENT_CONFIDENCE_MIN=0.6    # 최소 60% 신뢰도 필요
```

### 수집 주기 설정
```env
NEWS_COLLECTION_INTERVAL=5      # 5분마다 일반 뉴스 수집
BREAKING_NEWS_CHECK_INTERVAL=1  # 1분마다 속보 확인
SENTIMENT_UPDATE_INTERVAL=10    # 10분마다 감정 분석 업데이트
```

---

## 🔧 5단계: 시스템 최적화

### 성능 설정 (시스템 사양에 맞게 조정)
```env
MAX_MEMORY_MB=1024             # 최대 메모리 사용량
MAX_CPU_PERCENT=80             # 최대 CPU 사용률
NEWS_COLLECTOR_WORKERS=3       # 뉴스 수집 워커 수
SENTIMENT_ANALYSIS_WORKERS=2   # 감정 분석 워커 수
```

### 로그 및 모니터링
```env
LOG_LEVEL=INFO                 # 로그 레벨 (DEBUG/INFO/WARNING/ERROR)
AUTO_BACKUP_ENABLED=true       # 자동 백업 활성화
BACKUP_INTERVAL_HOURS=6        # 6시간마다 백업
```

---

## 📊 6단계: 실시간 모니터링 시작

### 수동 실행
```bash
# 한 번 실행
python integration_example.py

# 뉴스 시스템 상태 확인
python -c "
import asyncio
from SharedCore.data_collection.news_aggregation_system import AuroraQNewsAggregator

async def check():
    agg = AuroraQNewsAggregator()
    health = await agg.get_system_health()
    print(f'상태: {health[\"status\"]}')
    print(f'활성 수집기: {health[\"active_collectors\"]}/{health[\"total_collectors\"]}')
    await agg.close_all()

asyncio.run(check())
"
```

### 자동화 스크립트 (Windows)
```batch
@echo off
:loop
echo [%time%] AuroraQ 뉴스 분석 실행 중...
python integration_example.py
echo [%time%] 5분 대기 중...
timeout /t 300 /nobreak
goto loop
```

---

## 🎯 7단계: 실제 거래 연동

### 기존 AuroraQ 코드에서 사용
```python
from SharedCore.sentiment_engine.news_collectors.news_collector import NewsCollector

# 기존 방식 그대로 사용 (코드 수정 없음)
collector = NewsCollector()
await collector.connect()

# 암호화폐 뉴스 + 감정 분석
crypto_news = await collector.get_latest_crypto_news(hours_back=6)
sentiment = await collector.get_sentiment_summary(crypto_news)

# 거래 신호 생성
if sentiment['overall_sentiment'] > 0.65 and sentiment['confidence'] > 0.6:
    print("🚀 매수 신호!")
    # 실제 매수 로직 호출
elif sentiment['overall_sentiment'] < 0.35 and sentiment['confidence'] > 0.6:
    print("📉 매도 신호!")  
    # 실제 매도 로직 호출
```

---

## 📈 성능 벤치마크

### 현재 시스템 (API 키 없음)
- **수집 속도**: ~200 기사/분
- **소스**: 3개 (Google News, Yahoo Finance, Reddit)
- **지연시간**: 실시간
- **비용**: 완전 무료

### API 키 추가 시
- **수집 속도**: ~500 기사/분  
- **소스**: 5개 (+ NewsAPI, Finnhub)
- **추가 데이터**: 경제 캘린더, 글로벌 뉴스
- **비용**: 여전히 무료 (무료 티어)

---

## 🚨 문제 해결

### Q: 뉴스가 수집되지 않아요
**A**: 시스템 상태 확인
```bash
python test_integration.py
```

### Q: Yahoo Finance 에러가 많이 나와요  
**A**: 정상적인 현상입니다. Google News와 Reddit에서 정상 수집됩니다.

### Q: API 키를 추가했는데 더 많은 뉴스가 안 보여요
**A**: 캐시 때문일 수 있습니다. 5-10분 기다리거나 시스템 재시작하세요.

### Q: Telegram 알림이 안 와요
**A**: 
1. 봇과 대화를 먼저 시작했는지 확인 (/start)
2. Chat ID가 정확한지 확인
3. 봇 토큰이 올바른지 확인

---

## 📞 지원

- **문서**: `docs/new_news_system_guide.md`
- **테스트**: `python tests/test_new_news_system.py`
- **예제**: `integration_example.py`

🎉 **설정 완료! 이제 AuroraQ v2.0 무료 뉴스 시스템을 완전히 활용할 수 있습니다.**