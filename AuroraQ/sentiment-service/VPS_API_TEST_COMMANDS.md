# 🚀 VPS Sentiment Service API 테스트 명령어

## 🔍 문제 원인
- `/api/v1/sentiment/analyze/realtime`는 **POST** 메서드 엔드포인트입니다.
- GET 요청으로는 404 Not Found가 발생합니다.

## ✅ 올바른 API 사용법

### 1. 서비스 상태 확인 (GET)
```bash
# Health Check
curl -s http://109.123.239.30:8000/health | jq

# API Documentation 확인
curl -s http://109.123.239.30:8000/docs
```

### 2. 실시간 감정 분석 (POST) - 올바른 사용법
```bash
# 기본 realtime 분석
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin surges to new all-time high as institutional adoption grows",
    "symbol": "BTC"
  }' | jq

# 더 짧은 텍스트로 테스트
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin price is rising rapidly",
    "symbol": "BTC"
  }' | jq

# 부정적 감정 테스트
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze/realtime \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin crashes amid regulatory concerns",
    "symbol": "BTC"
  }' | jq
```

### 3. 일반 감정 분석 (POST)
```bash
# 기본 분석
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin adoption by major corporations signals bullish market sentiment",
    "symbol": "BTC",
    "include_detailed": false
  }' | jq

# 상세 분석
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ethereum network upgrades drive positive investor confidence",
    "symbol": "ETH",
    "include_detailed": true
  }' | jq
```

### 4. 배치 분석 (POST)
```bash
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Bitcoin reaches new highs",
      "Ethereum network congestion causes delays",
      "DeFi protocols show strong growth"
    ],
    "symbol": "CRYPTO"
  }' | jq
```

### 5. 모델 정보 확인 (GET)
```bash
curl -s http://109.123.239.30:8000/api/v1/sentiment/model/info | jq
```

### 6. Sentiment Health Check (GET)  
```bash
curl -s http://109.123.239.30:8000/api/v1/sentiment/health | jq
```

## 📊 예상 응답 형식

### Realtime Analysis Response
```json
{
  "sentiment_score": 0.75,
  "label": "POSITIVE",
  "confidence": 0.85,
  "processing_time": 0.003,
  "keywords": ["surges", "high", "institutional", "adoption"],
  "scenario_tag": "BULLISH",
  "metadata": {
    "symbol": "BTC",
    "model": "keyword_realtime",
    "category_scores": {
      "bullish_keywords": 0.8,
      "bearish_keywords": 0.1
    }
  }
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-07-30T00:15:00Z",
  "version": "1.0.0",
  "uptime": 3600.5,
  "dependencies": {
    "redis": "connected",
    "model": "loaded"
  }
}
```

## 🛠 디버깅 명령어

### 서비스 로그 확인
```bash
# Docker 컨테이너 로그
docker logs aurora-sentiment-service --tail 50 -f

# 또는 systemd 서비스 로그
journalctl -u aurora-sentiment -f --lines 50
```

### 네트워크 연결 확인
```bash
# 포트 8000 확인
netstat -tlnp | grep :8000

# 서비스 프로세스 확인
ps aux | grep sentiment
```

## 🔧 문제 해결

### 1. 404 Not Found 해결
- **원인**: GET 요청을 POST 엔드포인트에 전송
- **해결**: POST 메서드와 JSON 페이로드 사용

### 2. 500 Internal Server Error 시
```bash
# 서비스 상태 확인
curl -s http://109.123.239.30:8000/health

# 로그 확인
docker logs aurora-sentiment-service --tail 20
```

### 3. Connection Refused 시
```bash
# 서비스 실행 상태 확인
docker ps | grep sentiment

# 포트 바인딩 확인
docker port aurora-sentiment-service
```

## 🎯 성능 테스트

### Realtime 엔드포인트 성능 확인
```bash
# 응답 시간 측정
time curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze/realtime \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin is rising", "symbol": "BTC"}' \
  -w "Response Time: %{time_total}s\n"
```

---

**✅ 이제 올바른 POST 방식으로 API를 호출하면 정상 작동할 것입니다!**