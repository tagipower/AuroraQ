# 🚀 Aurora Advanced Sentiment Service v3.0 배포 가이드

고도화된 멀티모달 감정 분석 시스템의 완전한 배포 및 운영 가이드

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [시스템 요구사항](#시스템-요구사항)
3. [설치 및 설정](#설치-및-설정)
4. [고급 피처 구성](#고급-피처-구성)
5. [성능 최적화](#성능-최적화)
6. [모니터링 및 운영](#모니터링-및-운영)
7. [트러블슈팅](#트러블슈팅)
8. [API 참조](#api-참조)

## 🌟 시스템 개요

Aurora Advanced Sentiment Service v3.0은 차세대 AI 기반 감정 분석 플랫폼입니다.

### 핵심 기능

**🧠 멀티모달 감정 분석**
- 텍스트, 가격 행동, 거래량, 소셜 미디어 데이터 통합 분석
- 실시간 키워드 분석 (0.5초 내 응답)
- FinBERT 기반 정확한 배치 처리

**🔮 ML 리파인 예측**
- 앙상블 모델 기반 방향 예측
- 확률적 불확실성 정량화
- 시간 지평별 예측 (1h, 4h, 24h, 1w)

**💥 이벤트 영향도 분석**
- 실시간 이벤트 감지 및 분류
- 시장 반응 지연시간 추정
- 파급효과 및 지속시간 예측

**🚨 고급 이상 탐지**
- 통계적 이상치 감지
- 블랙 스완 확률 계산
- 시장 조작 패턴 식별

**🌐 네트워크 분석**
- 소셜 미디어 바이럴 점수
- 정보 확산 속도 측정
- 군집행동 감지

**📊 실시간 대시보드**
- 9개 패널 (3x3) 고급 레이아웃
- Claude Code 스타일 타이핑 효과
- 실시간 데이터 스트리밍

## 💻 시스템 요구사항

### 하드웨어 요구사항

**추천 사양 (48GB VPS)**
```yaml
CPU: 16+ cores (Intel/AMD 64-bit)
RAM: 48GB+
Storage: 500GB+ NVMe SSD
Network: 1Gbps+
OS: Ubuntu 20.04 LTS / 22.04 LTS
```

**메모리 할당 계획**
```yaml
Core Services: 24GB
  - Redis Cache: 8GB
  - FastAPI Service: 4GB
  - Sentiment Fusion: 4GB
  - Data Collection: 3GB
  - FinBERT Processing: 3GB
  - Dashboard Service: 2GB

Trading Platform: 20GB
  - Live Trading Engine: 8GB
  - Simulation Engine: 6GB
  - Risk Management: 3GB
  - Portfolio Manager: 3GB

System & Buffer: 4GB
  - OS & System: 2GB
  - Emergency Buffer: 2GB
```

### 소프트웨어 요구사항

**Python 환경**
```bash
Python 3.9+
pip 21.0+
virtualenv / conda
```

**데이터베이스**
```bash
Redis 6.0+
PostgreSQL 13+ (선택사항)
```

**시스템 도구**
```bash
Docker 20.10+
Docker Compose 2.0+
nginx 1.18+
supervisor / systemd
htop, iotop (모니터링)
```

## 🛠️ 설치 및 설정

### 1. 환경 준비

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y python3.9 python3.9-venv python3.9-dev
sudo apt install -y redis-server postgresql postgresql-contrib
sudo apt install -y nginx supervisor htop iotop

# Docker 설치 (선택사항)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### 2. Python 환경 설정

```bash
# 프로젝트 디렉토리 생성
sudo mkdir -p /opt/aurora-sentiment
sudo chown $USER:$USER /opt/aurora-sentiment
cd /opt/aurora-sentiment

# 가상환경 생성
python3.9 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements/base.txt
pip install -r sentiment-service/requirements.txt
```

### 3. Redis 구성 (8GB 할당)

```bash
# Redis 설정 파일 편집
sudo nano /etc/redis/redis.conf
```

**Redis 최적화 설정**
```conf
# 메모리 설정
maxmemory 8gb
maxmemory-policy allkeys-lru

# 성능 최적화
save 900 1
save 300 10
save 60 10000

# 네트워킹
bind 127.0.0.1
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300

# 고급 설정
databases 16
rdbcompression yes
rdbchecksum yes
stop-writes-on-bgsave-error yes
```

```bash
# Redis 재시작
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Redis 상태 확인
redis-cli ping
redis-cli info memory
```

### 4. 환경 변수 설정

```bash
# 환경 변수 파일 생성
cp sentiment-service/.env.example sentiment-service/.env
nano sentiment-service/.env
```

**production 환경 설정**
```env
# 기본 설정
APP_NAME="Aurora Advanced Sentiment Service"
APP_VERSION="3.0.0"
DEBUG=false
ENVIRONMENT=production

# 서버 설정
HOST=0.0.0.0
PORT=8080
MAX_WORKERS=4

# Redis 설정
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=300

# 보안 설정
ALLOWED_HOSTS=["localhost", "your-domain.com"]
CORS_ORIGINS=["http://localhost:3000", "https://your-domain.com"]

# 로깅 설정
LOG_LEVEL=INFO
ENABLE_METRICS=true

# 고급 기능 설정
MODEL_WARMUP=true
ML_REFINEMENT_ENABLED=true
ANOMALY_DETECTION_ENABLED=true
NETWORK_ANALYSIS_ENABLED=true

# 성능 설정
FUSION_CACHE_SIZE=10000
PREDICTION_CACHE_TTL=600
FEATURE_CACHE_TTL=300
```

### 5. 서비스 시작

**개발 모드**
```bash
cd sentiment-service
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

**프로덕션 모드**
```bash
cd sentiment-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
```

## 🔧 고급 피처 구성

### 1. 멀티모달 감정 분석 설정

**키워드 사전 커스터마이징**
```python
# models/advanced_keyword_scorer.py 수정
self.custom_keywords = {
    "crypto_specific": {
        "hodl": 0.6, "diamond_hands": 0.8, "paper_hands": -0.7,
        "ath": 0.7, "dip": -0.3, "rekt": -0.9
    },
    "defi_terms": {
        "yield": 0.5, "liquidity": 0.4, "rugpull": -0.9,
        "governance": 0.3, "staking": 0.6
    }
}
```

**감정 가중치 조정**
```python
# processors/advanced_fusion_manager.py 수정
self.fusion_weights = {
    "text_sentiment": 0.35,     # 텍스트 감정 비중
    "price_action": 0.25,       # 가격 행동 비중  
    "volume_analysis": 0.20,    # 거래량 분석 비중
    "social_signals": 0.15,     # 소셜 신호 비중
    "technical_indicators": 0.05 # 기술적 지표 비중
}
```

### 2. ML 예측 엔진 구성

**앙상블 모델 설정**
```python
self.ml_models = {
    "gradient_boosting": {
        "enabled": True,
        "weight": 0.4,
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6
        }
    },
    "neural_network": {
        "enabled": True,
        "weight": 0.3,
        "architecture": "transformer",
        "layers": [128, 64, 32]
    },
    "random_forest": {
        "enabled": True,
        "weight": 0.3,
        "n_estimators": 50
    }
}
```

**예측 신뢰도 임계값**
```python
self.confidence_thresholds = {
    "very_high": 0.95,  # 95% 이상
    "high": 0.85,       # 85-95%
    "medium": 0.70,     # 70-85%
    "low": 0.50         # 50-70%
}
```

### 3. 이상 탐지 시스템 구성

**이상 탐지 임계값**
```python
self.anomaly_thresholds = {
    "price_anomaly": 3.0,      # Z-score 3σ
    "volume_anomaly": 2.5,     # Z-score 2.5σ
    "sentiment_anomaly": 2.0,  # Z-score 2σ
    "correlation_anomaly": 0.8, # 상관관계 임계값
    "black_swan_threshold": 0.05 # 5% 확률
}
```

**이상 유형별 대응 설정**
```python
self.anomaly_responses = {
    "critical": "immediate_alert",
    "high": "priority_investigation", 
    "medium": "enhanced_monitoring",
    "low": "standard_logging"
}
```

### 4. 대시보드 커스터마이징

**패널 업데이트 주기**
```python
self.update_intervals = {
    "sentiment_fusion": 5,      # 5초
    "ml_predictions": 10,       # 10초
    "event_impact": 15,         # 15초
    "strategy_performance": 30, # 30초
    "anomaly_detection": 5,     # 5초
    "network_analysis": 20,     # 20초
    "market_pulse": 10,         # 10초
    "system_intelligence": 30,  # 30초
    "live_data_feed": 3         # 3초
}
```

**시각적 효과 설정**
```python
self.visual_effects = {
    "typing_speed": 0.008,      # 타이핑 속도
    "color_transitions": True,   # 색상 전환
    "rainbow_headers": True,     # 무지개 헤더
    "neon_footers": True,       # 네온 푸터
    "pulse_alerts": True        # 펄스 알림
}
```

## ⚡ 성능 최적화

### 1. Redis 최적화

**메모리 최적화**
```bash
# Redis 메모리 분석
redis-cli --bigkeys
redis-cli --memkeys

# 메모리 사용량 모니터링
redis-cli info memory | grep used_memory_human
```

**다층 캐싱 전략**
```python
# Hot Data (Sub-second access)
hot_cache_config = {
    "sentiment_scores": {"ttl": 30, "size": "1GB"},
    "realtime_signals": {"ttl": 15, "size": "512MB"}
}

# Warm Data (Second-level access)  
warm_cache_config = {
    "news_analysis": {"ttl": 300, "size": "2GB"},
    "event_detection": {"ttl": 180, "size": "1GB"}
}

# Cold Data (Background access)
cold_cache_config = {
    "historical_data": {"ttl": 3600, "size": "4GB"},
    "statistics": {"ttl": 1800, "size": "512MB"}
}
```

### 2. API 성능 최적화

**비동기 처리 최적화**
```python
# 병렬 처리 설정
async def optimize_batch_processing():
    semaphore = asyncio.Semaphore(10)  # 동시 처리 제한
    
    async with semaphore:
        tasks = [
            process_sentiment_analysis(),
            process_ml_prediction(),
            process_event_detection()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

**응답 압축 및 최적화**
```python
# FastAPI 최적화 설정
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
    compresslevel=6
)

# JSON 응답 최적화
class OptimizedJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(',', ':')
        ).encode('utf-8')
```

### 3. 시스템 리소스 최적화

**CPU 최적화**
```bash
# CPU 코어별 프로세스 할당
taskset -c 0-5 python trading_engine.py      # Trading: 6 cores
taskset -c 6-9 python sentiment_service.py   # Sentiment: 4 cores  
taskset -c 10-12 python data_collector.py    # Data: 3 cores
taskset -c 13-14 python dashboard.py         # Dashboard: 2 cores
```

**메모리 최적화**
```python
# 메모리 모니터링
import psutil
import gc

def optimize_memory():
    # 가비지 컬렉션
    gc.collect()
    
    # 메모리 사용량 체크
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        # 캐시 정리
        clear_old_cache()
        
    return memory.percent
```

### 4. 데이터베이스 최적화

**Redis 연결 풀링**
```python
import redis.asyncio as redis

# 연결 풀 설정
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=50,
    retry_on_timeout=True,
    socket_keepalive=True,
    socket_keepalive_options={}
)

redis_client = redis.Redis(connection_pool=redis_pool)
```

## 📊 모니터링 및 운영

### 1. 시스템 모니터링

**핵심 메트릭**
```yaml
Performance Metrics:
  - API Response Time: <200ms (avg), <500ms (95th percentile)
  - Throughput: >100 requests/sec
  - Memory Usage: <85% of allocated
  - CPU Usage: <80% average
  - Cache Hit Rate: >90%

Business Metrics:
  - Prediction Accuracy: >75%
  - Sentiment Analysis Quality: >0.85
  - Event Detection Rate: >90%
  - Anomaly Detection Precision: >80%
```

**모니터링 스크립트**
```bash
#!/bin/bash
# monitor_aurora.sh

# 시스템 리소스 체크
echo "=== System Resources ==="
echo "Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}')"

# 서비스 상태 체크
echo "=== Service Status ==="
curl -s http://localhost:8080/health | jq '.status'
curl -s http://localhost:8080/api/v1/fusion/health/advanced | jq '.status'

# Redis 상태 체크
echo "=== Redis Status ==="
redis-cli ping
redis-cli info memory | grep used_memory_human
```

### 2. 로그 관리

**로그 레벨 구성**
```python
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/aurora/sentiment-service.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'detailed'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}
```

**로그 로테이션 설정**
```bash
# /etc/logrotate.d/aurora-sentiment
/var/log/aurora/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 aurora aurora
    postrotate
        systemctl reload aurora-sentiment
    endscript
}
```

### 3. 백업 및 복구

**자동 백업 스크립트**
```bash
#!/bin/bash
# backup_aurora.sh

BACKUP_DIR="/backup/aurora/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Redis 백업
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/

# 설정 파일 백업
cp -r /opt/aurora-sentiment/sentiment-service/.env $BACKUP_DIR/
cp -r /opt/aurora-sentiment/sentiment-service/config/ $BACKUP_DIR/

# 로그 백업 (최근 7일)
find /var/log/aurora/ -name "*.log" -mtime -7 -exec cp {} $BACKUP_DIR/ \;

# 압축
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### 4. 알림 시스템

**Webhook 알림 설정**
```python
import aiohttp
import json

async def send_alert(level: str, message: str, details: dict = None):
    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    
    payload = {
        "text": f"🚨 Aurora Alert - {level.upper()}",
        "attachments": [
            {
                "color": "danger" if level == "critical" else "warning",
                "fields": [
                    {"title": "Message", "value": message, "short": False},
                    {"title": "Timestamp", "value": datetime.now().isoformat(), "short": True},
                    {"title": "Service", "value": "Aurora Sentiment v3.0", "short": True}
                ]
            }
        ]
    }
    
    if details:
        payload["attachments"][0]["fields"].append(
            {"title": "Details", "value": json.dumps(details, indent=2), "short": False}
        )
    
    async with aiohttp.ClientSession() as session:
        await session.post(webhook_url, json=payload)
```

## 🔧 트러블슈팅

### 1. 일반적인 문제들

**메모리 부족 문제**
```bash
# 메모리 사용량 확인
free -h
ps aux --sort=-%mem | head

# 해결 방법
sudo systemctl restart aurora-sentiment
sudo systemctl restart redis-server

# 스왑 추가 (임시)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Redis 연결 문제**
```bash
# Redis 상태 확인
sudo systemctl status redis-server
redis-cli ping

# 로그 확인
sudo tail -f /var/log/redis/redis-server.log

# 연결 수 확인
redis-cli info clients
```

**API 응답 지연**
```bash
# 프로세스 모니터링
sudo htop

# 네트워크 확인
sudo netstat -tulpn | grep :8080

# 로그 확인
sudo tail -f /var/log/aurora/sentiment-service.log
```

### 2. 성능 문제 해결

**높은 CPU 사용률**
```python
# CPU 사용률 모니터링
import psutil

def monitor_cpu():
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        if proc.info['cpu_percent'] > 80:
            print(f"High CPU: {proc.info}")

# 해결 방법
# 1. 워커 프로세스 수 조정
# 2. 비동기 처리 최적화
# 3. 캐싱 전략 개선
```

**메모리 누수**
```python
# 메모리 누수 탐지
import tracemalloc

tracemalloc.start()

# 코드 실행 후
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

### 3. 데이터 품질 문제

**예측 정확도 저하**
```python
# 정확도 모니터링
def monitor_prediction_accuracy():
    recent_predictions = get_recent_predictions(hours=24)
    accuracy = calculate_accuracy(recent_predictions)
    
    if accuracy < 0.7:  # 70% 임계값
        logger.warning(f"Prediction accuracy dropped: {accuracy:.2f}")
        # 모델 재학습 트리거
        trigger_model_retraining()
```

**감정 분석 이상**
```python
# 감정 점수 분포 확인
def check_sentiment_distribution():
    recent_scores = get_recent_sentiment_scores(hours=1)
    
    # 극단값 확인
    extreme_count = sum(1 for score in recent_scores if abs(score) > 0.9)
    extreme_ratio = extreme_count / len(recent_scores)
    
    if extreme_ratio > 0.3:  # 30% 이상이 극단값
        logger.warning("High extreme sentiment ratio detected")
```

## 📖 API 참조

### 1. 고급 융합 분석 API

**엔드포인트**: `POST /api/v1/fusion/advanced/{symbol}`

**요청 예시**
```bash
curl -X POST "http://localhost:8080/api/v1/fusion/advanced/BTCUSDT" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bitcoin surges amid institutional adoption",
    "include_market_data": true,
    "include_social_data": true
  }'
```

**응답 예시**
```json
{
  "symbol": "BTCUSDT",
  "fusion_score": 0.742,
  "market_outlook": "strong_bullish",
  "overall_confidence": 0.856,
  "ml_prediction": {
    "direction": "bullish",
    "probability": 0.834,
    "volatility_forecast": 0.234,
    "confidence_level": "high"
  },
  "event_impact": {
    "impact_score": 0.678,
    "lag_estimate": 45.2,
    "duration_estimate": 18.5
  },
  "anomaly_detection": {
    "anomaly_flag": false,
    "anomaly_score": 0.123,
    "severity": "low"
  }
}
```

### 2. 피처 추출 API

**엔드포인트**: `GET /api/v1/fusion/features/{symbol}`

**요청 예시**
```bash
curl "http://localhost:8080/api/v1/fusion/features/BTCUSDT?feature_types=multimodal,temporal,risk&format=json"
```

### 3. AI 인사이트 API

**엔드포인트**: `GET /api/v1/fusion/insights/{symbol}`

**요청 예시**
```bash
curl "http://localhost:8080/api/v1/fusion/insights/BTCUSDT?insight_types=pattern,risk&confidence_threshold=0.7"
```

### 4. 전략 성과 API

**엔드포인트**: `GET /api/v1/fusion/performance/strategy`

**요청 예시**
```bash
curl "http://localhost:8080/api/v1/fusion/performance/strategy?strategy_name=AuroraQ_Advanced&time_period=30d"
```

### 5. 헬스체크 API

**엔드포인트**: `GET /api/v1/fusion/health/advanced`

**응답 예시**
```json
{
  "status": "healthy",
  "components": {
    "advanced_keyword_scorer": {"status": "healthy", "accuracy": 0.892},
    "ml_prediction_engine": {"status": "healthy", "ensemble_size": 3},
    "anomaly_detector": {"status": "healthy", "detection_rate": 0.045}
  },
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "cache_hit_rate": 0.934
  }
}
```

## 🚀 대시보드 실행

### 터미널 대시보드 시작

```bash
cd sentiment-service/dashboard
python advanced_aurora_dashboard.py
```

**실행 화면**
```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                     🌟 AURORA ADVANCED AI DASHBOARD v3.0 🌟                                                                   ║
║  🕒 2024-01-15 14:30:25 | ⏱️ Uptime: 2:15:30 | 🔄 Updates: 1,234                                                                            ║
║  📊 Success Rate: 98.5% | 🚀 AI Engine: ACTIVE | 🌐 Connection: CONNECTED                                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────┬─────────────────┬─────────────────┐
│  🧠 AI 감정 융합  │    🔮 ML 예측    │  💥 이벤트 영향도  │
│ 📊 융합점수: +0.74│ 🚀 방향: BULLISH │ 💥 종합영향도: 0.82│
│ 📝 텍스트: +0.68 │ 🎯 확률: 83.4%   │ 📈 감정변화: +0.15 │
│ 📈 가격행동: +0.71│ 📊 변동성: 23.4% │ ━━━━━━━━━━━━━━━━━━ │
│ 📊 거래량: +0.45 │ 💪 신뢰도: HIGH  │ 🏛️ regulat: 0.85 │
└─────────────────┼─────────────────┼─────────────────┤
│   📈 전략 성과    │    🚨 이상 탐지   │   🌐 네트워크 분석  │
│ 💰 ROI: +12.5%  │ ✅ 정상 상태     │ 🚀 소셜감정: +0.34 │
│ 📊 샤프: 1.82   │ 🔍 지속 모니터링 중│ 🐦 트위터: +0.42  │
│ 📉 MDD: 8.3%    │ 📊 이상 점수: 0.0│ 📱 레딧: +0.28    │
└─────────────────┼─────────────────┼─────────────────┤
│   💓 시장 펄스    │  🤖 시스템 AI    │  📡 실시간 피드   │
│ 🐂 시장국면: bull │ 🎯 예측정확도: 82%│ 📰 뉴스: 5건     │
│ ₿ BTC: $43,250  │ 🤝 앙상블합의: 89%│ 🎯 신호: 2개     │
│ 😱 공포탐욕: 72  │ 🧠 인사이트: 3개  │ ━━━━━━━━━━━━━━━━━━ │
└─────────────────┴─────────────────┴─────────────────┘
```

## 📚 추가 리소스

### 문서 링크
- [API 전체 문서](./api-docs.html)
- [모델 아키텍처 가이드](./model-architecture.md)
- [성능 튜닝 가이드](./performance-tuning.md)
- [보안 가이드](./security-guide.md)

### 커뮤니티 지원
- GitHub Issues: 버그 리포트 및 기능 요청
- Discord: 실시간 커뮤니티 지원
- Documentation: 최신 문서 및 튜토리얼

---

**Aurora Advanced Sentiment Service v3.0** - 차세대 AI 기반 감정 분석 플랫폼 🚀