# 🏗️ AuroraQ Architecture V3.0

**Advanced AI Trading System with Microservices Architecture**

최신 업데이트: 2025년 7월 28일
버전: 3.0 (FinBERT + Microservices + Docker)

---

## 📋 목차

1. [아키텍처 개요](#-아키텍처-개요)
2. [시스템 구성요소](#-시스템-구성요소)
3. [마이크로서비스 아키텍처](#-마이크로서비스-아키텍처)
4. [감정분석 서비스 독립화](#-감정분석-서비스-독립화)
5. [데이터 플로우](#-데이터-플로우)
6. [API 설계](#-api-설계)
7. [배포 및 확장성](#-배포-및-확장성)
8. [성능 최적화](#-성능-최적화)

---

## 🎯 아키텍처 개요

### **V3.0 주요 변화점**
- **마이크로서비스 아키텍처**: 모노리식 → 서비스 지향 아키텍처
- **감정분석 서비스 독립화**: FinBERT 기반 전용 서비스
- **API 기반 통신**: REST API + gRPC 하이브리드
- **컨테이너화**: Docker + Kubernetes 지원
- **리소스 효율화**: GPU/CPU 자원 최적화

### **설계 원칙**
- **분리와 독립성**: 각 서비스는 독립적으로 배포/확장 가능
- **성능 최적화**: 모델 로딩과 캐싱을 중앙집중화
- **확장성**: 수평적 확장 지원
- **안정성**: 서비스별 장애 격리

---

## 🏢 시스템 구성요소

### **Core AI Agents**
```
┌─────────────────┐    ┌─────────────────┐
│    AuroraQ      │    │     MacroQ      │
│  (Crypto Bot)   │    │  (Stock Bot)    │
│                 │    │                 │
│ • PPO RL        │    │ • PPO RL        │
│ • Risk Mgmt     │    │ • Macro Anal    │
│ • Execution     │    │ • Portfolio     │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   API Gateway         │
         │ (FastAPI + Load Bal)  │
         └───────────────────────┘
```

### **Microservices Layer**
```
┌──────────────────────────────────────────────────────────┐
│                 Microservices Ecosystem                 │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│  Sentiment  │   Market    │   Risk      │   Execution     │
│  Service    │   Data      │   Service   │   Service       │
│             │   Service   │             │                 │
│ • FinBERT   │ • Binance   │ • VaR Calc  │ • Order Mgmt    │
│ • Fusion    │ • Feedly    │ • Position  │ • Portfolio     │
│ • Router    │ • Cache     │ • Limits    │ • Reporting     │
│ • History   │ • Redis     │ • Alert     │ • Monitoring    │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

### **Infrastructure Layer**
```
┌──────────────────────────────────────────────────────────┐
│                Infrastructure & DevOps                  │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│   Docker    │ Kubernetes  │   Monitoring│     Storage     │
│ Containers  │   Cluster   │   Stack     │    Systems      │
│             │             │             │                 │
│ • Services  │ • Auto      │ • Prometheus│ • PostgreSQL    │
│ • Images    │   Scaling   │ • Grafana   │ • Redis         │
│ • Networks  │ • Load Bal  │ • Logging   │ • MinIO         │
│ • Volumes   │ • Service   │ • Alerts    │ • Backup        │
│             │   Mesh      │             │                 │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

---

## 🔬 마이크로서비스 아키텍처

### **Service Architecture Overview**
```mermaid
graph TB
    A[AuroraQ Agent] --> G[API Gateway]
    B[MacroQ Agent] --> G
    
    G --> S1[Sentiment Service]
    G --> S2[Market Data Service]
    G --> S3[Risk Service]
    G --> S4[Execution Service]
    
    S1 --> D1[(Redis Cache)]
    S1 --> D2[(PostgreSQL)]
    S2 --> D1
    S2 --> E1[Binance API]
    S2 --> E2[Feedly API]
    
    S3 --> D2
    S4 --> D2
    S4 --> E1
```

### **Service Responsibilities**

#### **1. Sentiment Service** 🧠
```yaml
responsibilities:
  - FinBERT 기반 금융 감정 분석
  - 다중 소스 감정 융합
  - Live/Backtest 모드 라우팅
  - 감정 히스토리 관리
  
tech_stack:
  - FastAPI + uvicorn
  - FinBERT (ProsusAI/finbert)
  - PyTorch + transformers
  - Redis (캐싱)
  - PostgreSQL (히스토리)
  
resources:
  - CPU: 4 cores
  - Memory: 8GB
  - GPU: Optional (성능 향상)
  
endpoints:
  - POST /analyze/text
  - POST /analyze/batch
  - GET /sentiment/{asset}
  - POST /fusion/multi-source
  - GET /health
```

#### **2. Market Data Service** 📊
```yaml
responsibilities:
  - 실시간 시장 데이터 수집
  - 뉴스 데이터 수집 (Feedly)
  - 데이터 정규화 및 캐싱
  - 히스토리 데이터 관리
  
tech_stack:
  - FastAPI + asyncio
  - aiohttp (API 호출)
  - pandas + numpy
  - Redis (실시간 캐싱)
  - TimescaleDB (시계열 데이터)
  
endpoints:
  - GET /market/{symbol}/ohlcv
  - GET /news/{asset}/latest
  - GET /market/multiple
  - WebSocket /ws/market/{symbol}
```

#### **3. Risk Service** ⚠️
```yaml
responsibilities:
  - 리스크 메트릭 계산
  - 포지션 크기 결정
  - 손실 한도 모니터링
  - 리스크 알림
  
endpoints:
  - POST /risk/calculate
  - GET /risk/limits/{agent}
  - POST /risk/validate-order
  - GET /risk/metrics
```

#### **4. Execution Service** 🎯
```yaml
responsibilities:
  - 주문 실행 및 관리
  - 포트폴리오 추적
  - 성과 분석
  - 보고서 생성
  
endpoints:
  - POST /orders/place
  - GET /portfolio/{agent}
  - GET /performance/metrics
  - GET /orders/history
```

---

## 🧠 감정분석 서비스 독립화

### **Architecture Design**

#### **Service Structure**
```
sentiment-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 애플리케이션
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── analyze.py      # 감정 분석 엔드포인트
│   │   │   ├── fusion.py       # 융합 엔드포인트
│   │   │   ├── history.py      # 히스토리 엔드포인트
│   │   │   └── health.py       # 헬스체크
│   │   └── middleware/
│   │       ├── auth.py         # 인증 미들웨어
│   │       ├── rate_limit.py   # 속도 제한
│   │       └── logging.py      # 로깅
│   ├── core/
│   │   ├── __init__.py
│   │   ├── sentiment/
│   │   │   ├── finbert_analyzer.py
│   │   │   ├── fusion_manager.py
│   │   │   ├── sentiment_router.py
│   │   │   └── history_loader.py
│   │   ├── cache/
│   │   │   ├── redis_client.py
│   │   │   └── memory_cache.py
│   │   └── config/
│   │       ├── settings.py
│   │       └── logging.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py         # Pydantic 요청 모델
│   │   ├── responses.py        # Pydantic 응답 모델
│   │   └── database.py         # DB 모델
│   └── utils/
│       ├── __init__.py
│       ├── model_loader.py     # FinBERT 모델 로더
│       └── validators.py       # 입력 검증
├── tests/
│   ├── test_api.py
│   ├── test_sentiment.py
│   └── test_integration.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── deploy/
    ├── k8s/                    # Kubernetes 매니페스트
    ├── helm/                   # Helm 차트
    └── scripts/
```

#### **API Schema Design**
```python
# Pydantic Models
class SentimentAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    asset: Optional[str] = "CRYPTO"
    mode: Literal["live", "backtest"] = "live"
    timestamp: Optional[datetime] = None

class SentimentAnalyzeResponse(BaseModel):
    sentiment_score: float = Field(..., ge=0.0, le=1.0)
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    keywords: List[str] = []
    scenario_tag: str
    processing_time: float
    model_version: str

class BatchAnalyzeRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    asset: Optional[str] = "CRYPTO"
    mode: Literal["live", "backtest"] = "live"

class FusionRequest(BaseModel):
    sentiment_scores: Dict[str, float] = Field(..., min_items=1)
    symbol: str = "BTCUSDT"
    timestamp: Optional[datetime] = None

class FusionResponse(BaseModel):
    fused_score: float = Field(..., ge=0.0, le=1.0)
    weights_used: Dict[str, float]
    confidence: float
    trend: str
    metadata: Dict[str, Any] = {}
```

### **Docker Configuration**

#### **Dockerfile**
```dockerfile
# sentiment-service/docker/Dockerfile
FROM python:3.11-slim as base

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# FinBERT 모델 사전 다운로드 (선택적)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

# 애플리케이션 코드 복사
COPY app/ ./app/

# 비루트 유저 생성
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 포트 노출
EXPOSE 8001

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 애플리케이션 시작
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

#### **Docker Compose**
```yaml
# sentiment-service/docker/docker-compose.yml
version: '3.8'

services:
  sentiment-service:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8001:8001"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:pass@postgres:5432/sentiment
      - LOG_LEVEL=INFO
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - model_cache:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: sentiment
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  model_cache:
  redis_data:
  postgres_data:
```

---

## 🌊 데이터 플로우

### **Sentiment Analysis Flow**
```mermaid
sequenceDiagram
    participant A as AuroraQ
    participant G as API Gateway
    participant S as Sentiment Service
    participant R as Redis Cache
    participant F as FinBERT
    participant P as PostgreSQL

    A->>G: POST /sentiment/analyze
    G->>S: Forward request
    
    S->>R: Check cache
    alt Cache Hit
        R->>S: Return cached result
    else Cache Miss
        S->>F: Analyze text
        F->>S: Return sentiment
        S->>R: Cache result
        S->>P: Store history
    end
    
    S->>G: Return response
    G->>A: Forward response
```

### **Multi-Source Fusion Flow**
```mermaid
sequenceDiagram
    participant A as AuroraQ
    participant S as Sentiment Service
    participant N as News Collector
    participant T as Social Tracker
    participant F as Fusion Manager

    A->>S: Request BTC sentiment
    
    par Collect from multiple sources
        S->>N: Get news sentiment
        S->>T: Get social sentiment
    end
    
    S->>F: Fuse sentiment scores
    F->>F: Apply outlier detection
    F->>F: Calculate weighted average
    F->>S: Return fused score
    
    S->>A: Return comprehensive sentiment
```

---

## 📡 API 설계

### **REST API Endpoints**

#### **감정 분석 API**
```yaml
POST /api/v1/sentiment/analyze:
  summary: "단일 텍스트 감정 분석"
  parameters:
    - text: string (required)
    - asset: string (optional, default: "CRYPTO")
    - mode: string (optional, "live" | "backtest")
  responses:
    200:
      sentiment_score: float [0.0-1.0]
      label: string
      confidence: float
      keywords: array[string]
      scenario_tag: string
      processing_time: float

POST /api/v1/sentiment/batch:
  summary: "배치 텍스트 감정 분석"
  parameters:
    - texts: array[string] (max 100)
    - asset: string (optional)
  responses:
    200:
      results: array[SentimentResult]
      total_processed: integer
      avg_processing_time: float

GET /api/v1/sentiment/{asset}/realtime:
  summary: "실시간 자산 감정 조회"
  parameters:
    - asset: string (path parameter)
    - hours_back: integer (optional, default: 6)
  responses:
    200:
      sentiment_score: float
      confidence: float
      article_count: integer
      trend: string
      sources: array[string]

POST /api/v1/sentiment/fusion:
  summary: "다중 소스 감정 융합"
  parameters:
    - sentiment_scores: object
    - symbol: string
    - timestamp: datetime (optional)
  responses:
    200:
      fused_score: float
      weights_used: object
      confidence: float
      trend: string
```

#### **gRPC 인터페이스** (고성능 요구시)
```protobuf
syntax = "proto3";

service SentimentService {
  rpc AnalyzeText(AnalyzeRequest) returns (AnalyzeResponse);
  rpc AnalyzeBatch(BatchRequest) returns (BatchResponse);
  rpc GetRealtimeSentiment(RealtimeRequest) returns (RealtimeResponse);
  rpc FuseSentiments(FusionRequest) returns (FusionResponse);
}

message AnalyzeRequest {
  string text = 1;
  string asset = 2;
  string mode = 3;
  google.protobuf.Timestamp timestamp = 4;
}

message AnalyzeResponse {
  float sentiment_score = 1;
  string label = 2;
  float confidence = 3;
  repeated string keywords = 4;
  string scenario_tag = 5;
  float processing_time = 6;
}
```

### **Client Integration**

#### **AuroraQ Integration**
```python
# AuroraQ/services/sentiment_client.py
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

class SentimentClient:
    def __init__(self, base_url: str = "http://sentiment-service:8001"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_sentiment(
        self, 
        text: str, 
        asset: str = "BTC",
        mode: str = "live"
    ) -> Dict:
        """단일 텍스트 감정 분석"""
        async with self.session.post(
            f"{self.base_url}/api/v1/sentiment/analyze",
            json={
                "text": text,
                "asset": asset,
                "mode": mode
            }
        ) as response:
            return await response.json()
    
    async def get_realtime_sentiment(
        self, 
        asset: str = "BTC",
        hours_back: int = 6
    ) -> Dict:
        """실시간 감정 분석"""
        async with self.session.get(
            f"{self.base_url}/api/v1/sentiment/{asset}/realtime",
            params={"hours_back": hours_back}
        ) as response:
            return await response.json()
    
    async def batch_analyze(
        self, 
        texts: List[str], 
        asset: str = "BTC"
    ) -> Dict:
        """배치 감정 분석"""
        async with self.session.post(
            f"{self.base_url}/api/v1/sentiment/batch",
            json={
                "texts": texts,
                "asset": asset
            }
        ) as response:
            return await response.json()

# Usage in AuroraQ
class AuroraQAgent:
    def __init__(self):
        self.sentiment_client = SentimentClient()
    
    async def make_decision(self, market_data: Dict) -> Dict:
        async with self.sentiment_client as client:
            # 실시간 BTC 감정 분석
            sentiment = await client.get_realtime_sentiment("BTC")
            
            # 의사결정 로직
            if sentiment['sentiment_score'] > 0.7 and sentiment['confidence'] > 0.8:
                action = "BUY"
            elif sentiment['sentiment_score'] < 0.3 and sentiment['confidence'] > 0.8:
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                "action": action,
                "sentiment_score": sentiment['sentiment_score'],
                "confidence": sentiment['confidence']
            }
```

---

## 🚀 배포 및 확장성

### **Kubernetes Deployment**

#### **Sentiment Service Deployment**
```yaml
# deploy/k8s/sentiment-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-service
  labels:
    app: sentiment-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-service
  template:
    metadata:
      labels:
        app: sentiment-service
    spec:
      containers:
      - name: sentiment-service
        image: auroraq/sentiment-service:v3.0
        ports:
        - containerPort: 8001
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-service
spec:
  selector:
    app: sentiment-service
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8001
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### **Ingress Configuration**
```yaml
# deploy/k8s/sentiment-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  rules:
  - host: sentiment.auroraq.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentiment-service
            port:
              number: 80
```

### **Helm Chart**
```yaml
# deploy/helm/sentiment-service/Chart.yaml
apiVersion: v2
name: sentiment-service
description: AI-powered financial sentiment analysis service
version: 3.0.0
appVersion: "3.0.0"

# deploy/helm/sentiment-service/values.yaml
replicaCount: 3

image:
  repository: auroraq/sentiment-service
  tag: "v3.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8001

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: sentiment.auroraq.local
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 2
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

env:
  REDIS_URL: "redis://redis-service:6379"
  LOG_LEVEL: "INFO"
  MODEL_CACHE_SIZE: "1000"

redis:
  enabled: true
  architecture: standalone
  auth:
    enabled: false

postgresql:
  enabled: true
  auth:
    postgresPassword: "your-secure-password"
  primary:
    persistence:
      enabled: true
      size: 20Gi
```

---

## 🎯 성능 최적화

### **모델 로딩 최적화**
```python
# app/utils/model_loader.py
import os
import logging
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from functools import lru_cache

logger = logging.getLogger(__name__)

class FinBERTModelManager:
    """FinBERT 모델 관리자 - 싱글톤 패턴"""
    
    _instance: Optional['FinBERTModelManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.model = None
            self.tokenizer = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model_cache = {}
            self._initialized = True
    
    async def load_model(self, model_name: str = "ProsusAI/finbert"):
        """모델 로딩 (애플리케이션 시작시 한 번만)"""
        if self.model is not None:
            return
        
        logger.info(f"Loading FinBERT model: {model_name} on {self.device}")
        
        try:
            # 캐시된 모델 경로 확인
            cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # GPU 사용 가능시 모델을 GPU로 이동
            if self.device == "cuda":
                model = model.cuda()
            
            self.model = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info("FinBERT model loaded successfully ✅")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

# 전역 모델 매니저 인스턴스
model_manager = FinBERTModelManager()

@lru_cache(maxsize=1000)
def get_cached_sentiment(text_hash: str, text: str) -> dict:
    """LRU 캐시를 사용한 감정 분석"""
    if model_manager.model is None:
        raise RuntimeError("Model not loaded")
    
    result = model_manager.model(text)[0]
    
    # 결과 정규화
    positive_score = next(r['score'] for r in result if r['label'] == 'positive')
    negative_score = next(r['score'] for r in result if r['label'] == 'negative')
    neutral_score = next(r['score'] for r in result if r['label'] == 'neutral')
    
    # 최고 점수 레이블 결정
    max_score = max(positive_score, negative_score, neutral_score)
    if max_score == positive_score:
        label = "positive"
        normalized_score = 0.5 + (positive_score - 0.5)
    elif max_score == negative_score:
        label = "negative"
        normalized_score = 0.5 - (negative_score - 0.5)
    else:
        label = "neutral"
        normalized_score = 0.5
    
    return {
        "sentiment_score": min(max(normalized_score, 0.0), 1.0),
        "label": label,
        "confidence": max_score,
        "raw_scores": {
            "positive": positive_score,
            "negative": negative_score,
            "neutral": neutral_score
        }
    }
```

### **Redis 캐싱 전략**
```python
# app/core/cache/redis_client.py
import json
import hashlib
from typing import Optional, Any, Dict
import aioredis
from datetime import datetime, timedelta

class RedisCacheManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Redis 연결"""
        self.redis = aioredis.from_url(self.redis_url)
    
    async def close(self):
        """Redis 연결 종료"""
        if self.redis:
            await self.redis.close()
    
    def _get_cache_key(self, prefix: str, data: Any) -> str:
        """캐시 키 생성"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        hash_key = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{hash_key}"
    
    async def get_sentiment_cache(
        self, 
        text: str, 
        asset: str = "CRYPTO"
    ) -> Optional[Dict]:
        """감정 분석 캐시 조회"""
        if not self.redis:
            return None
        
        cache_key = self._get_cache_key("sentiment", {"text": text[:100], "asset": asset})
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set_sentiment_cache(
        self, 
        text: str, 
        asset: str, 
        result: Dict,
        ttl_seconds: int = 3600
    ):
        """감정 분석 결과 캐싱"""
        if not self.redis:
            return
        
        cache_key = self._get_cache_key("sentiment", {"text": text[:100], "asset": asset})
        
        try:
            # 메타데이터 추가
            cache_data = {
                **result,
                "cached_at": datetime.utcnow().isoformat(),
                "cache_key": cache_key
            }
            
            await self.redis.setex(
                cache_key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def get_realtime_sentiment_cache(self, asset: str) -> Optional[Dict]:
        """실시간 감정 분석 캐시 조회"""
        cache_key = f"realtime_sentiment:{asset}"
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                # 10분 이내 데이터만 유효
                cached_time = datetime.fromisoformat(data['timestamp'])
                if datetime.utcnow() - cached_time < timedelta(minutes=10):
                    return data
        except Exception as e:
            logger.error(f"Realtime cache get error: {e}")
        
        return None
    
    async def set_realtime_sentiment_cache(
        self, 
        asset: str, 
        result: Dict,
        ttl_seconds: int = 600  # 10분
    ):
        """실시간 감정 분석 결과 캐싱"""
        cache_key = f"realtime_sentiment:{asset}"
        
        try:
            cache_data = {
                **result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis.setex(
                cache_key, 
                ttl_seconds, 
                json.dumps(cache_data, default=str)
            )
        except Exception as e:
            logger.error(f"Realtime cache set error: {e}")
```

### **배치 처리 최적화**
```python
# app/api/routes/analyze.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

class BatchProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, texts: List[str], asset: str) -> List[Dict]:
        """배치 처리 최적화"""
        # 청크 단위로 분할 (메모리 효율성)
        chunk_size = 16
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # 병렬 처리
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(self._process_chunk(chunk, asset))
            tasks.append(task)
        
        # 모든 청크 처리 완료 대기
        results = await asyncio.gather(*tasks)
        
        # 결과 병합
        all_results = []
        for chunk_results in results:
            all_results.extend(chunk_results)
        
        return all_results
    
    async def _process_chunk(self, texts: List[str], asset: str) -> List[Dict]:
        """청크 단위 처리"""
        loop = asyncio.get_event_loop()
        
        # CPU 집약적 작업을 별도 스레드에서 실행
        results = await loop.run_in_executor(
            self.executor,
            self._analyze_texts_sync,
            texts,
            asset
        )
        
        return results
    
    def _analyze_texts_sync(self, texts: List[str], asset: str) -> List[Dict]:
        """동기 텍스트 분석 (스레드 풀에서 실행)"""
        results = []
        
        for text in texts:
            try:
                # 캐시된 감정 분석 사용
                text_hash = hashlib.md5(text.encode()).hexdigest()
                result = get_cached_sentiment(text_hash, text)
                
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment_score": result["sentiment_score"],
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "asset": asset
                })
                
            except Exception as e:
                logger.error(f"Batch analysis error for text: {e}")
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment_score": 0.5,
                    "label": "neutral",
                    "confidence": 0.0,
                    "asset": asset,
                    "error": str(e)
                })
        
        return results

# 전역 배치 프로세서
batch_processor = BatchProcessor()

@router.post("/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(
    request: BatchAnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """배치 감정 분석"""
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 100 texts allowed per batch"
        )
    
    start_time = time.time()
    
    try:
        # 배치 처리
        results = await batch_processor.process_batch(request.texts, request.asset)
        
        processing_time = time.time() - start_time
        
        # 백그라운드에서 캐싱
        background_tasks.add_task(
            cache_batch_results,
            request.texts,
            request.asset,
            results
        )
        
        return BatchAnalyzeResponse(
            results=results,
            total_processed=len(results),
            avg_processing_time=processing_time / len(results),
            total_processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 📊 모니터링 및 관찰성

### **Prometheus Metrics**
```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# 메트릭 정의
SENTIMENT_REQUESTS = Counter(
    'sentiment_requests_total',
    'Total sentiment analysis requests',
    ['method', 'asset', 'status']
)

SENTIMENT_DURATION = Histogram(
    'sentiment_analysis_duration_seconds',
    'Sentiment analysis duration',
    ['method', 'asset']
)

MODEL_LOADING_TIME = Histogram(
    'model_loading_duration_seconds',
    'Model loading duration'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Cache hits',
    ['cache_type']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Active connections'
)

def track_sentiment_metrics(asset: str = "unknown"):
    """감정 분석 메트릭 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                SENTIMENT_REQUESTS.labels(
                    method=func.__name__,
                    asset=asset,
                    status='success'
                ).inc()
                return result
                
            except Exception as e:
                SENTIMENT_REQUESTS.labels(
                    method=func.__name__,
                    asset=asset,
                    status='error'
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                SENTIMENT_DURATION.labels(
                    method=func.__name__,
                    asset=asset
                ).observe(duration)
        
        return wrapper
    return decorator
```

### **Structured Logging**
```python
# app/core/config/logging.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """구조화된 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 추가 필드가 있으면 포함
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # 예외 정보 포함
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging():
    """로깅 설정"""
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    # 특정 로거 레벨 설정
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

class SentimentLogger:
    """감정 분석 전용 로거"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_analysis(
        self, 
        text: str, 
        result: Dict[str, Any],
        processing_time: float,
        cached: bool = False
    ):
        """감정 분석 로그"""
        self.logger.info(
            "Sentiment analysis completed",
            extra={
                'extra_fields': {
                    'text_length': len(text),
                    'sentiment_score': result.get('sentiment_score'),
                    'label': result.get('label'),
                    'confidence': result.get('confidence'),
                    'processing_time': processing_time,
                    'cached': cached,
                    'operation': 'sentiment_analysis'
                }
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """에러 로그"""
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={
                'extra_fields': {
                    **context,
                    'error_type': type(error).__name__,
                    'operation': 'error'
                }
            },
            exc_info=True
        )
```

---

## 🔒 보안 및 인증

### **API 키 기반 인증**
```python
# app/api/middleware/auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import os

security = HTTPBearer()

class AuthManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
        self.algorithm = "HS256"
        self.api_keys = set(os.getenv("API_KEYS", "").split(","))
    
    def verify_api_key(self, api_key: str) -> bool:
        """API 키 검증"""
        return api_key in self.api_keys
    
    def create_access_token(self, data: dict) -> str:
        """액세스 토큰 생성"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=24)
        to_encode.update({"exp": expire})
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

auth_manager = AuthManager()

async def verify_authentication(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """인증 확인"""
    token = credentials.credentials
    
    # API 키 방식 확인
    if auth_manager.verify_api_key(token):
        return {"type": "api_key", "key": token}
    
    # JWT 토큰 방식 확인
    try:
        payload = auth_manager.verify_token(token)
        return {"type": "jwt", "payload": payload}
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### **Rate Limiting**
```python
# app/api/middleware/rate_limit.py
from fastapi import HTTPException, Request
import time
from collections import defaultdict, deque
from typing import Dict, Deque
import asyncio

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, Deque[float]] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(
        self, 
        identifier: str, 
        max_requests: int = 100, 
        window_seconds: int = 60
    ) -> bool:
        """요청 허용 여부 확인"""
        async with self.lock:
            now = time.time()
            window_start = now - window_seconds
            
            # 윈도우 밖의 요청 제거
            while (self.requests[identifier] and 
                   self.requests[identifier][0] < window_start):
                self.requests[identifier].popleft()
            
            # 현재 요청 수 확인
            if len(self.requests[identifier]) >= max_requests:
                return False
            
            # 현재 요청 추가
            self.requests[identifier].append(now)
            return True

rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting 미들웨어"""
    # 클라이언트 식별 (IP 주소 또는 API 키)
    client_ip = request.client.host
    auth_header = request.headers.get("Authorization")
    
    identifier = f"ip:{client_ip}"
    if auth_header:
        identifier = f"auth:{auth_header[:20]}"  # API 키의 앞 20자리
    
    # Rate limit 확인
    allowed = await rate_limiter.is_allowed(identifier, max_requests=100, window_seconds=60)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    response = await call_next(request)
    return response
```

---

## 🎯 결론

**AuroraQ Architecture V3.0**은 다음과 같은 핵심 이점을 제공합니다:

### **✅ 주요 달성 목표**
1. **마이크로서비스 아키텍처**: 각 서비스 독립적 개발/배포/확장
2. **FinBERT 기반 고급 감정분석**: 금융 도메인 특화 AI 모델
3. **컨테이너화 및 오케스트레이션**: Docker + Kubernetes 완전 지원
4. **API 기반 통신**: REST + gRPC 하이브리드 아키텍처
5. **리소스 효율화**: 모델 로딩 중앙집중화 및 캐싱 최적화

### **🚀 확장성 및 성능**
- **수평적 확장**: HPA를 통한 자동 스케일링
- **로드 밸런싱**: 트래픽 분산 및 고가용성
- **캐싱 전략**: Redis 기반 다층 캐싱
- **배치 처리**: 대용량 텍스트 처리 최적화

### **🔒 보안 및 운영**
- **API 키 + JWT** 이중 인증
- **Rate Limiting** 및 DDoS 방어
- **구조화된 로깅** 및 메트릭
- **헬스체크** 및 모니터링

### **🎨 개발자 경험**
- **간단한 API 인터페이스**: RESTful + 직관적 엔드포인트
- **완전한 Docker 지원**: 로컬 개발부터 프로덕션까지
- **포괄적인 문서화**: API 스키마 + 사용 예제
- **테스트 자동화**: 단위/통합/E2E 테스트

**이제 AuroraQ와 MacroQ는 고성능, 확장 가능한 마이크로서비스 기반 감정분석을 통해 더욱 정확한 AI 거래 결정을 내릴 수 있습니다!** 🎉