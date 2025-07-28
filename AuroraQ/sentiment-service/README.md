# AuroraQ Sentiment Service

AI-powered sentiment analysis microservice for financial market analysis using FinBERT and advanced fusion algorithms.

## 🚀 Features

- **FinBERT-based Analysis**: High-accuracy financial sentiment analysis using ProsusAI/finbert
- **Multi-source Fusion**: Intelligent fusion of sentiment scores from multiple sources
- **REST & gRPC APIs**: Dual API support for flexible integration
- **Redis Caching**: High-performance caching with configurable TTL
- **Prometheus Metrics**: Comprehensive monitoring and observability
- **Docker Ready**: Production-ready containerization
- **Async Processing**: High-throughput asynchronous processing
- **Rate Limiting**: Built-in protection against abuse
- **Health Checks**: Kubernetes-ready health monitoring

## 📋 Requirements

- Python 3.11+
- Redis 6.0+
- Docker & Docker Compose (for containerized deployment)
- 2GB RAM minimum (4GB recommended)
- GPU support optional (CPU optimized)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AuroraQ/      │    │   Sentiment     │    │     Redis       │
│   MacroQ        │◄──►│   Service       │◄──►│     Cache       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   FinBERT       │
                       │   Model         │
                       └─────────────────┘
```

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Clone and Setup**:
```bash
cd sentiment-service
cp .env.example .env  # Edit with your configuration
```

2. **Start Services**:
```bash
docker-compose up -d
```

3. **Verify Health**:
```bash
curl http://localhost:8000/health
```

### Manual Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start Redis**:
```bash
redis-server
```

3. **Configure Environment**:
```bash
export REDIS_URL="redis://localhost:6379"
export LOG_LEVEL="INFO"
export FINBERT_MODEL_NAME="ProsusAI/finbert"
```

4. **Start Service**:
```bash
python -m app.main
```

## 📖 API Documentation

### REST API

The service runs on port 8000 with automatic OpenAPI documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### Key Endpoints

**Sentiment Analysis**:
```http
POST /api/v1/sentiment/analyze
Content-Type: application/json

{
  "text": "Bitcoin price surges to new all-time high",
  "symbol": "BTC",
  "include_detailed": true
}
```

**Batch Analysis**:
```http
POST /api/v1/sentiment/analyze/batch
Content-Type: application/json

{
  "texts": [
    "Positive market sentiment drives crypto rally",
    "Regulatory concerns weigh on digital assets"
  ],
  "symbol": "CRYPTO"
}
```

**Sentiment Fusion**:
```http
POST /api/v1/fusion/fuse
Content-Type: application/json

{
  "sentiment_scores": {
    "news": 0.8,
    "social": 0.6,
    "technical": 0.7
  },
  "symbol": "BTCUSDT"
}
```

### gRPC API

gRPC service runs on port 50051. Protocol buffer definitions are in `grpc_service/sentiment.proto`.

#### Example Client (Python):
```python
import grpc
from grpc_service import sentiment_pb2, sentiment_pb2_grpc

# Connect to service
channel = grpc.insecure_channel('localhost:50051')
client = sentiment_pb2_grpc.SentimentAnalyzerStub(channel)

# Analyze sentiment
request = sentiment_pb2.SentimentRequest(
    text="Bitcoin adoption accelerating globally",
    symbol="BTC"
)
response = client.AnalyzeSentiment(request)
print(f"Sentiment: {response.sentiment_score}")
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `LOG_LEVEL` | `INFO` | Logging level |
| `FINBERT_MODEL_NAME` | `ProsusAI/finbert` | FinBERT model name |
| `CACHE_TTL` | `300` | Cache TTL in seconds |
| `MAX_WORKERS` | `4` | Max worker processes |
| `ENABLE_METRICS` | `true` | Enable Prometheus metrics |
| `DEBUG` | `false` | Debug mode |

### Advanced Configuration

Create a `.env` file or modify `config/settings.py` for detailed configuration:

```env
# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30.0
FINBERT_BATCH_SIZE=16
FINBERT_MAX_LENGTH=512

# Fusion Algorithm
FUSION_OUTLIER_THRESHOLD=3.0
FUSION_CONFIDENCE_THRESHOLD=0.6

# Security
ALLOWED_HOSTS=["localhost", "127.0.0.1"]
CORS_ORIGINS=["http://localhost:3000"]

# Monitoring
PROMETHEUS_PORT=8080
HEALTH_CHECK_INTERVAL=30
```

## 🐳 Docker Deployment

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  sentiment-service:
    image: auroraQ/sentiment-service:latest
    ports:
      - "8000:8000"
      - "50051:50051"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
      - MAX_WORKERS=8
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-service
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
        image: auroraQ/sentiment-service:latest
        ports:
        - containerPort: 8000
        - containerPort: 50051
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

## 📊 Monitoring

### Prometheus Metrics

The service exposes metrics on `/metrics`:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram
- `sentiment_analysis_duration` - Sentiment analysis timing
- `fusion_operations_total` - Fusion operations counter
- `cache_hit_rate` - Cache effectiveness
- `model_load_time_seconds` - Model initialization time

### Health Checks

Health endpoint provides detailed status:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600.5,
  "components": {
    "redis": "healthy",
    "finbert_model": "healthy"
  },
  "metrics": {
    "memory_usage_mb": 1024,
    "active_connections": 5,
    "total_requests": 15420
  }
}
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd sentiment-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio black flake8 mypy

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test module
pytest tests/test_sentiment_api.py -v
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

## 🔌 Integration Examples

### AuroraQ Integration

```python
# AuroraQ integration example
import httpx
import asyncio

class SentimentClient:
    def __init__(self, base_url="http://sentiment-service:8000"):
        self.base_url = base_url
    
    async def get_market_sentiment(self, symbol: str, news_texts: list):
        async with httpx.AsyncClient() as client:
            # Batch analyze news
            response = await client.post(
                f"{self.base_url}/api/v1/sentiment/analyze/batch",
                json={
                    "texts": news_texts,
                    "symbol": symbol,
                    "include_detailed": True
                }
            )
            batch_result = response.json()
            
            # Calculate average sentiment
            scores = [r["sentiment_score"] for r in batch_result["results"]]
            avg_sentiment = sum(scores) / len(scores)
            
            # Fuse with other sources (social, technical)
            fusion_response = await client.post(
                f"{self.base_url}/api/v1/fusion/fuse",
                json={
                    "sentiment_scores": {
                        "news": avg_sentiment,
                        "social": 0.6,  # From social media analysis
                        "technical": 0.7  # From technical indicators
                    },
                    "symbol": symbol
                }
            )
            
            return fusion_response.json()

# Usage in trading strategy
async def trading_decision(symbol: str):
    sentiment_client = SentimentClient()
    news_data = await fetch_recent_news(symbol)
    
    sentiment_result = await sentiment_client.get_market_sentiment(
        symbol, [article["content"] for article in news_data]
    )
    
    fused_sentiment = sentiment_result["fused_score"]
    confidence = sentiment_result["confidence"]
    
    if fused_sentiment > 0.7 and confidence > 0.8:
        return "BUY"
    elif fused_sentiment < 0.3 and confidence > 0.8:
        return "SELL"
    else:
        return "HOLD"
```

### MacroQ Integration

```python
# MacroQ integration for macro economic sentiment
class MacroSentimentAnalyzer:
    def __init__(self, sentiment_service_url):
        self.sentiment_service = sentiment_service_url
    
    async def analyze_fed_minutes(self, fed_text: str):
        """Analyze Federal Reserve meeting minutes"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.sentiment_service}/api/v1/sentiment/analyze",
                json={
                    "text": fed_text,
                    "symbol": "SPY",  # S&P 500 as proxy
                    "include_detailed": True
                }
            )
            
            result = response.json()
            
            # Extract policy-relevant keywords
            policy_keywords = [
                kw for kw in result["keywords"] 
                if kw in ["inflation", "rates", "employment", "growth"]
            ]
            
            return {
                "sentiment": result["sentiment_score"],
                "confidence": result["confidence"],
                "policy_keywords": policy_keywords,
                "hawkish_dovish": "hawkish" if result["sentiment_score"] < 0.4 else "dovish"
            }
```

## 🛠️ Troubleshooting

### Common Issues

**1. Model Loading Errors**:
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Verify model download
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('ProsusAI/finbert')"
```

**2. Redis Connection Issues**:
```bash
# Check Redis connectivity
redis-cli ping

# Verify Redis URL
echo $REDIS_URL
```

**3. Memory Issues**:
```bash
# Monitor memory usage
docker stats sentiment-service

# Reduce batch size
export FINBERT_BATCH_SIZE=8
```

**4. Performance Optimization**:
```bash
# Enable model caching
export ENABLE_MODEL_CACHING=true

# Adjust worker count based on CPU cores
export MAX_WORKERS=8
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
python -m app.main
```

## 📈 Performance Benchmarks

Typical performance on standard hardware:

| Operation | Latency (p50) | Latency (p99) | Throughput |
|-----------|---------------|---------------|------------|
| Single Analysis | 45ms | 120ms | 800 req/s |
| Batch Analysis (10 items) | 180ms | 400ms | 200 batch/s |
| Sentiment Fusion | 5ms | 15ms | 5000 req/s |
| Cache Hit | 2ms | 8ms | 15000 req/s |

*Benchmarks on Intel i7-8700K, 16GB RAM, Redis on localhost*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- **AuroraQ**: Main AI trading agent
- **MacroQ**: Macro economic analysis agent
- **SharedCore**: Common data and infrastructure layer

## 📞 Support

For issues and questions:

1. Check existing [GitHub Issues](https://github.com/your-org/sentiment-service/issues)
2. Create a new issue with detailed description
3. Join our community Discord for real-time support

---

**Built with ❤️ for the AuroraQ ecosystem**