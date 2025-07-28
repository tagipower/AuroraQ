#!/bin/bash
# sentiment-service/scripts/start_service.sh
# Sentiment Service 시작 및 헬스체크 스크립트

set -e

# 색깔 출력 함수
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 설정 변수
SERVICE_NAME="sentiment-service"
COMPOSE_FILE="docker-compose.yml"
HEALTH_ENDPOINT="http://localhost:8000/health"
TIMEOUT=60
CHECK_INTERVAL=2

# 현재 디렉토리 확인
if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "docker-compose.yml not found. Please run this script from sentiment-service directory."
    exit 1
fi

log_info "Starting AuroraQ Sentiment Service..."

# 기존 컨테이너 정리
log_info "Cleaning up existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# 이미지 빌드 (선택적)
if [ "$1" = "--build" ] || [ "$1" = "-b" ]; then
    log_info "Building Docker images..."
    docker-compose build --no-cache
fi

# 서비스 시작
log_info "Starting services with docker-compose..."
docker-compose up -d

# 헬스체크 대기
log_info "Waiting for service to be ready..."
elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    if curl -f -s "$HEALTH_ENDPOINT" > /dev/null 2>&1; then
        log_success "Service is healthy!"
        break
    fi
    
    echo -n "."
    sleep $CHECK_INTERVAL
    elapsed=$((elapsed + CHECK_INTERVAL))
done

echo ""

if [ $elapsed -ge $TIMEOUT ]; then
    log_error "Service failed to start within $TIMEOUT seconds"
    log_info "Checking container logs..."
    docker-compose logs --tail=20
    exit 1
fi

# 서비스 상태 확인
log_info "Checking service status..."

# 컨테이너 상태
log_info "Container status:"
docker-compose ps

# 헬스체크 상세 정보
log_info "Health check details:"
health_response=$(curl -s "$HEALTH_ENDPOINT" 2>/dev/null || echo "Failed to get health status")
echo "$health_response" | jq . 2>/dev/null || echo "$health_response"

# 메모리 사용량
log_info "Memory usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" $(docker-compose ps -q)

# API 엔드포인트 테스트
log_info "Testing API endpoints..."

# Swagger UI
if curl -f -s "http://localhost:8000/docs" > /dev/null 2>&1; then
    log_success "Swagger UI available at: http://localhost:8000/docs"
else
    log_warning "Swagger UI not accessible"
fi

# ReDoc
if curl -f -s "http://localhost:8000/redoc" > /dev/null 2>&1; then
    log_success "ReDoc available at: http://localhost:8000/redoc"
else
    log_warning "ReDoc not accessible"
fi

# Prometheus metrics
if curl -f -s "http://localhost:8000/metrics" > /dev/null 2>&1; then
    log_success "Prometheus metrics available at: http://localhost:8000/metrics"
else
    log_warning "Prometheus metrics not accessible"
fi

# 간단한 API 테스트
log_info "Testing sentiment analysis API..."
test_response=$(curl -s -X POST "http://localhost:8000/api/v1/sentiment/analyze" \
    -H "Content-Type: application/json" \
    -d '{"text": "Bitcoin price is rising", "symbol": "BTC"}' 2>/dev/null)

if echo "$test_response" | jq -e '.sentiment_score' > /dev/null 2>&1; then
    sentiment_score=$(echo "$test_response" | jq -r '.sentiment_score')
    log_success "API test successful - Sentiment score: $sentiment_score"
else
    log_warning "API test failed or returned unexpected response"
    echo "Response: $test_response"
fi

log_success "Sentiment service startup complete!"
log_info "Service URLs:"
echo "  • Health Check: $HEALTH_ENDPOINT"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • ReDoc: http://localhost:8000/redoc"
echo "  • Prometheus Metrics: http://localhost:8000/metrics"

log_info "To stop the service, run:"
echo "  docker-compose down"

log_info "To view logs, run:"
echo "  docker-compose logs -f"