#!/bin/bash
# AuroraQ VPS 실전매매 시스템 배포 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로그 함수
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

# 스크립트 시작
log_info "AuroraQ VPS 실전매매 시스템 배포 시작..."

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log_info "프로젝트 디렉토리: $PROJECT_DIR"
cd "$PROJECT_DIR"

# 환경 변수 파일 확인
if [ ! -f ".env" ]; then
    log_error ".env 파일이 없습니다. .env.example을 복사하여 설정하세요."
    exit 1
fi

if [ ! -f ".env.trading" ]; then
    log_error ".env.trading 파일이 없습니다."
    exit 1
fi

# Docker 및 Docker Compose 확인
if ! command -v docker &> /dev/null; then
    log_error "Docker가 설치되지 않았습니다."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose가 설치되지 않았습니다."
    exit 1
fi

# Docker Compose 명령어 결정
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

log_info "사용할 Docker Compose: $DOCKER_COMPOSE"

# 필요한 디렉토리 생성
log_info "필요한 디렉토리 생성 중..."
mkdir -p {data/{redis,postgres,prometheus,grafana},logs/{raw,summary,training,tagged},cache,models}

# 권한 설정
log_info "디렉토리 권한 설정 중..."
chmod -R 755 data logs cache models
chmod +x scripts/*.sh 2>/dev/null || true

# 기존 컨테이너 정리 (선택사항)
read -p "기존 컨테이너를 정리하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "기존 컨테이너 정리 중..."
    $DOCKER_COMPOSE down -v --remove-orphans || true
    docker system prune -f || true
fi

# 이미지 빌드
log_info "Docker 이미지 빌드 중..."
$DOCKER_COMPOSE build --no-cache

# 서비스 시작 (단계별)
log_info "인프라 서비스 시작 중..."
$DOCKER_COMPOSE up -d redis postgres prometheus grafana

# 인프라 서비스 헬스 체크
log_info "인프라 서비스 헬스 체크 중..."
sleep 10

# Redis 헬스 체크
if ! docker exec auroraq-redis redis-cli ping > /dev/null 2>&1; then
    log_error "Redis 서비스가 정상적으로 시작되지 않았습니다."
    exit 1
fi
log_success "Redis 서비스 정상"

# PostgreSQL 헬스 체크
if ! docker exec auroraq-postgres pg_isready -U ${POSTGRES_USER:-onnx_user} > /dev/null 2>&1; then
    log_error "PostgreSQL 서비스가 정상적으로 시작되지 않았습니다."
    exit 1
fi
log_success "PostgreSQL 서비스 정상"

# ONNX 센티먼트 서비스 시작
log_info "ONNX 센티먼트 서비스 시작 중..."
$DOCKER_COMPOSE up -d onnx-sentiment

# ONNX 서비스 헬스 체크 (최대 2분 대기)
log_info "ONNX 서비스 헬스 체크 중... (최대 2분 대기)"
for i in {1..24}; do
    if curl -f http://localhost:8001/onnx/health > /dev/null 2>&1; then
        log_success "ONNX 센티먼트 서비스 정상"
        break
    fi
    if [ $i -eq 24 ]; then
        log_error "ONNX 서비스가 2분 내에 시작되지 않았습니다."
        log_info "서비스 로그 확인:"
        $DOCKER_COMPOSE logs onnx-sentiment
        exit 1
    fi
    sleep 5
done

# 실전매매 시스템 시작
log_info "VPS 실전매매 시스템 시작 중..."
$DOCKER_COMPOSE up -d vps-trading

# 실전매매 시스템 헬스 체크
log_info "실전매매 시스템 헬스 체크 중... (최대 1분 대기)"
for i in {1..12}; do
    if curl -f http://localhost:8004/trading/health > /dev/null 2>&1; then
        log_success "VPS 실전매매 시스템 정상"
        break
    fi
    if [ $i -eq 12 ]; then
        log_error "실전매매 시스템이 1분 내에 시작되지 않았습니다."
        log_info "서비스 로그 확인:"
        $DOCKER_COMPOSE logs vps-trading
        exit 1
    fi
    sleep 5
done

# 서비스 상태 확인
log_info "모든 서비스 상태 확인 중..."
$DOCKER_COMPOSE ps

# 서비스 URL 출력
log_info "=== 서비스 접속 정보 ==="
echo "ONNX 센티먼트 API: http://localhost:8001"
echo "실전매매 API: http://localhost:8004" 
echo "실전매매 WebSocket: ws://localhost:8003"
echo "Grafana 대시보드: http://localhost:3000 (admin/admin123)"
echo "Prometheus: http://localhost:9090"
echo "Redis: localhost:6379"
echo "PostgreSQL: localhost:5432"

# 로그 모니터링 안내
log_info "=== 로그 모니터링 ==="
echo "전체 로그: $DOCKER_COMPOSE logs -f"
echo "실전매매 로그: $DOCKER_COMPOSE logs -f vps-trading"
echo "ONNX 로그: $DOCKER_COMPOSE logs -f onnx-sentiment"

# 통합 로깅 상태 확인
log_info "통합 로깅 시스템 상태 확인..."
if [ -d "logs" ]; then
    log_success "통합 로깅 디렉토리 생성됨"
    ls -la logs/
else
    log_warning "통합 로깅 디렉토리가 생성되지 않았습니다."
fi

# 실전매매 모드 경고
TRADING_MODE=$(grep "^TRADING_MODE=" .env.trading | cut -d'=' -f2)
if [ "$TRADING_MODE" = "live" ]; then
    log_warning "⚠️  실거래 모드가 활성화되어 있습니다!"
    log_warning "⚠️  실제 자금이 사용될 수 있습니다!"
    log_warning "⚠️  충분한 테스트 후 사용하세요!"
else
    log_info "📄 페이퍼 트레이딩 모드로 실행 중입니다."
fi

# 배포 완료
log_success "🎉 AuroraQ VPS 실전매매 시스템 배포 완료!"
log_info "시스템이 정상적으로 시작되었습니다."

# 후속 작업 안내
log_info "=== 후속 작업 ==="
echo "1. Grafana 대시보드 설정"
echo "2. 실전매매 전략 파라미터 조정"
echo "3. 알림 설정 (Slack, Discord 등)"
echo "4. 백테스팅 실행 및 검증"
echo "5. 모니터링 및 로그 확인"

# 중지 스크립트 안내
log_info "시스템 중지: ./scripts/stop_trading_system.sh"
log_info "시스템 재시작: ./scripts/restart_trading_system.sh"

exit 0