#!/bin/bash
# AuroraQ VPS 실전매매 시스템 중지 스크립트

set -e

# 색상 정의
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

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

log_info "AuroraQ VPS 실전매매 시스템 중지 중..."
cd "$PROJECT_DIR"

# Docker Compose 명령어 결정
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# 실전매매 시스템 먼저 중지
log_info "실전매매 시스템 중지 중..."
$DOCKER_COMPOSE stop vps-trading

# 잠시 대기 (포지션 정리 시간)
log_info "포지션 정리 대기 중... (10초)"
sleep 10

# 모든 서비스 중지
log_info "모든 서비스 중지 중..."
$DOCKER_COMPOSE down

# 선택적으로 볼륨도 제거
read -p "데이터 볼륨도 제거하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_warning "데이터 볼륨 제거 중..."
    $DOCKER_COMPOSE down -v
fi

log_success "AuroraQ VPS 실전매매 시스템이 중지되었습니다."