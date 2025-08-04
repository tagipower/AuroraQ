#!/bin/bash
# AuroraQ VPS 실전매매 시스템 재시작 스크립트

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

log_info "AuroraQ VPS 실전매매 시스템 재시작 중..."
cd "$PROJECT_DIR"

# Docker Compose 명령어 결정
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# 실전매매 서비스만 재시작 (빠른 재시작)
read -p "전체 시스템을 재시작하시겠습니까? (y: 전체, N: 실전매매만): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "전체 시스템 재시작 중..."
    $DOCKER_COMPOSE restart
else
    log_info "실전매매 서비스만 재시작 중..."
    $DOCKER_COMPOSE restart vps-trading
fi

# 헬스 체크
log_info "서비스 헬스 체크 중..."
sleep 10

if curl -f http://localhost:8004/trading/health > /dev/null 2>&1; then
    log_success "실전매매 시스템 재시작 완료"
else
    log_warning "실전매매 시스템 헬스 체크 실패 - 로그를 확인하세요"
    $DOCKER_COMPOSE logs --tail=20 vps-trading
fi

log_success "시스템 재시작이 완료되었습니다."