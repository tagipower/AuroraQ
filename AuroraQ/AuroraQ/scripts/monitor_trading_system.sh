#!/bin/bash
# AuroraQ VPS 실전매매 시스템 모니터링 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

log_header() {
    echo -e "${CYAN}=== $1 ===${NC}"
}

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Docker Compose 명령어 결정
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# 헤더 출력
clear
log_header "AuroraQ VPS 실전매매 시스템 모니터링"

# 1. 컨테이너 상태 확인
log_header "컨테이너 상태"
$DOCKER_COMPOSE ps

echo ""

# 2. 서비스 헬스 체크
log_header "서비스 헬스 체크"

# Redis 체크
if docker exec auroraq-redis redis-cli ping > /dev/null 2>&1; then
    log_success "Redis: 정상"
else
    log_error "Redis: 비정상"
fi

# PostgreSQL 체크
if docker exec auroraq-postgres pg_isready -U ${POSTGRES_USER:-onnx_user} > /dev/null 2>&1; then
    log_success "PostgreSQL: 정상"
else
    log_error "PostgreSQL: 비정상"
fi

# ONNX 센티먼트 서비스 체크
if curl -f http://localhost:8001/onnx/health > /dev/null 2>&1; then
    log_success "ONNX 센티먼트: 정상"
else
    log_error "ONNX 센티먼트: 비정상"
fi

# 실전매매 시스템 체크
if curl -f http://localhost:8004/trading/health > /dev/null 2>&1; then
    log_success "VPS 실전매매: 정상"
else
    log_error "VPS 실전매매: 비정상"
fi

echo ""

# 3. 리소스 사용량 확인
log_header "컨테이너 리소스 사용량"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""

# 4. 디스크 사용량 확인
log_header "디스크 사용량"
echo "로그 디렉토리:"
du -sh logs/ 2>/dev/null || echo "로그 디렉토리 없음"

echo "데이터 디렉토리:"
du -sh data/ 2>/dev/null || echo "데이터 디렉토리 없음"

echo "캐시 디렉토리:"
du -sh cache/ 2>/dev/null || echo "캐시 디렉토리 없음"

echo ""

# 5. 최근 로그 확인
log_header "최근 실전매매 로그 (마지막 10줄)"
$DOCKER_COMPOSE logs --tail=10 vps-trading 2>/dev/null || echo "실전매매 컨테이너가 실행 중이지 않습니다."

echo ""

# 6. 네트워크 포트 확인
log_header "네트워크 포트 상태"
echo "포트 8001 (ONNX API): $(netstat -an | grep :8001 | wc -l) 연결"
echo "포트 8003 (Trading WebSocket): $(netstat -an | grep :8003 | wc -l) 연결" 
echo "포트 8004 (Trading API): $(netstat -an | grep :8004 | wc -l) 연결"

echo ""

# 7. 실전매매 통계 (API 호출)
log_header "실전매매 시스템 통계"
TRADING_STATS=$(curl -s http://localhost:8004/api/stats 2>/dev/null || echo "{}")
if [ "$TRADING_STATS" != "{}" ]; then
    echo "$TRADING_STATS" | python3 -m json.tool 2>/dev/null || echo "통계 파싱 실패"
else
    echo "실전매매 API에서 통계를 가져올 수 없습니다."
fi

echo ""

# 8. 통합 로깅 상태
log_header "통합 로깅 상태"
if [ -d "logs" ]; then
    echo "로그 파일 수:"
    find logs/ -name "*.log" -o -name "*.jsonl" -o -name "*.csv" | wc -l
    echo ""
    echo "로그 카테고리별 파일 수:"
    for category in raw summary training tagged; do
        if [ -d "logs/$category" ]; then
            count=$(find "logs/$category" -type f | wc -l)
            echo "  $category: $count 파일"
        fi
    done
else
    echo "로그 디렉토리가 없습니다."
fi

echo ""

# 9. 시스템 메모리 상태
log_header "시스템 메모리 상태"
free -h

echo ""

# 10. 실행 중인 프로세스 (Docker)
log_header "Docker 프로세스"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""

# 모니터링 완료
log_success "모니터링 완료 - $(date)"

# 실시간 모니터링 옵션
echo ""
read -p "실시간 로그를 보시겠습니까? (1: 전체, 2: 실전매매만, N: 종료): " -n 1 -r
echo

case $REPLY in
    1)
        log_info "전체 실시간 로그 시작... (Ctrl+C로 종료)"
        $DOCKER_COMPOSE logs -f
        ;;
    2)
        log_info "실전매매 실시간 로그 시작... (Ctrl+C로 종료)"
        $DOCKER_COMPOSE logs -f vps-trading
        ;;
    *)
        log_info "모니터링을 종료합니다."
        exit 0
        ;;
esac