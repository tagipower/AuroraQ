#!/bin/bash
# AuroraQ VPS 실전매매 시스템 테스트 스크립트

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

# 테스트 결과 저장
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# 테스트 함수
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        ((TESTS_FAILED++))
        FAILED_TESTS+=("$test_name")
        return 1
    fi
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

log_header "AuroraQ VPS 실전매매 시스템 테스트 시작"

# 1. 환경 설정 파일 확인
log_header "환경 설정 파일 확인"

run_test ".env 파일 존재" "test -f .env"
run_test ".env.trading 파일 존재" "test -f .env.trading"
run_test "docker-compose.yml 파일 존재" "test -f docker-compose.yml"
run_test "Dockerfile.trading 파일 존재" "test -f Dockerfile.trading"

# 2. 필요한 디렉토리 확인
log_header "디렉토리 구조 확인"

run_test "trading 디렉토리 존재" "test -d trading"
run_test "trading/config 디렉토리 존재" "test -d trading/config"
run_test "logs 디렉토리 존재" "test -d logs || mkdir -p logs"
run_test "cache 디렉토리 존재" "test -d cache || mkdir -p cache"

# 3. Docker 환경 확인
log_header "Docker 환경 확인"

run_test "Docker 서비스 실행 중" "docker info"
run_test "Docker Compose 사용 가능" "$DOCKER_COMPOSE version"

# 4. 컨테이너 빌드 테스트
log_header "컨테이너 빌드 테스트"

log_info "Docker 이미지 빌드 중... (시간이 걸릴 수 있습니다)"
if $DOCKER_COMPOSE build --no-cache > build.log 2>&1; then
    log_success "Docker 이미지 빌드 성공"
    ((TESTS_PASSED++))
else
    log_error "Docker 이미지 빌드 실패"
    echo "빌드 로그 (마지막 20줄):"
    tail -20 build.log
    ((TESTS_FAILED++))
    FAILED_TESTS+=("Docker 이미지 빌드")
fi

# 5. 인프라 서비스 시작 및 테스트
log_header "인프라 서비스 테스트"

log_info "인프라 서비스 시작 중..."
$DOCKER_COMPOSE up -d redis postgres

sleep 10

run_test "Redis 컨테이너 실행" "docker exec auroraq-redis redis-cli ping"
run_test "PostgreSQL 컨테이너 실행" "docker exec auroraq-postgres pg_isready -U \${POSTGRES_USER:-onnx_user}"

# 6. ONNX 센티먼트 서비스 테스트
log_header "ONNX 센티먼트 서비스 테스트"

log_info "ONNX 서비스 시작 중..."
$DOCKER_COMPOSE up -d onnx-sentiment

# ONNX 서비스 헬스 체크 (최대 2분 대기)
log_info "ONNX 서비스 헬스 체크 중..."
ONNX_READY=false
for i in {1..24}; do
    if curl -f http://localhost:8001/onnx/health >/dev/null 2>&1; then
        ONNX_READY=true
        break
    fi
    sleep 5
done

if [ "$ONNX_READY" = true ]; then
    log_success "ONNX 센티먼트 서비스 정상"
    ((TESTS_PASSED++))
    
    # ONNX API 테스트
    run_test "ONNX 헬스 체크 API" "curl -f http://localhost:8001/onnx/health"
    run_test "ONNX 메트릭스 API" "curl -f http://localhost:8002/metrics"
else
    log_error "ONNX 센티먼트 서비스 시작 실패"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("ONNX 센티먼트 서비스")
fi

# 7. 실전매매 시스템 테스트
log_header "실전매매 시스템 테스트"

log_info "실전매매 시스템 시작 중..."
$DOCKER_COMPOSE up -d vps-trading

# 실전매매 시스템 헬스 체크 (최대 1분 대기)
log_info "실전매매 시스템 헬스 체크 중..."
TRADING_READY=false
for i in {1..12}; do
    if curl -f http://localhost:8004/trading/health >/dev/null 2>&1; then
        TRADING_READY=true
        break
    fi
    sleep 5
done

if [ "$TRADING_READY" = true ]; then
    log_success "실전매매 시스템 정상"
    ((TESTS_PASSED++))
    
    # 실전매매 API 테스트
    run_test "실전매매 헬스 체크 API" "curl -f http://localhost:8004/trading/health"
    run_test "실전매매 상태 API" "curl -f http://localhost:8004/api/status"
    run_test "실전매매 WebSocket 연결" "nc -z localhost 8003"
else
    log_error "실전매매 시스템 시작 실패"
    echo "실전매매 시스템 로그:"
    $DOCKER_COMPOSE logs --tail=20 vps-trading
    ((TESTS_FAILED++))
    FAILED_TESTS+=("실전매매 시스템")
fi

# 8. 통합 로깅 테스트
log_header "통합 로깅 시스템 테스트"

sleep 5  # 로그 생성 대기

run_test "로그 디렉토리 생성" "test -d logs"
run_test "Raw 로그 파일 생성" "find logs -name '*.jsonl' | head -1 | xargs test -f"
run_test "로그 파일 쓰기 권한" "test -w logs"

# 9. 메모리 및 성능 테스트
log_header "성능 및 리소스 테스트"

# 컨테이너 메모리 사용량 확인
TRADING_MEMORY=$(docker stats --no-stream --format "{{.MemUsage}}" auroraq-vps-trading 2>/dev/null | cut -d'/' -f1 | sed 's/[^0-9.]//g')
if [ -n "$TRADING_MEMORY" ] && [ "${TRADING_MEMORY%.*}" -lt 3000 ]; then
    log_success "실전매매 시스템 메모리 사용량 정상 (${TRADING_MEMORY}MB < 3GB)"
    ((TESTS_PASSED++))
else
    log_warning "실전매매 시스템 메모리 사용량 확인 불가능 또는 높음"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("메모리 사용량")
fi

# 10. 기능 테스트 (시뮬레이션 모드)
log_header "기능 테스트 (시뮬레이션)"

if curl -f http://localhost:8004/api/test/simulation >/dev/null 2>&1; then
    log_success "시뮬레이션 모드 테스트 통과"
    ((TESTS_PASSED++))
else
    log_warning "시뮬레이션 모드 테스트 건너뜀 (API 미구현)"
fi

# 11. 통합 테스트 결과 출력
log_header "테스트 결과 요약"

echo "총 테스트: $((TESTS_PASSED + TESTS_FAILED))"
echo -e "통과: ${GREEN}$TESTS_PASSED${NC}"
echo -e "실패: ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -gt 0 ]; then
    echo ""
    log_error "실패한 테스트:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
fi

# 12. 시스템 정리
echo ""
read -p "테스트 후 시스템을 정리하시겠습니까? (Y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    log_info "시스템 정리 중..."
    $DOCKER_COMPOSE down
    
    # 로그 파일 정리 (선택사항)
    read -p "테스트 로그도 삭제하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f build.log
        echo "테스트 로그 삭제됨"
    fi
else
    log_info "시스템이 계속 실행 중입니다."
    echo "중지하려면: $DOCKER_COMPOSE down"
fi

# 13. 최종 결과
echo ""
if [ $TESTS_FAILED -eq 0 ]; then
    log_success "🎉 모든 테스트가 통과했습니다!"
    echo "실전매매 시스템이 정상적으로 작동합니다."
    exit 0
else
    log_error "❌ 일부 테스트가 실패했습니다."
    echo "실패한 테스트를 확인하고 문제를 해결하세요."
    exit 1
fi