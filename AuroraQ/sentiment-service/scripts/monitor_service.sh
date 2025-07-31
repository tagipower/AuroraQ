#!/bin/bash

# AuroraQ Sentiment Service 모니터링 스크립트
# VPS에서 서비스 상태를 지속적으로 모니터링

set -e

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 설정
VPS_IP="109.123.239.30"
SERVICE_NAME="aurora-sentiment"
DEPLOY_DIR="/opt/aurora-sentiment"
LOG_FILE="/var/log/aurora-sentiment-monitor.log"
ALERT_EMAIL="${ALERT_EMAIL:-admin@example.com}"

# 로그 함수
log_info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# 시스템 리소스 모니터링
monitor_system_resources() {
    log_info "시스템 리소스 모니터링 중..."
    
    # CPU 사용률
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2+$4}' | sed 's/%us,//')
    
    # 메모리 사용률
    MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    
    # 디스크 사용률
    DISK_USAGE=$(df -h / | awk 'NR==2{printf("%s"), $5}' | sed 's/%//')
    
    log_info "시스템 리소스 - CPU: ${CPU_USAGE}%, Memory: ${MEMORY_USAGE}%, Disk: ${DISK_USAGE}%"
    
    # 임계값 확인
    if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
        log_warning "CPU 사용률이 높습니다: ${CPU_USAGE}%"
    fi
    
    if (( $(echo "$MEMORY_USAGE > 85" | bc -l) )); then
        log_warning "메모리 사용률이 높습니다: ${MEMORY_USAGE}%"
    fi
    
    if [ "$DISK_USAGE" -gt 85 ]; then
        log_warning "디스크 사용률이 높습니다: ${DISK_USAGE}%"
    fi
}

# Docker 컨테이너 상태 확인
check_docker_containers() {
    log_info "Docker 컨테이너 상태 확인 중..."
    
    cd "$DEPLOY_DIR" || exit 1
    
    # 컨테이너 상태 확인
    CONTAINERS=$(docker compose ps --format "table {{.Name}}\t{{.Status}}")
    echo "$CONTAINERS"
    
    # 실행 중이지 않은 컨테이너 확인
    FAILED_CONTAINERS=$(docker compose ps --filter "status=exited" --format "{{.Name}}")
    
    if [ -n "$FAILED_CONTAINERS" ]; then
        log_error "실행 중이지 않은 컨테이너: $FAILED_CONTAINERS"
        
        # 자동 재시작 시도
        log_info "컨테이너 재시작 시도 중..."
        docker compose restart $FAILED_CONTAINERS
        
        sleep 10
        
        # 재시작 후 상태 확인
        STILL_FAILED=$(docker compose ps --filter "status=exited" --format "{{.Name}}")
        if [ -n "$STILL_FAILED" ]; then
            log_error "재시작 실패한 컨테이너: $STILL_FAILED"
            return 1
        else
            log_success "컨테이너 재시작 성공"
        fi
    else
        log_success "모든 컨테이너가 정상 실행 중"
    fi
}

# API 헬스체크
check_api_health() {
    log_info "API 헬스체크 수행 중..."
    
    # FastAPI 헬스체크
    if curl -f -s --connect-timeout 10 "http://localhost:8000/health" > /dev/null; then
        log_success "FastAPI 헬스체크 성공"
        
        # API 응답 시간 측정
        RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' "http://localhost:8000/health")
        log_info "API 응답 시간: ${RESPONSE_TIME}초"
        
        # 응답 시간 임계값 확인 (5초)
        if (( $(echo "$RESPONSE_TIME > 5.0" | bc -l) )); then
            log_warning "API 응답 시간이 느립니다: ${RESPONSE_TIME}초"
        fi
    else
        log_error "FastAPI 헬스체크 실패"
        return 1
    fi
    
    # Prometheus 메트릭 확인
    if curl -f -s --connect-timeout 10 "http://localhost:9090/-/healthy" > /dev/null; then
        log_success "Prometheus 헬스체크 성공"
    else
        log_warning "Prometheus 헬스체크 실패"
    fi
}

# 로그 분석
analyze_logs() {
    log_info "최근 로그 분석 중..."
    
    cd "$DEPLOY_DIR" || exit 1
    
    # 최근 5분간 에러 로그 확인
    ERROR_COUNT=$(docker compose logs --since=5m sentiment-service 2>&1 | grep -c "ERROR" || true)
    
    if [ "$ERROR_COUNT" -gt 0 ]; then
        log_warning "최근 5분간 에러 로그 ${ERROR_COUNT}개 발견"
        
        # 최근 에러 로그 출력
        echo "최근 에러 로그:"
        docker compose logs --since=5m sentiment-service 2>&1 | grep "ERROR" | tail -5
    else
        log_info "최근 5분간 에러 로그 없음"
    fi
}

# 배치 스케줄러 상태 확인
check_batch_scheduler() {
    log_info "배치 스케줄러 상태 확인 중..."
    
    # 스케줄러 API를 통한 상태 확인
    SCHEDULER_RESPONSE=$(curl -s --connect-timeout 10 "http://localhost:8000/api/v1/scheduler/stats" || echo "error")
    
    if [[ "$SCHEDULER_RESPONSE" == *"success"* ]]; then
        log_success "배치 스케줄러 정상 동작 중"
        
        # 실행 중인 작업 수 확인
        RUNNING_TASKS=$(echo "$SCHEDULER_RESPONSE" | jq -r '.stats.running_tasks_count // 0' 2>/dev/null || echo "0")
        SCHEDULED_TASKS=$(echo "$SCHEDULER_RESPONSE" | jq -r '.stats.scheduled_tasks_count // 0' 2>/dev/null || echo "0")
        
        log_info "배치 스케줄러 - 실행 중: ${RUNNING_TASKS}, 예약됨: ${SCHEDULED_TASKS}"
    else
        log_error "배치 스케줄러 상태 확인 실패"
        return 1
    fi
}

# 디스크 공간 정리
cleanup_disk_space() {
    log_info "디스크 공간 정리 중..."
    
    # Docker 정리
    docker system prune -f --volumes > /dev/null 2>&1 || true
    
    # 오래된 로그 파일 정리 (30일 이상)
    find /var/log -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true
    find "$DEPLOY_DIR/logs" -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true
    
    # 임시 파일 정리
    find /tmp -type f -atime +7 -delete 2>/dev/null || true
    
    log_success "디스크 공간 정리 완료"
}

# 알림 전송
send_alert() {
    local message="$1"
    local severity="$2"
    
    log_info "알림 전송: $message"
    
    # 이메일 알림 (mailutils 설치 필요)
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "[${severity}] AuroraQ Sentiment Service Alert" "$ALERT_EMAIL"
    fi
    
    # Slack 웹훅 (설정 시)
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"[${severity}] AuroraQ Sentiment Service: $message\"}" \
             "$SLACK_WEBHOOK_URL" 2>/dev/null || true
    fi
}

# 자동 복구 시도
auto_recovery() {
    log_info "자동 복구 시도 중..."
    
    # 서비스 재시작
    systemctl restart "$SERVICE_NAME"
    
    # 재시작 후 대기
    sleep 30
    
    # 재시작 후 상태 확인
    if check_api_health && check_docker_containers; then
        log_success "자동 복구 성공"
        send_alert "서비스가 자동으로 복구되었습니다." "INFO"
        return 0
    else
        log_error "자동 복구 실패"
        send_alert "자동 복구에 실패했습니다. 수동 개입이 필요합니다." "CRITICAL"
        return 1
    fi
}

# 전체 헬스체크
full_health_check() {
    local failed=0
    
    log_info "=== 전체 헬스체크 시작 ==="
    
    # 시스템 리소스 모니터링
    monitor_system_resources
    
    # Docker 컨테이너 확인
    if ! check_docker_containers; then
        failed=1
    fi
    
    # API 헬스체크
    if ! check_api_health; then
        failed=1
    fi
    
    # 배치 스케줄러 확인
    if ! check_batch_scheduler; then
        failed=1
    fi
    
    # 로그 분석
    analyze_logs
    
    if [ $failed -eq 1 ]; then
        log_error "헬스체크 실패 - 자동 복구 시도"
        auto_recovery
    else
        log_success "모든 헬스체크 통과"
    fi
    
    log_info "=== 전체 헬스체크 완료 ==="
}

# 상세 상태 보고서
generate_status_report() {
    log_info "=== 상태 보고서 ==="
    
    echo "시간: $(date)"
    echo "호스트: $(hostname)"
    echo "가동 시간: $(uptime)"
    echo ""
    
    # 시스템 리소스
    echo "=== 시스템 리소스 ==="
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2+$4}')"
    echo "메모리: $(free -h)"
    echo "디스크: $(df -h /)"
    echo ""
    
    # Docker 상태
    echo "=== Docker 컨테이너 ==="
    cd "$DEPLOY_DIR" && docker compose ps
    echo ""
    
    # 서비스 상태
    echo "=== 서비스 상태 ==="
    systemctl status "$SERVICE_NAME" --no-pager -l
    echo ""
    
    # API 테스트
    echo "=== API 테스트 ==="
    curl -s "http://localhost:8000/health" | jq . 2>/dev/null || echo "API 응답 없음"
    echo ""
    
    # 최근 로그 (마지막 10줄)
    echo "=== 최근 로그 ==="
    cd "$DEPLOY_DIR" && docker compose logs --tail=10 sentiment-service
}

# 사용법 출력
usage() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  health      - 전체 헬스체크 수행"
    echo "  monitor     - 지속적 모니터링 모드"
    echo "  status      - 상세 상태 보고서 출력"
    echo "  cleanup     - 디스크 공간 정리"
    echo "  recovery    - 자동 복구 시도"
    echo "  help        - 도움말 출력"
}

# 지속적 모니터링 모드
continuous_monitoring() {
    log_info "지속적 모니터링 모드 시작"
    
    while true; do
        full_health_check
        
        # 5분 대기
        log_info "5분 후 다음 모니터링 수행..."
        sleep 300
    done
}

# 메인 함수
main() {
    case "${1:-health}" in
        "health")
            full_health_check
            ;;
        "monitor")
            continuous_monitoring
            ;;
        "status")
            generate_status_report
            ;;
        "cleanup")
            cleanup_disk_space
            ;;
        "recovery")
            auto_recovery
            ;;
        "help")
            usage
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            usage
            exit 1
            ;;
    esac
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi