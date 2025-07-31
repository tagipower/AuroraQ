#!/bin/bash

# AuroraQ Sentiment Service Cron Job 설정 스크립트
# 자동 모니터링 및 유지보수 작업 스케줄링

set -e

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 설정
DEPLOY_DIR="/opt/aurora-sentiment"
SCRIPT_DIR="${DEPLOY_DIR}/scripts"
LOG_DIR="/var/log/aurora-sentiment"

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

# 로그 디렉토리 생성
create_log_directory() {
    log_info "로그 디렉토리 생성 중..."
    
    mkdir -p "$LOG_DIR"
    chmod 755 "$LOG_DIR"
    
    log_success "로그 디렉토리 생성 완료: $LOG_DIR"
}

# Cron Job 설정
setup_cron_jobs() {
    log_info "Cron Job 설정 중..."
    
    # 기존 cron job 백업
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    
    # 새로운 cron job 작성
    cat > /tmp/aurora_sentiment_cron << EOF
# AuroraQ Sentiment Service Cron Jobs
# 자동 생성된 파일 - 수정하지 마세요

# 1. 헬스체크 (5분마다)
*/5 * * * * ${SCRIPT_DIR}/monitor_service.sh health >> ${LOG_DIR}/health_check.log 2>&1

# 2. 상세 상태 보고서 (1시간마다)
0 * * * * ${SCRIPT_DIR}/monitor_service.sh status >> ${LOG_DIR}/status_report.log 2>&1

# 3. 디스크 정리 (매일 새벽 2시)
0 2 * * * ${SCRIPT_DIR}/monitor_service.sh cleanup >> ${LOG_DIR}/maintenance.log 2>&1

# 4. 로그 회전 (매일 새벽 3시)
0 3 * * * ${SCRIPT_DIR}/rotate_logs.sh >> ${LOG_DIR}/log_rotation.log 2>&1

# 5. 시스템 백업 (주 1회, 일요일 새벽 1시)
0 1 * * 0 ${SCRIPT_DIR}/backup_system.sh >> ${LOG_DIR}/backup.log 2>&1

# 6. SSL 인증서 갱신 체크 (월 1회, 매월 1일 새벽 4시)
0 4 1 * * ${SCRIPT_DIR}/check_ssl.sh >> ${LOG_DIR}/ssl_check.log 2>&1

# 7. 도커 시스템 정리 (주 1회, 토요일 새벽 1시)
0 1 * * 6 /usr/bin/docker system prune -f --volumes >> ${LOG_DIR}/docker_cleanup.log 2>&1

# 8. API 키 유효성 검사 (매일 새벽 5시)
0 5 * * * ${SCRIPT_DIR}/validate_api_keys.sh >> ${LOG_DIR}/api_validation.log 2>&1

EOF
    
    # cron job 설치
    crontab /tmp/aurora_sentiment_cron
    
    # 임시 파일 정리
    rm /tmp/aurora_sentiment_cron
    
    log_success "Cron Job 설정 완료"
}

# 로그 로테이션 스크립트 생성
create_log_rotation_script() {
    log_info "로그 로테이션 스크립트 생성 중..."
    
    cat > "${SCRIPT_DIR}/rotate_logs.sh" << 'EOF'
#!/bin/bash

# 로그 로테이션 스크립트
LOG_DIR="/var/log/aurora-sentiment"
DEPLOY_DIR="/opt/aurora-sentiment"
RETENTION_DAYS=30

# 시스템 로그 로테이션
find "$LOG_DIR" -name "*.log" -type f -mtime +$RETENTION_DAYS -delete

# Docker 로그 로테이션
cd "$DEPLOY_DIR"
docker compose logs --no-color > "${LOG_DIR}/docker_full_$(date +%Y%m%d).log"

# 오래된 Docker 로그 정리
find "$LOG_DIR" -name "docker_full_*.log" -type f -mtime +7 -delete

# 애플리케이션 로그 압축
find "${DEPLOY_DIR}/logs" -name "*.log" -type f -mtime +1 -exec gzip {} \;

# 압축된 로그 정리
find "${DEPLOY_DIR}/logs" -name "*.log.gz" -type f -mtime +$RETENTION_DAYS -delete

echo "로그 로테이션 완료: $(date)"
EOF
    
    chmod +x "${SCRIPT_DIR}/rotate_logs.sh"
    
    log_success "로그 로테이션 스크립트 생성 완료"
}

# 시스템 백업 스크립트 생성
create_backup_script() {
    log_info "시스템 백업 스크립트 생성 중..."
    
    cat > "${SCRIPT_DIR}/backup_system.sh" << 'EOF'
#!/bin/bash

# 시스템 백업 스크립트
DEPLOY_DIR="/opt/aurora-sentiment"
BACKUP_DIR="/opt/aurora-sentiment-backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="aurora_sentiment_backup_${DATE}"

# 백업 디렉토리 생성
mkdir -p "$BACKUP_DIR"

# 애플리케이션 백업
echo "애플리케이션 백업 시작: $(date)"

# 설정 파일 백업
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}_config.tar.gz" \
    -C "$DEPLOY_DIR" \
    .env docker-compose.yml nginx/ config/ scripts/

# 데이터 백업
if [ -d "${DEPLOY_DIR}/data" ]; then
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}_data.tar.gz" \
        -C "$DEPLOY_DIR" data/
fi

# 로그 백업 (최근 7일)
find "${DEPLOY_DIR}/logs" -name "*.log*" -mtime -7 | \
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}_logs.tar.gz" -T -

# Docker 볼륨 백업
cd "$DEPLOY_DIR"
docker run --rm -v aurora-sentiment_redis_data:/data \
    -v "$BACKUP_DIR":/backup \
    alpine tar -czf "/backup/${BACKUP_NAME}_redis.tar.gz" -C /data .

# 오래된 백업 정리 (30일 이상)
find "$BACKUP_DIR" -name "aurora_sentiment_backup_*" -type f -mtime +30 -delete

echo "백업 완료: $(date)"
echo "백업 파일 위치: $BACKUP_DIR"
EOF
    
    chmod +x "${SCRIPT_DIR}/backup_system.sh"
    
    log_success "시스템 백업 스크립트 생성 완료"
}

# SSL 인증서 체크 스크립트 생성
create_ssl_check_script() {
    log_info "SSL 인증서 체크 스크립트 생성 중..."
    
    cat > "${SCRIPT_DIR}/check_ssl.sh" << 'EOF'
#!/bin/bash

# SSL 인증서 만료일 체크 스크립트
DEPLOY_DIR="/opt/aurora-sentiment"
DOMAIN="109.123.239.30"  # VPS IP
DAYS_WARNING=30

# SSL 인증서 존재 확인
if [ -f "${DEPLOY_DIR}/nginx/ssl/cert.pem" ]; then
    # 인증서 만료일 확인
    EXPIRY_DATE=$(openssl x509 -in "${DEPLOY_DIR}/nginx/ssl/cert.pem" -noout -dates | grep notAfter | cut -d= -f2)
    EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
    CURRENT_EPOCH=$(date +%s)
    DAYS_LEFT=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))
    
    echo "SSL 인증서 만료일: $EXPIRY_DATE"
    echo "남은 일수: $DAYS_LEFT days"
    
    if [ $DAYS_LEFT -lt $DAYS_WARNING ]; then
        echo "WARNING: SSL 인증서가 ${DAYS_LEFT}일 후 만료됩니다!"
        
        # Let's Encrypt 자동 갱신 시도 (certbot 설치되어 있다면)
        if command -v certbot &> /dev/null; then
            echo "Let's Encrypt 인증서 갱신 시도 중..."
            certbot renew --nginx --quiet
            
            if [ $? -eq 0 ]; then
                echo "SSL 인증서 갱신 성공"
                systemctl reload nginx
            else
                echo "SSL 인증서 갱신 실패"
            fi
        fi
    else
        echo "SSL 인증서 상태 양호"
    fi
else
    echo "SSL 인증서 파일을 찾을 수 없습니다."
fi
EOF
    
    chmod +x "${SCRIPT_DIR}/check_ssl.sh"
    
    log_success "SSL 인증서 체크 스크립트 생성 완료"
}

# API 키 유효성 검사 스크립트 생성
create_api_validation_script() {
    log_info "API 키 유효성 검사 스크립트 생성 중..."
    
    cat > "${SCRIPT_DIR}/validate_api_keys.sh" << 'EOF'
#!/bin/bash

# API 키 유효성 검사 스크립트
DEPLOY_DIR="/opt/aurora-sentiment"

echo "API 키 유효성 검사 시작: $(date)"

# .env 파일에서 환경변수 로드
if [ -f "${DEPLOY_DIR}/.env" ]; then
    source "${DEPLOY_DIR}/.env"
else
    echo "ERROR: .env 파일을 찾을 수 없습니다."
    exit 1
fi

# API 키 테스트 함수
test_api_key() {
    local service="$1"
    local url="$2"
    local headers="$3"
    
    echo -n "$service API 테스트... "
    
    if curl -s --connect-timeout 10 --max-time 30 $headers "$url" >/dev/null 2>&1; then
        echo "OK"
        return 0
    else
        echo "FAIL"
        return 1
    fi
}

# 각 API 서비스 테스트
failed_services=0

# NewsAPI 테스트
if [ -n "$NEWSAPI_KEY" ]; then
    if ! test_api_key "NewsAPI" "https://newsapi.org/v2/top-headlines?country=us&apiKey=$NEWSAPI_KEY" ""; then
        failed_services=$((failed_services + 1))
    fi
fi

# Finnhub 테스트
if [ -n "$FINNHUB_API_KEY" ]; then
    if ! test_api_key "Finnhub" "https://finnhub.io/api/v1/quote?symbol=AAPL&token=$FINNHUB_API_KEY" ""; then
        failed_services=$((failed_services + 1))
    fi
fi

# Google Custom Search 테스트
if [ -n "$GOOGLE_SEARCH_API_KEY" ] && [ -n "$GOOGLE_CUSTOM_SEARCH_ID" ]; then
    if ! test_api_key "Google Search" "https://www.googleapis.com/customsearch/v1?key=$GOOGLE_SEARCH_API_KEY&cx=$GOOGLE_CUSTOM_SEARCH_ID&q=test" ""; then
        failed_services=$((failed_services + 1))
    fi
fi

# Reddit API 테스트 (OAuth 필요하므로 간단한 연결 테스트만)
if [ -n "$REDDIT_CLIENT_ID" ] && [ -n "$REDDIT_CLIENT_SECRET" ]; then
    if ! test_api_key "Reddit" "https://www.reddit.com/api/v1/access_token" "-X POST -H 'User-Agent: AuroraQ-Test/1.0'"; then
        failed_services=$((failed_services + 1))
    fi
fi

# 결과 출력
if [ $failed_services -eq 0 ]; then
    echo "모든 API 키 테스트 통과"
else
    echo "WARNING: ${failed_services}개의 API 서비스에서 문제 발생"
fi

echo "API 키 유효성 검사 완료: $(date)"
EOF
    
    chmod +x "${SCRIPT_DIR}/validate_api_keys.sh"
    
    log_success "API 키 유효성 검사 스크립트 생성 완료"
}

# Cron 상태 확인
check_cron_status() {
    log_info "Cron 서비스 상태 확인 중..."
    
    # Cron 서비스 상태
    if systemctl is-active --quiet cron; then
        log_success "Cron 서비스 실행 중"
    else
        log_warning "Cron 서비스가 실행되지 않음 - 시작 시도"
        systemctl start cron
        systemctl enable cron
    fi
    
    # 설정된 Cron Job 확인
    echo ""
    echo "=== 현재 설정된 Cron Jobs ==="
    crontab -l | grep -v "^#" | grep -v "^$"
    
    log_success "Cron 상태 확인 완료"
}

# 로그 로테이션 설정 (logrotate)
setup_logrotate() {
    log_info "Logrotate 설정 중..."
    
    cat > /etc/logrotate.d/aurora-sentiment << 'EOF'
/var/log/aurora-sentiment/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        # 애플리케이션 로그 재시작 신호 (필요시)
        systemctl reload aurora-sentiment || true
    endscript
}

/opt/aurora-sentiment/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    copytruncate
}
EOF
    
    # Logrotate 설정 테스트
    logrotate -d /etc/logrotate.d/aurora-sentiment
    
    log_success "Logrotate 설정 완료"
}

# 모니터링 대시보드 URL 생성
create_monitoring_dashboard() {
    log_info "모니터링 대시보드 정보 생성 중..."
    
    cat > "${DEPLOY_DIR}/MONITORING.md" << 'EOF'
# AuroraQ Sentiment Service 모니터링 대시보드

## 서비스 접속 정보
- **API 서버**: http://109.123.239.30:8000
- **API 문서**: http://109.123.239.30:8000/docs
- **헬스체크**: http://109.123.239.30:8000/health
- **Prometheus**: http://109.123.239.30:9090

## 모니터링 명령어

### 서비스 상태 확인
```bash
# 전체 헬스체크
./scripts/monitor_service.sh health

# 상세 상태 보고서
./scripts/monitor_service.sh status

# 지속적 모니터링
./scripts/monitor_service.sh monitor
```

### 로그 확인
```bash
# 실시간 로그
docker compose logs -f sentiment-service

# 특정 기간 로그
docker compose logs --since="2h" sentiment-service

# 에러 로그만
docker compose logs sentiment-service 2>&1 | grep ERROR
```

### 시스템 관리
```bash
# 서비스 재시작
systemctl restart aurora-sentiment

# 백업 수행
./scripts/backup_system.sh

# 디스크 정리
./scripts/monitor_service.sh cleanup
```

## 로그 파일 위치
- **애플리케이션 로그**: `/opt/aurora-sentiment/logs/`
- **시스템 로그**: `/var/log/aurora-sentiment/`
- **Docker 로그**: `docker compose logs`

## 알림 설정
모니터링 스크립트에서 다음 환경변수를 설정하여 알림을 받을 수 있습니다:

```bash
export ALERT_EMAIL="admin@example.com"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

## 자동화된 작업
- **헬스체크**: 5분마다
- **상태 보고서**: 1시간마다  
- **디스크 정리**: 매일 새벽 2시
- **로그 로테이션**: 매일 새벽 3시
- **시스템 백업**: 주 1회 (일요일 새벽 1시)
- **SSL 체크**: 월 1회 (매월 1일 새벽 4시)
EOF
    
    log_success "모니터링 대시보드 정보 생성 완료"
}

# 사용법 출력
usage() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  install     - 모든 Cron Job 및 스크립트 설치"
    echo "  status      - Cron 상태 확인"
    echo "  uninstall   - Cron Job 제거"
    echo "  help        - 도움말 출력"
}

# Cron Job 제거
uninstall_cron_jobs() {
    log_info "Cron Job 제거 중..."
    
    # 백업
    crontab -l > /tmp/crontab_backup_before_uninstall_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
    
    # AuroraQ 관련 cron job 제거
    crontab -l 2>/dev/null | grep -v "aurora-sentiment" | grep -v "AuroraQ" | crontab - || true
    
    log_success "Cron Job 제거 완료"
}

# 메인 함수
main() {
    case "${1:-install}" in
        "install")
            create_log_directory
            create_log_rotation_script
            create_backup_script
            create_ssl_check_script
            create_api_validation_script
            setup_cron_jobs
            setup_logrotate
            create_monitoring_dashboard
            check_cron_status
            log_success "모든 Cron Job 및 모니터링 설정 완료!"
            ;;
        "status")
            check_cron_status
            ;;
        "uninstall")
            uninstall_cron_jobs
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