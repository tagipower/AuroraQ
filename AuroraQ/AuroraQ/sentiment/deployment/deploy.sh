#!/bin/bash
# Sentiment Service VPS Deployment Script
# AuroraQ 감정 서비스 VPS 배포 스크립트

set -e

echo "=== AuroraQ Sentiment Service VPS Deployment ==="

# 설정
SERVICE_NAME="auroaq-sentiment-service"
SERVICE_USER="auroaq"
SERVICE_DIR="/opt/auroaq/sentiment-service"
VENV_DIR="/opt/auroaq/venv"
LOG_DIR="/var/log/auroaq"
SYSTEMD_SERVICE="/etc/systemd/system/${SERVICE_NAME}.service"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 루트 권한 확인
if [[ $EUID -ne 0 ]]; then
   log_error "This script must be run as root"
   exit 1
fi

# 시스템 요구사항 확인
check_requirements() {
    log_info "Checking system requirements..."
    
    # 메모리 확인 (최소 4GB)
    MEM_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    if [ "$MEM_GB" -lt 4 ]; then
        log_warn "Low memory detected: ${MEM_GB}GB (recommended: 4GB+)"
    else
        log_info "Memory check passed: ${MEM_GB}GB"
    fi
    
    # 디스크 공간 확인 (최소 10GB)
    DISK_GB=$(df -BG / | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
    if [ "$DISK_GB" -lt 10 ]; then
        log_error "Insufficient disk space: ${DISK_GB}GB (required: 10GB+)"
        exit 1
    else
        log_info "Disk space check passed: ${DISK_GB}GB available"
    fi
    
    # Python 3.8+ 확인
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Python version: $PYTHON_VERSION"
    else
        log_error "Python 3.8+ is required"
        exit 1
    fi
}

# 사용자 및 디렉토리 설정
setup_user_and_dirs() {
    log_info "Setting up user and directories..."
    
    # 서비스 사용자 생성
    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/false -d "$SERVICE_DIR" "$SERVICE_USER"
        log_info "Created service user: $SERVICE_USER"
    else
        log_info "Service user already exists: $SERVICE_USER"
    fi
    
    # 디렉토리 생성
    mkdir -p "$SERVICE_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "/etc/auroaq"
    
    # 권한 설정
    chown -R "$SERVICE_USER:$SERVICE_USER" "$SERVICE_DIR"
    chown -R "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
    
    log_info "Directories configured"
}

# Python 가상환경 설정
setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    # 시스템 패키지 설치
    apt-get update
    apt-get install -y python3-pip python3-venv python3-dev build-essential \
                       postgresql-client redis-tools curl wget git
    
    # 가상환경 생성
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_info "Created virtual environment: $VENV_DIR"
    fi
    
    # 가상환경 활성화 및 패키지 설치
    source "$VENV_DIR/bin/activate"
    
    # pip 업그레이드
    pip install --upgrade pip setuptools wheel
    
    # 기본 패키지 설치
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers datasets tokenizers
    pip install asyncio aiohttp beautifulsoup4 lxml
    pip install numpy pandas scipy scikit-learn
    pip install psutil redis psycopg2-binary
    pip install apscheduler backoff
    pip install pydantic fastapi uvicorn
    pip install python-dotenv pyyaml
    
    log_info "Python environment configured"
}

# 서비스 파일 복사
copy_service_files() {
    log_info "Copying service files..."
    
    # 현재 디렉토리에서 서비스 파일들 복사
    if [ -d "./sentiment-service" ]; then
        cp -r ./sentiment-service/* "$SERVICE_DIR/"
        log_info "Service files copied"
    else
        log_error "Service files not found in current directory"
        exit 1
    fi
    
    # 권한 설정
    chown -R "$SERVICE_USER:$SERVICE_USER" "$SERVICE_DIR"
    chmod +x "$SERVICE_DIR/deployment/service_runner.py"
}

# 설정 파일 생성
create_config_files() {
    log_info "Creating configuration files..."
    
    # 환경 변수 파일
    cat > "/etc/auroaq/sentiment-service.env" << EOF
# AuroraQ Sentiment Service Environment Variables
DEPLOYMENT_MODE=vps
DEBUG=false

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=auroaq_sentiment
DB_USER=auroaq
DB_PASSWORD=change_this_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# API Keys (configure these)
NEWSAPI_KEY=your_newsapi_key_here
FINNHUB_KEY=your_finnhub_key_here
OPENAI_KEY=your_openai_key_here

# Monitoring
DISCORD_WEBHOOK=
SLACK_WEBHOOK=
EOF
    
    chmod 600 "/etc/auroaq/sentiment-service.env"
    chown "$SERVICE_USER:$SERVICE_USER" "/etc/auroaq/sentiment-service.env"
    
    log_info "Configuration files created"
    log_warn "Please update API keys in /etc/auroaq/sentiment-service.env"
}

# systemd 서비스 생성
create_systemd_service() {
    log_info "Creating systemd service..."
    
    cat > "$SYSTEMD_SERVICE" << EOF
[Unit]
Description=AuroraQ Sentiment Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$SERVICE_DIR
Environment=PATH=$VENV_DIR/bin
EnvironmentFile=/etc/auroaq/sentiment-service.env
ExecStart=$VENV_DIR/bin/python $SERVICE_DIR/deployment/service_runner.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# 리소스 제한 (VPS 최적화)
MemoryMax=3G
CPUQuota=200%
TasksMax=100

# 보안 설정
NoNewPrivileges=true
PrivateTmp=true
PrivateDevices=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=$LOG_DIR $SERVICE_DIR/cache

[Install]
WantedBy=multi-user.target
EOF
    
    # systemd 재로드
    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    
    log_info "Systemd service created and enabled"
}

# 방화벽 설정
configure_firewall() {
    log_info "Configuring firewall..."
    
    if command -v ufw &> /dev/null; then
        # 헬스체크 포트 허용
        ufw allow 8080/tcp comment "AuroraQ Health Check"
        ufw allow 8081/tcp comment "AuroraQ Metrics"
        log_info "Firewall rules added"
    else
        log_warn "UFW not found, please configure firewall manually"
    fi
}

# 로그 회전 설정
setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    cat > "/etc/logrotate.d/$SERVICE_NAME" << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $SERVICE_USER $SERVICE_USER
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
    
    log_info "Log rotation configured"
}

# 헬스체크 스크립트 생성
create_health_check() {
    log_info "Creating health check script..."
    
    cat > "/usr/local/bin/${SERVICE_NAME}-health" << EOF
#!/bin/bash
# Health check script for AuroraQ Sentiment Service

HEALTH_URL="http://localhost:8080/health"
TIMEOUT=10

if curl -s --max-time \$TIMEOUT "\$HEALTH_URL" > /dev/null; then
    echo "Service is healthy"
    exit 0
else
    echo "Service health check failed"
    exit 1
fi
EOF
    
    chmod +x "/usr/local/bin/${SERVICE_NAME}-health"
    log_info "Health check script created"
}

# 서비스 시작
start_service() {
    log_info "Starting service..."
    
    # 서비스 시작
    systemctl start "$SERVICE_NAME"
    
    # 상태 확인
    sleep 5
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log_info "Service started successfully"
        
        # 헬스체크 시도
        sleep 10
        if "/usr/local/bin/${SERVICE_NAME}-health" &> /dev/null; then
            log_info "Health check passed"
        else
            log_warn "Health check failed, service may still be starting"
        fi
    else
        log_error "Service failed to start"
        systemctl status "$SERVICE_NAME"
        exit 1
    fi
}

# 배포 요약
deployment_summary() {
    echo
    log_info "=== Deployment Summary ==="
    echo "Service Name: $SERVICE_NAME"
    echo "Service Directory: $SERVICE_DIR"
    echo "Log Directory: $LOG_DIR"
    echo "Config File: /etc/auroaq/sentiment-service.env"
    echo "Systemd Service: $SYSTEMD_SERVICE"
    echo
    log_info "Useful Commands:"
    echo "  - Start service: systemctl start $SERVICE_NAME"
    echo "  - Stop service: systemctl stop $SERVICE_NAME"
    echo "  - Restart service: systemctl restart $SERVICE_NAME"
    echo "  - View logs: journalctl -u $SERVICE_NAME -f"
    echo "  - Health check: /usr/local/bin/${SERVICE_NAME}-health"
    echo "  - Service status: systemctl status $SERVICE_NAME"
    echo
    log_warn "Next Steps:"
    echo "  1. Configure API keys in /etc/auroaq/sentiment-service.env"
    echo "  2. Set up PostgreSQL database"
    echo "  3. Set up Redis server"
    echo "  4. Restart service: systemctl restart $SERVICE_NAME"
    echo
}

# 메인 실행
main() {
    log_info "Starting AuroraQ Sentiment Service deployment..."
    
    check_requirements
    setup_user_and_dirs
    setup_python_environment
    copy_service_files
    create_config_files
    create_systemd_service
    configure_firewall
    setup_log_rotation
    create_health_check
    start_service
    
    deployment_summary
    
    log_info "Deployment completed successfully!"
}

# 스크립트 실행
main "$@"