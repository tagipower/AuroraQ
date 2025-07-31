#!/bin/bash

# AuroraQ Sentiment Service VPS 배포 스크립트
# VPS 정보: IP 109.123.239.30, Singapore, Ubuntu 22.04

set -e  # 에러 발생시 스크립트 중단

# 색상 코드
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

# 전역 변수
VPS_IP="109.123.239.30"
VPS_USER="${VPS_USER:-root}"
SERVICE_NAME="aurora-sentiment"
DEPLOY_DIR="/opt/aurora-sentiment"
BACKUP_DIR="/opt/aurora-sentiment-backup"

# SSH 키 확인
check_ssh_key() {
    log_info "SSH 연결 확인 중..."
    
    if ! ssh -o ConnectTimeout=10 aurora-vps exit &>/dev/null; then
        log_error "SSH 연결 실패. SSH 키를 확인해주세요."
        echo "다음 명령어로 SSH 키를 추가하세요:"
        echo "ssh-copy-id ${VPS_USER}@${VPS_IP}"
        exit 1
    fi
    
    log_success "SSH 연결 성공"
}

# VPS 시스템 요구사항 확인
check_system_requirements() {
    log_info "VPS 시스템 요구사항 확인 중..."
    
    ssh aurora-vps << 'EOF'
        # 운영체제 확인
        if ! grep -q "Ubuntu 22.04" /etc/os-release; then
            echo "WARNING: Ubuntu 22.04가 아닙니다."
        fi
        
        # 메모리 확인 (최소 2GB 권장)
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY_GB" -lt 2 ]; then
            echo "WARNING: 메모리가 부족합니다. 최소 2GB 권장"
        fi
        
        # 디스크 공간 확인 (최소 10GB 필요)
        DISK_GB=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
        if [ "$DISK_GB" -lt 10 ]; then
            echo "ERROR: 디스크 공간이 부족합니다. 최소 10GB 필요"
            exit 1
        fi
        
        echo "시스템 요구사항 확인 완료"
EOF
    
    log_success "시스템 요구사항 확인 완료"
}

# Docker 설치
install_docker() {
    log_info "Docker 설치 중..."
    
    ssh aurora-vps << 'EOF'
        # Docker가 이미 설치되어 있는지 확인
        if command -v docker &> /dev/null; then
            echo "Docker가 이미 설치되어 있습니다."
            docker --version
            exit 0
        fi
        
        # 패키지 업데이트
        apt-get update
        
        # 필수 패키지 설치
        apt-get install -y \
            apt-transport-https \
            ca-certificates \
            curl \
            gnupg \
            lsb-release
        
        # Docker GPG 키 추가
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        
        # Docker 저장소 추가
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Docker 설치
        apt-get update
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        
        # Docker 서비스 시작 및 자동 시작 설정
        systemctl start docker
        systemctl enable docker
        
        # 현재 사용자를 docker 그룹에 추가
        usermod -aG docker $USER
        
        echo "Docker 설치 완료"
        docker --version
        docker compose version
EOF
    
    log_success "Docker 설치 완료"
}

# 방화벽 설정
configure_firewall() {
    log_info "방화벽 설정 중..."
    
    ssh aurora-vps << 'EOF'
        # UFW 설치 및 기본 설정
        apt-get install -y ufw
        
        # 기본 정책 설정
        ufw --force reset
        ufw default deny incoming
        ufw default allow outgoing
        
        # SSH 허용
        ufw allow ssh
        ufw allow 22/tcp
        
        # 서비스 포트 허용
        ufw allow 8000/tcp  # FastAPI
        ufw allow 9090/tcp  # Prometheus
        ufw allow 80/tcp    # HTTP
        ufw allow 443/tcp   # HTTPS
        
        # 방화벽 활성화
        ufw --force enable
        
        echo "방화벽 설정 완료"
        ufw status
EOF
    
    log_success "방화벽 설정 완료"
}

# 프로젝트 파일 전송
upload_project_files() {
    log_info "프로젝트 파일 전송 중..."
    
    # 백업 디렉토리 생성
    ssh aurora-vps "mkdir -p ${BACKUP_DIR}"
    
    # 기존 파일 백업 (있다면)
    ssh aurora-vps "
        if [ -d '${DEPLOY_DIR}' ]; then
            echo '기존 설치 발견, 백업 중...'
            cp -r ${DEPLOY_DIR} ${BACKUP_DIR}/backup-\$(date +%Y%m%d-%H%M%S)
        fi
    "
    
    # 배포 디렉토리 생성
    ssh aurora-vps "mkdir -p ${DEPLOY_DIR}"
    
    # 프로젝트 파일 업로드
    log_info "소스 코드 전송 중..."
    rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='logs/*' --exclude='data/*' --exclude='models/*' \
          ./ "aurora-vps:${DEPLOY_DIR}/"
    
    log_success "프로젝트 파일 전송 완료"
}

# 환경 설정 파일 생성
create_env_file() {
    log_info "환경 설정 파일 생성 중..."
    
    ssh aurora-vps << EOF
cat > ${DEPLOY_DIR}/.env << 'ENVEOF'
# Production Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=2

# Redis Configuration
REDIS_URL=redis://redis:6379/0
CACHE_TTL=300

# Model Configuration
FINBERT_MODEL_NAME=ProsusAI/finbert
FINBERT_MODEL_PATH=/app/models/finbert
FINBERT_CACHE_DIR=/app/cache/transformers
FINBERT_MAX_LENGTH=512
FINBERT_BATCH_SIZE=8

# Performance Configuration
ENABLE_MODEL_CACHING=true
MODEL_WARMUP=true
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=30.0

# Security Configuration
ALLOWED_HOSTS=${VPS_IP},localhost,127.0.0.1
CORS_ORIGINS=*

# Monitoring Configuration
ENABLE_METRICS=true
PROMETHEUS_PORT=8080

# External API Keys (수동으로 설정 필요)
GOOGLE_NEWS_API_KEY=
YAHOO_FINANCE_API_KEY=
NEWSAPI_KEY=
FINNHUB_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
GOOGLE_SEARCH_API_KEY=
GOOGLE_CUSTOM_SEARCH_ID=
BING_SEARCH_API_KEY=

# AuroraQ Integration
AURORA_API_URL=http://host.docker.internal:8080
AURORA_API_KEY=

# News Processing
NEWS_MAX_ARTICLES=30
NEWS_HOURS_BACK=12
NEWS_RELEVANCE_THRESHOLD=0.4
ENVEOF

echo "환경 설정 파일 생성 완료"
EOF
    
    log_success "환경 설정 파일 생성 완료"
    log_warning "API 키들을 수동으로 설정해야 합니다: ${DEPLOY_DIR}/.env"
}

# 디렉토리 구조 생성
create_directories() {
    log_info "필요한 디렉토리 생성 중..."
    
    ssh aurora-vps << EOF
        cd ${DEPLOY_DIR}
        
        # 데이터 디렉토리
        mkdir -p data logs models cache
        mkdir -p nginx/ssl
        mkdir -p monitoring/grafana/{dashboards,datasources}
        
        # 권한 설정
        chmod 755 data logs models cache
        chmod +x scripts/*.sh
        
        echo "디렉토리 구조 생성 완료"
EOF
    
    log_success "디렉토리 구조 생성 완료"
}

# Nginx 설정 생성
create_nginx_config() {
    log_info "Nginx 설정 생성 중..."
    
    ssh aurora-vps << EOF
cat > ${DEPLOY_DIR}/nginx/nginx.conf << 'NGINXEOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # 로그 포맷
    log_format main '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                    '\$status \$body_bytes_sent "\$http_referer" '
                    '"\$http_user_agent" "\$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    
    # 성능 최적화
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip 압축
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    
    upstream sentiment_backend {
        server sentiment-service:8000;
    }
    
    server {
        listen 80;
        server_name ${VPS_IP};
        
        # API 요청
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://sentiment_backend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # 타임아웃 설정
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # 헬스체크
        location /health {
            proxy_pass http://sentiment_backend;
            access_log off;
        }
        
        # 메트릭스 (내부 접근만)
        location /metrics {
            allow 127.0.0.1;
            allow 172.20.0.0/16;
            deny all;
            proxy_pass http://sentiment_backend;
        }
        
        # 정적 파일
        location /static/ {
            alias /app/static/;
            expires 1d;
            add_header Cache-Control "public, immutable";
        }
        
        # 기본 페이지
        location / {
            proxy_pass http://sentiment_backend;
        }
    }
}
NGINXEOF

echo "Nginx 설정 생성 완료"
EOF
    
    log_success "Nginx 설정 생성 완료"
}

# Systemd 서비스 생성
create_systemd_service() {
    log_info "Systemd 서비스 생성 중..."
    
    ssh aurora-vps << EOF
cat > /etc/systemd/system/${SERVICE_NAME}.service << 'SERVICEEOF'
[Unit]
Description=AuroraQ Sentiment Analysis Service
After=docker.service
Requires=docker.service

[Service]
Type=forking
RemainAfterExit=yes
WorkingDirectory=${DEPLOY_DIR}
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose restart
TimeoutStartSec=0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

# 서비스 등록 및 활성화
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

echo "Systemd 서비스 생성 완료"
EOF
    
    log_success "Systemd 서비스 생성 완료"
}

# 모니터링 설정
setup_monitoring() {
    log_info "모니터링 설정 중..."
    
    ssh aurora-vps << EOF
        cd ${DEPLOY_DIR}
        
        # Prometheus 설정이 이미 있는지 확인
        if [ ! -f "config/prometheus.yml" ]; then
            echo "Prometheus 설정 파일이 없습니다. 기본 설정을 생성합니다."
            
            mkdir -p config
            cat > config/prometheus.yml << 'PROMEOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'aurora-sentiment'
    static_configs:
      - targets: ['sentiment-service:8080']
    scrape_interval: 30s
    metrics_path: '/metrics'
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
PROMEOF
        fi
        
        echo "모니터링 설정 완료"
EOF
    
    log_success "모니터링 설정 완료"
}

# 서비스 시작
start_services() {
    log_info "서비스 시작 중..."
    
    ssh aurora-vps << EOF
        cd ${DEPLOY_DIR}
        
        # Docker 이미지 빌드 및 서비스 시작
        docker compose build --no-cache
        docker compose up -d
        
        # 서비스 상태 확인
        sleep 30
        docker compose ps
        
        echo "서비스 시작 완료"
EOF
    
    log_success "서비스 시작 완료"
}

# 헬스체크
health_check() {
    log_info "헬스체크 수행 중..."
    
    sleep 10
    
    # API 헬스체크
    if curl -f "http://${VPS_IP}:8000/health" > /dev/null 2>&1; then
        log_success "API 헬스체크 성공"
    else
        log_error "API 헬스체크 실패"
        return 1
    fi
    
    # Prometheus 메트릭 확인
    if curl -f "http://${VPS_IP}:9090/-/healthy" > /dev/null 2>&1; then
        log_success "Prometheus 헬스체크 성공"
    else
        log_warning "Prometheus 헬스체크 실패"
    fi
    
    log_success "헬스체크 완료"
}

# 배포 정보 출력
print_deployment_info() {
    log_success "=== 배포 완료 ==="
    echo ""
    echo "서비스 접속 정보:"
    echo "  - API 서버: http://${VPS_IP}:8000"
    echo "  - API 문서: http://${VPS_IP}:8000/docs"
    echo "  - 헬스체크: http://${VPS_IP}:8000/health"
    echo "  - Prometheus: http://${VPS_IP}:9090"
    echo ""
    echo "유용한 명령어:"
    echo "  - 서비스 상태: ssh ${VPS_USER}@${VPS_IP} 'systemctl status ${SERVICE_NAME}'"
    echo "  - 로그 확인: ssh ${VPS_USER}@${VPS_IP} 'cd ${DEPLOY_DIR} && docker compose logs -f'"
    echo "  - 서비스 재시작: ssh ${VPS_USER}@${VPS_IP} 'systemctl restart ${SERVICE_NAME}'"
    echo ""
    echo "중요: ${DEPLOY_DIR}/.env 파일에서 API 키들을 설정해주세요!"
}

# 메인 배포 함수
main() {
    log_info "AuroraQ Sentiment Service VPS 배포 시작"
    log_info "VPS: ${VPS_USER}@${VPS_IP}"
    
    check_ssh_key
    check_system_requirements
    install_docker
    configure_firewall
    upload_project_files
    create_env_file
    create_directories
    create_nginx_config
    setup_monitoring
    create_systemd_service
    start_services
    health_check
    print_deployment_info
    
    log_success "배포가 완료되었습니다!"
}

# 스크립트 실행
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi