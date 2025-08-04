#!/bin/bash
# =============================================================================
# AuroraQ VPS 배포 스크립트 - 통합 버전
# VPS 서버에서 직접 실행하는 올인원 배포 스크립트
# =============================================================================

set -euo pipefail

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 설정 변수
PROJECT_NAME="auroraQ"
INSTALL_DIR="/opt/auroraQ"
VPS_HOST="${1:-$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')}"

# 로깅 함수
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() { echo -e "${PURPLE}$1${NC}\n${PURPLE}$(printf '=%.0s' {1..${#1}})${NC}"; }

# 메인 배포 함수
main() {
    log_header "🚀 AuroraQ VPS 배포 시작"
    
    echo "서버 IP: $VPS_HOST"
    echo "설치 경로: $INSTALL_DIR"
    echo "현재 시간: $(date)"
    echo ""
    
    # 실행 확인
    read -p "AuroraQ 시스템을 배포하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "배포가 취소되었습니다."
        exit 0
    fi
    
    # 단계별 실행
    install_prerequisites
    setup_firewall
    create_project_structure
    create_docker_files
    deploy_containers
    setup_monitoring
    verify_deployment
    show_final_info
    
    log_success "🎉 AuroraQ 배포 완료!"
}

# 사전 요구사항 설치
install_prerequisites() {
    log_header "사전 요구사항 설치"
    
    # 패키지 업데이트
    apt-get update -y
    
    # 필수 패키지 설치
    apt-get install -y curl wget git htop nano ufw fail2ban cron
    
    # Docker 설치
    if ! command -v docker &> /dev/null; then
        log_info "Docker 설치 중..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        systemctl enable docker
        systemctl start docker
        usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    # Docker Compose 설치
    if ! command -v docker-compose &> /dev/null; then
        log_info "Docker Compose 설치 중..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    log_success "사전 요구사항 설치 완료"
}

# 방화벽 설정
setup_firewall() {
    log_header "방화벽 설정"
    
    # UFW 설정
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 8004/tcp comment 'AuroraQ API'
    ufw allow 80/tcp comment 'HTTP'
    ufw allow 443/tcp comment 'HTTPS'
    ufw --force enable
    
    # Fail2ban 시작
    systemctl enable fail2ban
    systemctl start fail2ban
    
    log_success "방화벽 설정 완료"
}

# 프로젝트 구조 생성
create_project_structure() {
    log_header "프로젝트 구조 생성"
    
    mkdir -p $INSTALL_DIR/{logs,data,backups,config}
    mkdir -p $INSTALL_DIR/logs/{api,trading,system}
    mkdir -p $INSTALL_DIR/data/{cache,reports}
    
    chown -R $USER:$USER $INSTALL_DIR
    chmod -R 755 $INSTALL_DIR
    
    log_success "프로젝트 구조 생성 완료"
}

# Docker 파일 생성
create_docker_files() {
    log_header "Docker 설정 파일 생성"
    
    cd $INSTALL_DIR
    
    # Dockerfile 생성
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim-bullseye

# 환경 변수
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Seoul \
    API_PORT=8004

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc curl wget htop procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 디렉토리 생성
RUN mkdir -p /app/logs /app/data

# 포트 노출
EXPOSE 8004

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# 실행 명령
CMD ["python", "app.py"]
EOF

    # requirements.txt 생성
    cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
psutil==5.9.6
aiohttp==3.9.1
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
EOF

    # 메인 애플리케이션 파일 생성
    cat > app.py << 'EOF'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import psutil
import logging
from datetime import datetime
from typing import Dict, Any
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="AuroraQ Trading System API",
    description="VPS 최적화된 트레이딩 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "AuroraQ Trading System",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "memory_percent": memory.percent,
                "memory_used_mb": round(memory.used / 1024 / 1024, 2),
                "cpu_percent": cpu_percent
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/api/system/stats")
async def system_stats():
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "memory": {
                "percent": memory.percent,
                "used_mb": round(memory.used / 1024 / 1024, 2),
                "available_mb": round(memory.available / 1024 / 1024, 2)
            },
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "disk": {
                "percent": round((disk.used / disk.total) * 100, 2),
                "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk.free / 1024 / 1024 / 1024, 2)
            },
            "health_score": 100 if memory.percent < 80 and cpu_percent < 80 else 50
        }
    except Exception as e:
        logger.error(f"System stats failed: {e}")
        raise HTTPException(status_code=500, detail="System stats failed")

@app.get("/api/trading/status")
async def trading_status():
    return {
        "trading_active": True,
        "mode": "live",
        "positions_count": 0,
        "daily_pnl": 0.0,
        "last_update": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # 로그 디렉토리 생성
    os.makedirs("/app/logs", exist_ok=True)
    os.makedirs("/app/data", exist_ok=True)
    
    logger.info("Starting AuroraQ Trading System...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8004, 
        log_level="info",
        access_log=True
    )
EOF

    # Docker Compose 파일 생성
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  auroraQ:
    build: .
    container_name: auroraQ-trading
    restart: unless-stopped
    ports:
      - "0.0.0.0:8004:8004"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    environment:
      - TZ=Asia/Seoul
      - API_PORT=8004
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8004/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  redis:
    image: redis:7-alpine
    container_name: auroraQ-redis
    restart: unless-stopped
    ports:
      - "127.0.0.1:6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'

volumes:
  redis-data:
EOF

    # 환경 설정 파일
    cat > .env << 'EOF'
# AuroraQ VPS Configuration
API_PORT=8004
TZ=Asia/Seoul
LOG_LEVEL=INFO

# 보안 설정 (실제 값으로 변경 필요)
JWT_SECRET=your_jwt_secret_here
API_KEY=your_api_key_here

# 데이터베이스
REDIS_URL=redis://redis:6379/0
EOF

    log_success "Docker 설정 파일 생성 완료"
}

# 컨테이너 배포
deploy_containers() {
    log_header "Docker 컨테이너 배포"
    
    cd $INSTALL_DIR
    
    # 기존 컨테이너 정리
    docker-compose down 2>/dev/null || true
    docker system prune -f
    
    # 이미지 빌드 및 컨테이너 시작
    docker-compose build --no-cache
    docker-compose up -d
    
    # 컨테이너 시작 대기
    sleep 20
    
    log_success "Docker 컨테이너 배포 완료"
}

# 모니터링 설정
setup_monitoring() {
    log_header "모니터링 시스템 설정"
    
    # 백업 스크립트
    cat > $INSTALL_DIR/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/auroraQ/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

tar -czf "$BACKUP_DIR/logs.tar.gz" -C /opt/auroraQ logs/ 2>/dev/null || true
tar -czf "$BACKUP_DIR/data.tar.gz" -C /opt/auroraQ data/ 2>/dev/null || true
cp /opt/auroraQ/.env "$BACKUP_DIR/" 2>/dev/null || true

find /opt/auroraQ/backups/ -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
echo "백업 완료: $BACKUP_DIR"
EOF

    # 헬스 모니터링 스크립트
    cat > $INSTALL_DIR/health_monitor.sh << 'EOF'
#!/bin/bash
if ! curl -f http://localhost:8004/health &>/dev/null; then
    echo "$(date): API 서버 다운 - 재시작 시도" >> /opt/auroraQ/logs/alerts.log
    cd /opt/auroraQ && docker-compose restart auroraQ
fi
EOF

    chmod +x $INSTALL_DIR/{backup.sh,health_monitor.sh}
    
    # Cron 작업 설정
    (crontab -l 2>/dev/null; echo "0 2 * * * $INSTALL_DIR/backup.sh") | crontab -
    (crontab -l 2>/dev/null; echo "*/5 * * * * $INSTALL_DIR/health_monitor.sh") | crontab -
    
    # SystemD 서비스
    cat > /etc/systemd/system/auroraQ.service << EOF
[Unit]
Description=AuroraQ Trading System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable auroraQ.service
    
    log_success "모니터링 시스템 설정 완료"
}

# 배포 검증
verify_deployment() {
    log_header "배포 검증"
    
    cd $INSTALL_DIR
    
    # 컨테이너 상태 확인
    docker-compose ps
    
    # 헬스체크
    for i in {1..12}; do
        if curl -f http://localhost:8004/health 2>/dev/null; then
            log_success "API 서버 정상 동작 확인"
            break
        fi
        if [[ $i -eq 12 ]]; then
            log_error "API 서버 헬스체크 실패"
            docker-compose logs auroraQ
            return 1
        fi
        sleep 10
    done
    
    # 시스템 리소스 확인
    echo "=== 시스템 리소스 ==="
    free -h
    df -h
    
    log_success "배포 검증 완료"
}

# 최종 정보 출력
show_final_info() {
    log_header "🎉 배포 완료!"
    
    echo ""
    echo "📊 배포 정보:"
    echo "  ├─ 서버 IP: $VPS_HOST"
    echo "  ├─ 설치 경로: $INSTALL_DIR"
    echo "  └─ 배포 시간: $(date)"
    echo ""
    echo "🌐 서비스 접속:"
    echo "  ├─ API 서버: http://$VPS_HOST:8004"
    echo "  ├─ 헬스체크: http://$VPS_HOST:8004/health"
    echo "  ├─ 시스템 상태: http://$VPS_HOST:8004/api/system/stats"
    echo "  └─ API 문서: http://$VPS_HOST:8004/docs"
    echo ""
    echo "🔧 관리 명령어:"
    echo "  ├─ 상태 확인: cd $INSTALL_DIR && docker-compose ps"
    echo "  ├─ 로그 확인: cd $INSTALL_DIR && docker-compose logs -f"
    echo "  ├─ 재시작: cd $INSTALL_DIR && docker-compose restart"
    echo "  ├─ 백업: $INSTALL_DIR/backup.sh"
    echo "  └─ 서비스: systemctl {start|stop|restart} auroraQ"
    echo ""
    echo "⚠️  다음 단계:"
    echo "  1. .env 파일에서 보안 키 설정"
    echo "  2. SSL 인증서 설정 (선택사항)"
    echo "  3. 모니터링 대시보드 구성"
    echo ""
}

# 메인 함수 실행
main "$@"