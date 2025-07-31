#!/bin/bash
# VPS 배포 스크립트 - AuroraQ Sentiment Service
# VPS IP: 109.123.239.30 (Singapore)

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 환경 변수
VPS_IP="109.123.239.30"
VPS_USER="root"
PROJECT_NAME="sentiment-service"
REMOTE_DIR="/opt/auroraQ-sentiment"
SERVICE_PORT="8000"

echo -e "${BLUE}=== AuroraQ Sentiment Service VPS 배포 ===${NC}"
echo -e "${YELLOW}VPS IP: ${VPS_IP}${NC}"
echo -e "${YELLOW}Port: ${SERVICE_PORT}${NC}"

# 1. VPS 연결 테스트
echo -e "\n${BLUE}1. VPS 연결 테스트...${NC}"
if ssh -o ConnectTimeout=10 ${VPS_USER}@${VPS_IP} 'echo "VPS 연결 성공"'; then
    echo -e "${GREEN}✅ VPS 연결 성공${NC}"
else
    echo -e "${RED}❌ VPS 연결 실패${NC}"
    exit 1
fi

# 2. Docker 설치 확인
echo -e "\n${BLUE}2. Docker 설치 확인...${NC}"
ssh ${VPS_USER}@${VPS_IP} '
    if ! command -v docker &> /dev/null; then
        echo "Docker 설치 중..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        systemctl start docker
        systemctl enable docker
        echo "✅ Docker 설치 완료"
    else
        echo "✅ Docker 이미 설치됨"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "Docker Compose 설치 중..."
        curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
        echo "✅ Docker Compose 설치 완료"
    else
        echo "✅ Docker Compose 이미 설치됨"
    fi
'

# 3. 프로젝트 디렉토리 생성
echo -e "\n${BLUE}3. 프로젝트 디렉토리 설정...${NC}"
ssh ${VPS_USER}@${VPS_IP} "
    mkdir -p ${REMOTE_DIR}
    mkdir -p ${REMOTE_DIR}/logs
    mkdir -p ${REMOTE_DIR}/data
    echo '✅ 디렉토리 생성 완료'
"

# 4. 프로젝트 파일 전송
echo -e "\n${BLUE}4. 프로젝트 파일 전송...${NC}"
rsync -avz --exclude='__pycache__' \
           --exclude='*.pyc' \
           --exclude='.git' \
           --exclude='logs/*' \
           --exclude='data/*' \
           --exclude='models/*' \
           --exclude='.pytest_cache' \
           ./ ${VPS_USER}@${VPS_IP}:${REMOTE_DIR}/

echo -e "${GREEN}✅ 파일 전송 완료${NC}"

# 5. 환경 설정 파일 생성
echo -e "\n${BLUE}5. 환경 설정 파일 생성...${NC}"
ssh ${VPS_USER}@${VPS_IP} "
cat > ${REMOTE_DIR}/.env << 'EOF'
# AuroraQ Sentiment Service Environment Configuration

# Service Configuration
APP_NAME=AuroraQ Sentiment Service
APP_VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000
MAX_WORKERS=2

# Redis Configuration
REDIS_URL=redis://redis:6379
REDIS_PASSWORD=
REDIS_DB=0
CACHE_TTL=300

# Model Configuration
FINBERT_MODEL_PATH=/app/models/finbert
FINBERT_MAX_LENGTH=512
FINBERT_BATCH_SIZE=16
MODEL_WARMUP=true
ENABLE_MODEL_CACHING=true

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_ACCESS_LOG=true

# Security Configuration
ALLOWED_HOSTS=*
CORS_ORIGINS=*
RATE_LIMIT_RPM=1000

# Monitoring Configuration
ENABLE_METRICS=true
PROMETHEUS_PORT=8080

# Keyword Scorer Configuration
KEYWORD_CACHE_SIZE=1000
REALTIME_TIMEOUT_MS=500

# VPS Specific
ENVIRONMENT=production
TZ=Asia/Singapore
EOF
echo '✅ 환경 설정 파일 생성 완료'
"

# 6. Docker 이미지 빌드 및 실행
echo -e "\n${BLUE}6. Docker 서비스 시작...${NC}"
ssh ${VPS_USER}@${VPS_IP} "
    cd ${REMOTE_DIR}
    
    # 기존 컨테이너 정리
    docker-compose down --remove-orphans 2>/dev/null || true
    docker system prune -f
    
    # 새로운 컨테이너 빌드 및 시작
    docker-compose build --no-cache
    docker-compose up -d
    
    echo '✅ Docker 서비스 시작 완료'
"

# 7. 서비스 상태 확인
echo -e "\n${BLUE}7. 서비스 상태 확인...${NC}"
sleep 30  # 서비스 시작 대기

# 헬스 체크
for i in {1..10}; do
    if curl -s http://${VPS_IP}:${SERVICE_PORT}/health > /dev/null; then
        echo -e "${GREEN}✅ 서비스 정상 실행 중${NC}"
        break
    else
        echo -e "${YELLOW}⏳ 서비스 시작 대기 중... (${i}/10)${NC}"
        sleep 10
    fi
done

# 8. 서비스 정보 출력
echo -e "\n${BLUE}8. 서비스 정보${NC}"
echo -e "${GREEN}🌐 서비스 URL: http://${VPS_IP}:${SERVICE_PORT}${NC}"
echo -e "${GREEN}📊 헬스 체크: http://${VPS_IP}:${SERVICE_PORT}/health${NC}"
echo -e "${GREEN}📖 API 문서: http://${VPS_IP}:${SERVICE_PORT}/docs${NC}"
echo -e "${GREEN}📈 메트릭: http://${VPS_IP}:9090${NC}"

# 9. 실시간 키워드 분석 테스트
echo -e "\n${BLUE}9. 실시간 분석 테스트...${NC}"
TEST_RESPONSE=$(curl -s -X POST "http://${VPS_IP}:${SERVICE_PORT}/sentiment/analyze/realtime" \
    -H "Content-Type: application/json" \
    -d '{
        "text": "Bitcoin surges as institutional adoption grows",
        "symbol": "BTC"
    }' || echo "테스트 실패")

if [[ $TEST_RESPONSE != "테스트 실패" ]]; then
    echo -e "${GREEN}✅ 실시간 분석 테스트 성공${NC}"
    echo -e "${BLUE}응답:${NC} $TEST_RESPONSE"
else
    echo -e "${RED}⚠️ 실시간 분석 테스트 실패${NC}"
fi

# 10. 서비스 로그 확인
echo -e "\n${BLUE}10. 서비스 로그 (최근 20줄)${NC}"
ssh ${VPS_USER}@${VPS_IP} "
    cd ${REMOTE_DIR}
    docker-compose logs --tail=20 sentiment-service
"

# 11. 자동 재시작 설정
echo -e "\n${BLUE}11. 자동 재시작 설정...${NC}"
ssh ${VPS_USER}@${VPS_IP} "
cat > /etc/systemd/system/auroraQ-sentiment.service << 'EOF'
[Unit]
Description=AuroraQ Sentiment Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${REMOTE_DIR}
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable auroraQ-sentiment
echo '✅ 자동 재시작 설정 완료'
"

echo -e "\n${GREEN}🎉 배포 완료! 🎉${NC}"
echo -e "${BLUE}서비스가 VPS에서 24/7 실행됩니다.${NC}"
echo -e "${YELLOW}주요 엔드포인트:${NC}"
echo -e "  • 실시간 분석: POST /sentiment/analyze/realtime"
echo -e "  • 배치 분석: POST /sentiment/analyze/batch/realtime"
echo -e "  • 헬스 체크: GET /health"
echo -e "  • 통계: GET /sentiment/stats"

echo -e "\n${YELLOW}모니터링 명령어:${NC}"
echo -e "  ssh ${VPS_USER}@${VPS_IP} 'cd ${REMOTE_DIR} && docker-compose logs -f'"
echo -e "  ssh ${VPS_USER}@${VPS_IP} 'cd ${REMOTE_DIR} && docker-compose ps'"