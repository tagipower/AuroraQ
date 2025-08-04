# VPS AuroraQ Trading System - Production Dockerfile
FROM python:3.11-slim-bullseye

# 메타데이터
LABEL maintainer="AuroraQ Trading System"
LABEL version="1.0.0"
LABEL description="VPS Optimized Trading System with Advanced Monitoring & Security"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONPATH=/app \
    TRADING_MODE=live \
    VPS_MEMORY_LIMIT=3G \
    ENABLE_UNIFIED_LOGGING=true \
    API_HOST=0.0.0.0 \
    API_PORT=8004

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    # 빌드 도구
    gcc \
    g++ \
    make \
    # 시스템 도구
    curl \
    wget \
    git \
    htop \
    nano \
    # 네트워크 도구
    netcat-openbsd \
    telnet \
    # 프로세스 모니터링
    procps \
    # 타임존 설정
    tzdata \
    # SSL/TLS 지원
    ca-certificates \
    # 정리
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 작업 디렉토리 생성
WORKDIR /app

# 보안을 위한 시스템 사용자 생성
RUN useradd --create-home --shell /bin/bash --uid 1000 aurorauser && \
    chown -R aurorauser:aurorauser /app

# Python 의존성 파일 복사 및 설치 (멀티스테이지 최적화)
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 디렉토리 구조 생성 및 권한 설정
RUN mkdir -p \
    /app/logs/{raw,summary,training,tagged,debug,api,error,security,monitoring} \
    /app/data/{cache,backups,configs,reports} \
    /app/run \
    && chown -R aurorauser:aurorauser /app \
    && chmod +x /app/*.py \
    && chmod +x /app/scripts/*.sh 2>/dev/null || true

# 헬스체크 스크립트 생성
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
# 종합 헬스체크 스크립트
set -e

echo "🔍 AuroraQ 시스템 헬스체크 시작..."

# API 서버 상태 확인
if curl -f --connect-timeout 5 --max-time 10 http://localhost:8004/health > /dev/null 2>&1; then
    echo "✅ API 서버 정상"
else
    echo "❌ API 서버 응답 없음"
    exit 1
fi

# 메모리 사용량 확인 (3GB 제한)
MEMORY_MB=$(python3 -c "import psutil; print(int(psutil.Process().memory_info().rss / 1024 / 1024))")
if [ $MEMORY_MB -gt 3072 ]; then
    echo "⚠️ 메모리 사용량 초과: ${MEMORY_MB}MB > 3072MB"
    exit 1
else
    echo "✅ 메모리 사용량 정상: ${MEMORY_MB}MB"
fi

# 핵심 프로세스 확인
if pgrep -f "python.*api_system.py" > /dev/null; then
    echo "✅ API 시스템 프로세스 정상"
else
    echo "❌ API 시스템 프로세스 없음"
    exit 1
fi

echo "🎉 헬스체크 완료 - 모든 시스템 정상"
EOF

# 시작 스크립트 생성
RUN cat > /app/start_system.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 AuroraQ VPS 고도화 시스템 시작..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 시스템 정보:"
echo "   - Trading Mode: ${TRADING_MODE:-live}"
echo "   - Memory Limit: ${VPS_MEMORY_LIMIT:-3G}"
echo "   - API Port: ${API_PORT:-8004}"
echo "   - Time Zone: ${TZ:-Asia/Seoul}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 디렉토리 권한 재확인
chmod -R 755 /app/logs /app/data

# 시스템 검증 실행
echo "🔍 시스템 사전 검증 중..."
python system_validator.py --quick-check || {
    echo "⚠️ 시스템 검증에서 문제 발견됨 - 계속 진행합니다"
}

# 성능 최적화 실행
echo "⚡ 성능 최적화 실행 중..."
python -c "
from performance_optimizer import global_optimizer
result = global_optimizer.auto_optimize()
print(f'✅ 최적화 완료: {result}')
" || echo "⚠️ 성능 최적화 실패 - 기본 설정으로 진행"

# 모니터링 시스템 시작
echo "📊 모니터링 시스템 시작..."
python -c "
from monitoring_alert_system import start_monitoring
start_monitoring()
print('✅ 모니터링 시스템 시작됨')
" &

# 보안 시스템 초기화
echo "🔐 보안 시스템 초기화..."
python -c "
from security_system import global_security
scan_result = global_security.run_security_scan()
print(f'✅ 보안 점수: {scan_result[\"security_score\"]}/100')
" || echo "⚠️ 보안 검사 실패"

# API 서버 시작
echo "🌐 API 서버 시작..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
exec python api_system.py --mode server --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8004}
EOF

# 긴급 복구 스크립트 생성
RUN cat > /app/emergency_recovery.sh << 'EOF'
#!/bin/bash
echo "🚨 긴급 복구 모드 실행..."

# 메모리 정리
python -c "
from performance_optimizer import global_optimizer
result = global_optimizer.memory_manager.emergency_cleanup()
print(f'메모리 정리 완료: {result}')
"

# 에러 덤프 생성
python -c "
from debug_system import global_debugger
dump_file = global_debugger.emergency_debug_dump()
print(f'디버그 덤프 생성: {dump_file}')
"

echo "✅ 긴급 복구 완료"
EOF

# 모든 스크립트 실행 권한 부여
RUN chmod +x /app/*.sh

# 포트 노출
EXPOSE 8004 8003

# 볼륨 마운트 포인트
VOLUME ["/app/logs", "/app/data"]

# 사용자 전환
USER aurorauser

# 헬스체크 설정 (30초 간격, 10초 타임아웃, 3회 재시도)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# 기본 실행 명령
CMD ["/app/start_system.sh"]