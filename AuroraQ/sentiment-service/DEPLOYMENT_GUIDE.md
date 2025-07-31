# AuroraQ Sentiment Service Deployment Guide

## 🚀 완전한 배포 및 모니터링 시스템

### 📋 개요

AuroraQ 센티먼트 서비스는 이제 완전한 24/7 모니터링 시스템과 자동화된 배포 파이프라인을 갖추고 있습니다.

### 🛠️ 설정된 구성 요소

#### 1. **24/7 모니터링 시스템**
- **Health Check**: 5분마다 자동 상태 확인
- **Auto-restart**: 서비스 실패 시 자동 재시작 (최대 3회 시도)
- **Daily Restart**: 매일 오전 3시 정기 재시작
- **Disk Monitoring**: 디스크 사용량 90% 초과 시 자동 정리

#### 2. **시스템 서비스**
- **Systemd Service**: `aurora-sentiment.service`
- **부팅 시 자동 시작**: 서버 재부팅 시 자동으로 서비스 시작
- **로그 로테이션**: 7일간 로그 보관 후 자동 삭제

#### 3. **자동 배포 파이프라인**
- **GitHub Actions**: 코드 푸시 시 자동 테스트 및 배포
- **로컬 배포 스크립트**: `scripts/deploy.sh`
- **백업 시스템**: 배포 전 자동 백업 (최근 5개 보관)

### 📌 주요 명령어

#### 서비스 상태 확인
```bash
# 전체 상태 확인
aurora-status

# 서비스 로그 확인
journalctl -u aurora-sentiment -f

# Docker 컨테이너 상태
docker ps

# Health 체크
curl http://localhost:8000/health
```

#### 서비스 관리
```bash
# 서비스 재시작
systemctl restart aurora-sentiment

# 서비스 중지
systemctl stop aurora-sentiment

# 서비스 시작
systemctl start aurora-sentiment

# 수동 health 체크 실행
/usr/local/bin/aurora-health-check
```

### 🔧 Cron 작업 스케줄

| 시간 | 작업 | 설명 |
|------|------|------|
| */5 * * * * | Health Check | 5분마다 서비스 상태 확인 |
| 0 3 * * * | Daily Restart | 매일 오전 3시 재시작 |
| 0 2 * * * | Disk Cleanup | 매일 오전 2시 디스크 정리 |

### 📱 알림 시스템

Telegram 봇을 통한 실시간 알림:
- ⚠️ 서비스 장애 감지
- ✅ 서비스 복구 성공
- ❌ 재시작 실패
- 💾 디스크 사용량 경고

### 🚀 배포 프로세스

#### 로컬 배포
```bash
cd sentiment-service
./scripts/deploy.sh deploy  # 전체 배포 (테스트 포함)
./scripts/deploy.sh quick   # 빠른 배포 (테스트 제외)
./scripts/deploy.sh test    # 테스트만 실행
./scripts/deploy.sh build   # Docker 이미지 빌드만
```

#### GitHub Actions 자동 배포
1. `main` 브랜치에 코드 푸시
2. 자동으로 테스트 실행
3. 테스트 통과 시 Docker 이미지 빌드
4. VPS에 자동 배포
5. Health check 및 결과 알림

### 📊 모니터링 대시보드

#### 터미널 대시보드 (로컬)
```bash
cd sentiment-service/dashboard
python terminal_dashboard.py --service http://109.123.239.30:8000
```

#### Prometheus 메트릭
- URL: http://109.123.239.30:9090
- 주요 메트릭:
  - API 응답 시간
  - 요청 처리량
  - 에러율
  - 시스템 리소스 사용량

### 🔒 보안 설정

- Docker 컨테이너는 non-root 사용자로 실행
- 환경 변수를 통한 민감한 정보 관리
- 방화벽 설정으로 필요한 포트만 개방
- SSH 키 기반 인증만 허용

### 📝 로그 파일 위치

- **서비스 로그**: `journalctl -u aurora-sentiment`
- **Health 모니터 로그**: `/var/log/aurora-sentiment-monitor.log`
- **Docker 로그**: `docker logs auroraQ-sentiment-service`

### 🛠️ 트러블슈팅

#### 서비스가 시작되지 않을 때
```bash
# 로그 확인
docker logs auroraQ-sentiment-service
journalctl -u aurora-sentiment -n 100

# 수동으로 컨테이너 시작
cd /opt/aurora-sentiment
docker compose up
```

#### 디스크 공간 부족
```bash
# 수동 정리 실행
/usr/local/bin/aurora-disk-check

# Docker 시스템 정리
docker system prune -af
```

### 📈 성능 최적화

- Redis 캐싱으로 응답 시간 단축
- 프로세스 워커 수 최적화 (VPS 리소스에 맞춤)
- 모델 사전 로딩으로 콜드 스타트 방지
- 배치 처리로 효율성 향상

### 🔄 업데이트 절차

1. 코드 변경사항 커밋 및 푸시
2. GitHub Actions가 자동으로 테스트 및 배포
3. 배포 중 기존 서비스는 계속 실행 (무중단 배포)
4. 배포 완료 후 자동 health check
5. 문제 발생 시 자동 롤백

---

**VPS 정보**
- IP: 109.123.239.30
- 위치: Singapore
- 서비스 포트: 8000 (API), 9090 (Prometheus), 6379 (Redis)

**지원**
- 문제 발생 시 `/var/log/aurora-sentiment-monitor.log` 확인
- Telegram 알림 채널 모니터링
- `aurora-status` 명령으로 전체 상태 확인