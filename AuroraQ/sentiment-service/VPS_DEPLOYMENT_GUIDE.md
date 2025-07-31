# AuroraQ Sentiment Service VPS 배포 가이드

## 📋 개요

AuroraQ Sentiment Service를 싱가포르 VPS (IP: 109.123.239.30, Ubuntu 22.04)에 상시 운영하기 위한 완전 자동화 배포 가이드입니다.

## 🎯 주요 기능

### 핵심 서비스
- **실시간 감정 분석**: 키워드 기반 0.5초 이내 응답
- **배치 FinBERT 분석**: 15분 간격 고정밀 분석
- **빅 이벤트 감지**: 11가지 시장 이벤트 자동 감지
- **다중 데이터 소스**: Google News, Yahoo Finance, NewsAPI, Finnhub, Reddit
- **매매 신호 생성**: 실전/가상 매매용 신호 자동 생성
- **AuroraQ 통합**: 메인 트레이딩 시스템과 완전 연동
- **🆕 텔레그램 알림**: 실시간 매매 신호, 빅 이벤트, 시스템 상태 알림

### VPS 최적화 설정
- **리소스 효율성**: CPU 전용, 메모리 1.5GB 제한
- **배치 크기**: FinBERT 8개 문장 동시 처리
- **자동 스케줄링**: 7개 백그라운드 작업 자동 실행
- **24/7 모니터링**: 5분마다 헬스체크, 자동 복구

## 🚀 빠른 배포

### 1단계: 로컬에서 배포 스크립트 실행

```bash
# 프로젝트 디렉토리로 이동
cd sentiment-service

# 배포 스크립트 실행 권한 부여
chmod +x scripts/deploy_vps.sh

# VPS 배포 시작 (자동화)
./scripts/deploy_vps.sh

# 또는 수동으로 VPS 사용자 지정
VPS_USER=ubuntu ./scripts/deploy_vps.sh
```

### 2단계: API 키 설정 (필수)

VPS에 배포 후 환경변수 파일 편집:

```bash
# VPS 접속
ssh root@109.123.239.30

# 환경변수 파일 편집
cd /opt/aurora-sentiment
nano .env

# 다음 API 키들을 실제 값으로 교체:
GOOGLE_NEWS_API_KEY=your_actual_key_here
NEWSAPI_KEY=your_actual_key_here
FINNHUB_API_KEY=your_actual_key_here
REDDIT_CLIENT_ID=your_actual_key_here
REDDIT_CLIENT_SECRET=your_actual_key_here

# 텔레그램 알림 설정 (새로 추가!)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID_GENERAL=your_chat_id_here
TELEGRAM_CHAT_ID_TRADING=your_chat_id_here
TELEGRAM_CHAT_ID_EVENTS=your_chat_id_here
TELEGRAM_CHAT_ID_SYSTEM=your_chat_id_here
TELEGRAM_ENABLED=true

# ... 기타 API 키들

# 서비스 재시작
systemctl restart aurora-sentiment
```

### 3단계: 텔레그램 알림 설정 (새로 추가! 🆕)

```bash
# VPS에서 텔레그램 알림 테스트
cd /opt/aurora-sentiment
python3 scripts/test_telegram.py

# 텔레그램 설정이 완료되면 서비스 재시작
systemctl restart aurora-sentiment
```

### 4단계: 자동 모니터링 설정

```bash
# VPS에서 실행
cd /opt/aurora-sentiment
chmod +x scripts/*.sh

# Cron Job 및 모니터링 설정
./scripts/setup_cron.sh install

# 모니터링 상태 확인
./scripts/monitor_service.sh health
```

## 📊 서비스 접속 정보

### 메인 엔드포인트
- **API 서버**: http://109.123.239.30:8000
- **API 문서**: http://109.123.239.30:8000/docs  
- **헬스체크**: http://109.123.239.30:8000/health

### 모니터링 대시보드
- **Prometheus**: http://109.123.239.30:9090
- **시스템 메트릭**: http://109.123.239.30:8000/metrics

### 주요 API 엔드포인트
```
# 실시간 감정 분석
POST /api/v1/sentiment/analyze/realtime

# 배치 스케줄러 상태
GET /api/v1/scheduler/stats

# 빅 이벤트 감지
GET /api/v1/events/active

# 매매 신호 생성
POST /api/v1/trading/signal/generate

# 융합 감정 분석
POST /api/v1/fusion/analyze
```

## 🛠 관리 명령어

### 서비스 관리
```bash
# 서비스 상태 확인
systemctl status aurora-sentiment

# 서비스 재시작
systemctl restart aurora-sentiment

# 서비스 중지
systemctl stop aurora-sentiment

# 서비스 로그 확인
journalctl -u aurora-sentiment -f
```

### Docker 관리
```bash
cd /opt/aurora-sentiment

# 컨테이너 상태 확인
docker compose ps

# 실시간 로그 확인
docker compose logs -f sentiment-service

# 특정 컨테이너 재시작
docker compose restart sentiment-service

# 전체 재빌드
docker compose build --no-cache
docker compose up -d
```

### 모니터링 및 진단
```bash
# 전체 헬스체크
./scripts/monitor_service.sh health

# 상세 상태 보고서
./scripts/monitor_service.sh status

# 지속적 모니터링 (백그라운드)
nohup ./scripts/monitor_service.sh monitor &

# 자동 복구 실행
./scripts/monitor_service.sh recovery
```

## 📈 자동화된 작업 스케줄

### Cron Jobs (자동 실행)
- **헬스체크**: 5분마다 (*/5 * * * *)
- **뉴스 수집**: 5분마다 (배치 스케줄러)
- **FinBERT 분석**: 15분마다 (배치 스케줄러)
- **이벤트 감지**: 10분마다 (배치 스케줄러)
- **실전 매매 신호**: 3분마다 (배치 스케줄러)
- **가상 매매 신호**: 2분마다 (배치 스케줄러)
- **캐시 정리**: 30분마다 (배치 스케줄러)
- **시스템 유지보수**: 매일 새벽 2시

### 로그 관리
- **로그 로테이션**: 매일 새벽 3시
- **백업**: 주 1회 (일요일 새벽 1시)
- **디스크 정리**: 매일 새벽 2시
- **API 키 검증**: 매일 새벽 5시

## 🔧 문제 해결

### 일반적인 문제들

#### 1. API 응답 없음
```bash
# 서비스 상태 확인
./scripts/monitor_service.sh health

# 로그 확인
docker compose logs sentiment-service | tail -50

# 자동 복구 시도
./scripts/monitor_service.sh recovery
```

#### 2. 높은 메모리 사용률
```bash
# 리소스 사용량 확인
docker stats

# 메모리 정리
docker system prune -f
./scripts/monitor_service.sh cleanup

# 배치 크기 조정 (.env 파일)
FINBERT_BATCH_SIZE=4  # 기본값 8에서 줄임
```

#### 3. API 키 오류
```bash
# API 키 유효성 검사
./scripts/validate_api_keys.sh

# 환경변수 파일 확인
cat .env | grep -E "(API_KEY|CLIENT_ID|CLIENT_SECRET)"

# 서비스 재시작 (환경변수 다시 로드)
systemctl restart aurora-sentiment
```

#### 4. 디스크 공간 부족
```bash
# 디스크 사용량 확인
df -h

# 자동 정리
./scripts/monitor_service.sh cleanup

# 수동 정리
docker system prune -f --volumes
find /var/log -name "*.log" -type f -mtime +7 -delete
```

### 응급 복구

```bash
# 1. 전체 서비스 재시작
systemctl restart aurora-sentiment

# 2. Docker 강제 재시작
cd /opt/aurora-sentiment
docker compose down --remove-orphans
docker compose up -d

# 3. 백업에서 복구
ls /opt/aurora-sentiment-backups/
# 최신 백업 선택하여 복구

# 4. 완전 재배포 (최후 수단)
./scripts/deploy_vps.sh
```

## 📊 성능 모니터링

### 주요 메트릭
- **API 응답 시간**: < 0.5초 (실시간 분석)
- **메모리 사용량**: < 1.5GB
- **CPU 사용률**: < 80%
- **디스크 사용량**: < 85%

### 성능 최적화 팁
1. **배치 크기 조정**: `FINBERT_BATCH_SIZE=4-16`
2. **워커 수 조정**: `MAX_WORKERS=1-4`
3. **캐시 TTL 조정**: `CACHE_TTL=300-900`
4. **뉴스 수집량 조정**: `NEWS_MAX_ARTICLES=20-50`

## 🔒 보안 설정

### 방화벽 (자동 설정됨)
```bash
# 허용된 포트 확인
ufw status

# 필요시 포트 추가
ufw allow 8000/tcp
```

### SSL 인증서 (선택적)
```bash
# Let's Encrypt 설치
apt install certbot python3-certbot-nginx

# SSL 인증서 발급
certbot --nginx -d 109.123.239.30

# 자동 갱신 확인 (cron job에 포함됨)
certbot renew --dry-run
```

## 📞 지원 및 연락처

### 로그 위치
- **애플리케이션**: `/opt/aurora-sentiment/logs/`
- **시스템**: `/var/log/aurora-sentiment/`
- **Docker**: `docker compose logs`

### 유용한 명령어 참조
```bash
# 실시간 API 테스트
curl http://109.123.239.30:8000/health

# 배치 스케줄러 상태
curl http://109.123.239.30:8000/api/v1/scheduler/stats | jq

# 감정 분석 테스트
curl -X POST http://109.123.239.30:8000/api/v1/sentiment/analyze/realtime \
  -H "Content-Type: application/json" \
  -d '{"text": "Bitcoin price surges after ETF approval"}'

# 시스템 리소스 확인
htop
free -h
df -h
```

---

## 🎉 배포 완료 체크리스트

- [ ] VPS 배포 스크립트 실행 완료
- [ ] API 키 설정 완료
- [ ] 헬스체크 통과 확인
- [ ] 자동 모니터링 설정 완료
- [ ] 백업 시스템 테스트 완료
- [ ] AuroraQ 메인 시스템과 연동 확인
- [ ] 실시간 분석 API 테스트 완료
- [ ] 배치 스케줄러 동작 확인
- [ ] 모니터링 대시보드 접속 확인

**축하합니다! AuroraQ Sentiment Service가 성공적으로 배포되었습니다.** 🚀