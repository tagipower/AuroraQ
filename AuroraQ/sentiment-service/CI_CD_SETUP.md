# CI/CD Setup Guide

## GitHub Actions 설정 가이드

### 1. GitHub Secrets 설정

GitHub 저장소의 Settings → Secrets and variables → Actions에서 다음 시크릿을 추가하세요:

#### 필수 시크릿
- `VPS_SSH_KEY`: VPS 접속용 SSH 개인키
  ```bash
  # 로컬에서 SSH 키 내용 복사
  cat ~/.ssh/aurora_vps_key
  ```

#### 선택적 시크릿 (알림용)
- `TELEGRAM_BOT_TOKEN`: Telegram 봇 토큰
- `TELEGRAM_CHAT_ID`: Telegram 채팅 ID

### 2. 워크플로우 활성화

`.github/workflows/sentiment-service-deploy.yml` 파일이 main 브랜치에 푸시되면 자동으로 활성화됩니다.

### 3. 배포 트리거

다음 경우에 자동 배포가 실행됩니다:
- `main` 브랜치에 푸시
- `sentiment-service/` 디렉토리 내 파일 변경
- 수동 트리거 (Actions 탭에서 "Run workflow")

### 4. 배포 프로세스

1. **테스트 단계**
   - Python 3.11 환경 설정
   - 의존성 설치
   - pytest 실행
   - 코드 커버리지 리포트

2. **빌드 단계**
   - Docker 이미지 빌드
   - 이미지 태깅 (commit SHA)
   - 아티팩트로 저장

3. **배포 단계**
   - VPS에 SSH 연결
   - 백업 생성
   - 새 이미지 배포
   - Health check
   - 알림 발송

### 5. 로컬 배포 (수동)

GitHub Actions 없이 로컬에서 배포:

```bash
cd sentiment-service
./scripts/deploy.sh deploy
```

### 6. 롤백 절차

문제 발생 시 이전 버전으로 롤백:

```bash
# VPS에 SSH 접속
ssh aurora-vps

# 백업 목록 확인
ls -la /opt/backups/

# 특정 백업으로 복원
cd /opt
tar -xzf /opt/backups/aurora-sentiment-20250729_123456.tar.gz

# 서비스 재시작
cd /opt/aurora-sentiment
docker compose down
docker compose up -d
```

### 7. 모니터링

배포 상태 확인:
- GitHub Actions 탭에서 실시간 로그 확인
- Telegram 알림 확인
- VPS에서 `aurora-status` 명령 실행

### 8. 트러블슈팅

#### SSH 연결 실패
- VPS_SSH_KEY 시크릿이 올바른지 확인
- SSH 키에 개행문자가 포함되어 있는지 확인

#### Docker 빌드 실패
- requirements.txt 구문 오류 확인
- Dockerfile 경로 확인

#### Health check 실패
- 서비스 포트가 올바른지 확인
- 환경 변수 설정 확인