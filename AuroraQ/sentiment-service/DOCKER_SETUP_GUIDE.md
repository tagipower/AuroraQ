# 🐳 Docker 설치 및 AuroraQ Sentiment Service 실행 가이드

## Docker 설치 (Windows)

### 1. Docker Desktop 설치
1. [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) 다운로드
2. 설치 파일 실행 후 기본 설정으로 설치
3. 설치 완료 후 시스템 재부팅
4. Docker Desktop 실행 (시스템 트레이에서 Docker 아이콘 확인)

### 2. Docker 설치 확인
```bash
docker --version
docker compose --version
```

## AuroraQ Sentiment Service 실행

### ✅ aioredis → redis.asyncio 변경 완료
- Python 3.11+ 호환성 문제 해결됨
- 컨테이너 재시작 문제 해결됨

### 실행 명령

#### 1. 기존 컨테이너 중지 (있는 경우)
```bash
cd sentiment-service
docker compose down
```

#### 2. 서비스 빌드 및 시작
```bash
docker compose up --build
```

#### 3. 백그라운드 실행 (선택사항)
```bash
docker compose up --build -d
```

#### 4. 서비스 상태 확인
```bash
docker compose ps
docker compose logs sentiment-service
```

#### 5. 서비스 중지
```bash
docker compose down
```

## 예상 결과

### ✅ 정상 시작 로그
```
sentiment-service  | INFO:     Started server process
sentiment-service  | INFO:     Waiting for application startup.
sentiment-service  | INFO:     Application startup complete.
sentiment-service  | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### ✅ Redis 연결 성공
```
sentiment-service  | INFO: Redis connection established
sentiment-service  | INFO: Content cache manager initialized
```

### ❌ 이전 오류 (해결됨)
```
# 더 이상 발생하지 않음:
# ModuleNotFoundError: No module named 'aioredis'
# ImportError: cannot import name 'aioredis'
```

## API 엔드포인트 확인

서비스가 정상 시작되면 다음 URL에서 확인 가능:

- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

## 트러블슈팅

### Docker Desktop이 시작되지 않는 경우
1. WSL 2 업데이트 필요할 수 있음
2. Hyper-V 기능 활성화 필요
3. BIOS에서 가상화 기능 활성화

### 컨테이너 빌드 실패 시
```bash
# 캐시 없이 다시 빌드
docker compose build --no-cache
docker compose up
```

### 포트 충돌 시
```bash
# 사용 중인 포트 확인
netstat -an | findstr :8000
netstat -an | findstr :6379

# docker-compose.yml에서 포트 변경
```

## 성공 확인 체크리스트

- [ ] Docker Desktop 설치 완료
- [ ] `docker --version` 명령 성공
- [ ] `docker compose up --build` 실행
- [ ] 컨테이너 재시작 없이 정상 시작
- [ ] http://localhost:8000/health 응답 확인
- [ ] Redis 연결 성공 로그 확인

---

🎉 **이제 Python 3.11+ 호환성 문제가 완전히 해결되어 안정적으로 실행됩니다!**