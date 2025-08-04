# AuroraQ 통합 모니터링 대시보드 가이드

## 🎯 개요

기존 AuroraQ 시스템과 새로 개발된 예방적 관리 모니터링 시스템이 완전히 통합된 웹 기반 대시보드입니다.

## 🚀 빠른 시작

### Windows 환경
```bash
# 배치 파일로 실행 (권장)
start_integrated_dashboard.bat

# 또는 직접 실행
python integrated_monitoring_system.py
```

### Linux/macOS 환경
```bash
# Python으로 직접 실행
python3 integrated_monitoring_system.py

# 백그라운드 실행
nohup python3 integrated_monitoring_system.py &
```

## 🌐 접속 정보

- **웹 대시보드**: http://localhost:8080
- **API 문서**: http://localhost:8080/docs
- **WebSocket**: ws://localhost:8080/ws/integrated

## 📊 대시보드 구성

### 1. 📊 Overview 탭
**전체 시스템 종합 현황**
- 시스템 상태 (HEALTHY/WARNING/CRITICAL)
- 현재 거래 모드 (Paper/Live/Backtest/Dry Run)
- 예방 시스템 활성 상태
- 자동 복구 시스템 준비 상태
- 데이터 품질 지표 (목표: 80% 이상)
- 활성 알림 개수

### 2. 💰 Trading 탭
**거래 시스템 관리**
- **모드 전환 버튼**: 실시간 거래 모드 변경
  - 📝 Paper Trading (모의 거래)
  - 🟢 Live Trading (실거래)  
  - 📊 Backtest (백테스팅)
  - 🧪 Dry Run (테스트)
- **거래 현황**: 현재 모드, 연결 상태, 성과 지표

### 3. 🛡️ Prevention 탭
**예방적 장애 관리**
- 활성 위험 요소 개수
- 실행된 예방 조치 수
- 예상 비용 절감 효과
- 예방 조치 성공률
- 고위험 컴포넌트 목록

### 4. 🔧 Recovery 탭
**자동화된 복구 시스템**
- 현재 실행 중인 복구 작업
- 복구 성공률
- 평균 복구 시간
- 장애 예측 정확도
- 학습된 장애 패턴 수

### 5. 📈 Quality 탭
**데이터 품질 관리**
- 실시간 품질 메트릭
- 품질 개선 조치 내역
- 품질 트렌드 차트 (향후 구현)

### 6. 🚨 Alerts 탭
**알림 관리**
- 활성 알림 목록
- 알림 심각도별 분류
- 알림 해결 기능

## 🔧 주요 기능

### 실시간 모니터링
- **5초 간격 자동 업데이트**
- **WebSocket 실시간 통신**
- **자동 재연결 기능**

### 거래 모드 전환
```javascript
// 거래 모드 전환 (JavaScript)
await switchTradingMode('live');  // Paper → Live 전환
await switchTradingMode('paper'); // Live → Paper 전환
```

### API 엔드포인트

#### 시스템 상태
```bash
# 통합 시스템 상태
GET /api/integrated/status

# 거래 시스템 상태
GET /api/trading/status

# 예방 시스템 상태
GET /api/prevention/status

# 복구 시스템 상태
GET /api/recovery/status
```

#### 시스템 제어
```bash
# 거래 모드 전환
POST /api/trading/switch_mode
Content-Type: application/json
{"mode": "live"}

# 알림 해결
POST /api/alerts/{alert_id}/resolve
```

## 🔄 기존 시스템과의 통합

### 통합 아키텍처
```
┌─────────────────────────────────────────────────┐
│           통합 웹 대시보드 (Port 8080)              │
│  📊 Overview | 💰 Trading | 🛡️ Prevention        │
│  🔧 Recovery | 📈 Quality | 🚨 Alerts            │
├─────────────────────────────────────────────────┤
│                시스템 브리지                        │
├─────────────────────────────────────────────────┤
│  기존 AuroraQ 시스템    │   새로운 예방적 관리 시스템   │
│  ─────────────────    │   ─────────────────    │
│  • 터미널 대시보드        │   • 폴백 매니저            │
│  • 거래 엔진             │   • 품질 최적화기          │
│  • 센티멘트 분석         │   • 자동 복구 시스템        │
│  • 성능 최적화          │   • 예방적 장애 관리        │
└─────────────────────────────────────────────────┘
```

### 데이터 흐름
1. **기존 시스템**: 터미널 대시보드로 실시간 모니터링
2. **시스템 브리지**: 기존 컴포넌트 메트릭 수집
3. **통합 대시보드**: 웹 기반 통합 인터페이스 제공
4. **예방적 시스템**: 지능형 모니터링 및 자동 대응

## 🎛️ 설정 및 커스터마이징

### 포트 변경
```python
# integrated_monitoring_system.py 파일 수정
system.run_server(host="0.0.0.0", port=9000)  # 포트 변경
```

### 업데이트 간격 조정
```python
# 메트릭 업데이트 간격 (기본: 10초)
setInterval(() => this.fetchAllData(), 5000); // 5초로 변경
```

### 알림 임계값 설정
```python
# 시스템별 알림 임계값 조정
thresholds = {
    "fallback_rate_warning": 0.6,    # 60% 이상시 경고
    "quality_warning": 0.75,         # 75% 미만시 경고
    "cpu_usage_warning": 80,         # 80% 이상시 경고
}
```

## 🔒 보안 설정

### 접근 제한
```python
# 특정 IP만 접근 허용
allowed_ips = ["127.0.0.1", "192.168.1.0/24"]

# HTTPS 설정 (프로덕션 환경)
uvicorn.run(app, host="0.0.0.0", port=8080, 
           ssl_keyfile="key.pem", ssl_certfile="cert.pem")
```

### API 인증 (선택사항)
```python
# JWT 토큰 기반 인증
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.get("/api/protected")
async def protected_endpoint(token: str = Depends(security)):
    # 토큰 검증 로직
    pass
```

## 🚨 문제 해결

### 연결 문제
```bash
# 포트 사용 확인
netstat -an | findstr :8080

# 방화벽 설정 확인
# Windows: Windows Defender 방화벽에서 8080 포트 허용
# Linux: sudo ufw allow 8080
```

### 모듈 누락 오류
```bash
# 필수 패키지 설치
pip install fastapi uvicorn websockets psutil scikit-learn numpy

# 가상환경 사용 (권장)
python -m venv auroaq_env
auroaq_env\Scripts\activate  # Windows
source auroaq_env/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 성능 최적화
```python
# 워커 프로세스 수 증가 (멀티코어 활용)
uvicorn.run(app, host="0.0.0.0", port=8080, 
           workers=4, worker_class="uvicorn.workers.UvicornWorker")

# 메모리 사용량 최적화
import gc
gc.collect()  # 주기적 가비지 컬렉션
```

## 📱 모바일 접근

### 반응형 디자인
- **자동 레이아웃 조정**: 화면 크기에 따른 자동 적응
- **터치 최적화**: 모바일 터치 인터페이스 지원
- **빠른 로딩**: 경량화된 리소스로 빠른 로딩

### 모바일 브라우저 접근
```
스마트폰/태블릿에서 접근:
http://[서버IP]:8080

예시: http://192.168.1.100:8080
```

## 🔄 업데이트 및 유지보수

### 자동 업데이트 확인
```bash
# Git으로 최신 버전 확인
git pull origin main

# 패키지 업데이트
pip install --upgrade -r requirements.txt
```

### 로그 모니터링
```bash
# 로그 파일 위치
tail -f logs/integrated_dashboard.log

# 실시간 로그 확인
python integrated_monitoring_system.py --verbose
```

### 백업 및 복원
```bash
# 설정 백업
cp -r config/ config_backup_$(date +%Y%m%d)/

# 메트릭 데이터 백업
cp metrics_cache.json metrics_backup_$(date +%Y%m%d).json
```

## 🎯 주요 장점

### 1. 완전 통합
- ✅ 기존 터미널 대시보드와 병행 사용 가능
- ✅ 모든 AuroraQ 컴포넌트 통합 모니터링
- ✅ 단일 웹 인터페이스로 모든 제어

### 2. 실시간 모니터링
- ⚡ 5초 간격 실시간 업데이트
- 🔄 WebSocket 기반 즉시 알림
- 📊 실시간 메트릭 시각화

### 3. 지능형 관리
- 🛡️ 예방적 장애 관리
- 🤖 자동화된 복구
- 📈 예측적 품질 최적화

### 4. 사용자 친화적
- 🌐 웹 브라우저에서 접근
- 📱 모바일 반응형 디자인
- 🎨 직관적인 UI/UX

## 📞 지원 및 문의

### 기술 문의
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **로그 분석**: 상세 로그를 통한 문제 진단
- **커뮤니티**: 사용자 커뮤니티를 통한 팁 공유

---

**통합 대시보드**: 기존 AuroraQ의 강력함 + 새로운 예방적 관리의 지능형 = **차세대 통합 모니터링 솔루션** 🚀