# AuroraQ Sentiment Dashboard

AuroraQ 감정 분석 서비스를 위한 실시간 모니터링 대시보드입니다.

## 기능

### Phase 1 핵심 기능
- **실시간 감정 점수 패널**: BTC/ETH/CRYPTO 심볼별 0-1 범위 점수, 신뢰도, 추세 속도
- **뉴스 메타데이터 패널**: 최근 10개 뉴스의 제목, 출처, 영향도 (원문 제외)
- **빅 이벤트 타임라인**: FOMC, CPI, ETF 승인 등 이벤트와 영향도, 유효기간
- **VPS 리소스/시스템 모니터링**: CPU, 메모리, 디스크 사용량, API 응답시간, FinBERT 배치 상태
- **알림 로그 패널**: 최근 텔레그램 알림, STRONG+ 신호만 표시, 속도 제한 통계

## 기술 스택

- **Frontend**: React 18 + TypeScript
- **UI**: Tailwind CSS + Lucide React Icons
- **Charts**: Chart.js + react-chartjs-2
- **WebSocket**: Socket.io-client
- **Build**: Vite
- **Date**: date-fns (한국어 지원)

## 설치 및 실행

### 개발 환경

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 브라우저에서 http://localhost:3000 접속
```

### 프로덕션 빌드

```bash
# 빌드
npm run build

# 미리보기
npm run preview

# 호스트 모드로 실행 (외부 접근 허용)
npm run serve
```

## 환경 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# API 서버 URL
VITE_API_URL=http://109.123.239.30:8000

# WebSocket URL
VITE_WS_URL=ws://109.123.239.30:8000

# 환경
VITE_ENV=production

# 업데이트 간격 (초)
VITE_UPDATE_INTERVAL=30
```

## API 엔드포인트

대시보드는 다음 API 엔드포인트를 사용합니다:

- `GET /api/v1/sentiment/current` - 현재 감정점수
- `GET /api/v1/news/recent?limit=10` - 최근 뉴스 메타데이터
- `GET /api/v1/events/active` - 활성 빅 이벤트
- `GET /api/v1/scheduler/stats` - 시스템 통계
- `GET /api/v1/notifications/recent?limit=50` - 최근 알림
- `WebSocket /ws` - 실시간 업데이트

## WebSocket 이벤트

실시간 업데이트를 위한 WebSocket 이벤트:

- `sentiment_update` - 감정점수 업데이트
- `news_update` - 새 뉴스 알림
- `event_update` - 빅 이벤트 발생
- `system_update` - 시스템 상태 변경
- `notification` - 새 알림 수신

## 배포

### VPS 배포

```bash
# 프로덕션 빌드
npm run build

# dist 폴더를 웹서버에 업로드
# nginx 설정에서 API 프록시 설정 필요
```

### Docker 배포

```dockerfile
FROM nginx:alpine
COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

## 브라우저 호환성

- Chrome 90+
- Firefox 90+
- Safari 14+
- Edge 90+

## 성능 최적화

- 코드 스플리팅으로 청크 분할
- WebSocket 연결 최적화 및 자동 재연결
- 차트 데이터 메모이제이션
- 스크롤 가상화 (대량 데이터)

## 문제 해결

### WebSocket 연결 실패
- API 서버가 실행 중인지 확인
- 방화벽/프록시 설정 확인
- CORS 설정 확인

### 데이터 로딩 실패
- API 엔드포인트 URL 확인
- 네트워크 연결 상태 확인
- 브라우저 개발자 도구에서 오류 확인

## 라이선스

MIT License