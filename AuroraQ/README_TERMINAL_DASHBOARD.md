# Enhanced Terminal Dashboard System

개선된 터미널 대시보드 시스템으로 한글/이모지 정렬 문제를 해결하고 실시간 센티먼트 서비스 모니터링을 제공합니다.

## 🚀 주요 기능

### 1. 개선된 터미널 포매터 (`utils/enhanced_terminal_formatter.py`)

- **다국어 문자 지원**: 한글, 이모지, Wide 문자의 정확한 폭 계산
- **5가지 색상 테마**: DEFAULT, DARK, LIGHT, CYBERPUNK, MINIMAL
- **성능 최적화**: LRU 캐싱으로 17배 성능 향상
- **에러 핸들링**: 강력한 검증 모드와 안전한 폴백
- **유연한 레이아웃**: 헤더, 데이터 라인, 프로그레스 바, 테이블

### 2. 실시간 센티먼트 대시보드 (`sentiment-service/dashboard/terminal_dashboard.py`)

- **실시간 API 연동**: 센티먼트 서비스와 비동기 통신
- **라이브 모니터링**: 5초 간격 자동 갱신
- **시스템 메트릭**: CPU, 메모리, 네트워크, 디스크 사용률
- **연결 상태 추적**: 외부 API 연결 상태 실시간 모니터링
- **성능 통계**: 캐시 히트율, 렌더링 시간 추적

## 📦 설치 및 설정

```bash
# 의존성 설치
pip install wcwidth aiohttp

# 기본 포매터 테스트
python utils/enhanced_terminal_formatter.py

# 센티먼트 대시보드 실행 (한 번만)
python sentiment-service/dashboard/terminal_dashboard.py --once

# 실시간 대시보드 실행
python sentiment-service/dashboard/terminal_dashboard.py --live
```

## 🎨 테마 시스템

### 사용 가능한 테마
- `default`: 기본 색상 스키마
- `dark`: 다크 모드 최적화
- `light`: 라이트 모드 최적화  
- `cyberpunk`: 네온 색상 사이버펑크 스타일
- `minimal`: 최소한의 색상 사용

### 테마 사용법
```bash
# 사이버펑크 테마로 실행
python sentiment-service/dashboard/terminal_dashboard.py --theme cyberpunk --live

# 다크 테마로 한 번 실행
python sentiment-service/dashboard/terminal_dashboard.py --theme dark --once
```

## 🔧 API 구성

대시보드는 다음 엔드포인트에서 데이터를 수집합니다:

```python
endpoints = {
    'fusion_sentiment': '/fusion/current-sentiment',
    'system_health': '/admin/health', 
    'events': '/events/timeline',
    'strategies': '/trading/strategies/performance',
    'metrics': '/admin/metrics'
}
```

## 📊 대시보드 구성 요소

### 1. 메인 센티먼트 스코어
- 융합 센티먼트 점수 및 변화량
- 카테고리별 세부 점수 (News, Reddit, Tech, Historical)

### 2. 빅 이벤트 타임라인
- 최신 3개 이벤트 표시
- 임팩트, 센티먼트, 변동성 정보

### 3. 전략 성과
- AuroraQ/MacroQ 전략 성과
- ROI, Sharpe Ratio, 현재 스코어

### 4. 시스템 상태
- API 연결 상태
- 외부 서비스 연결 요약
- Redis 히트율

### 5. 시스템 메트릭
- CPU, 메모리, 네트워크, 디스크 사용률 (프로그레스 바)
- 실시간 성능 통계

## 🚀 사용 예시

### 기본 사용
```bash
# 한 번만 실행
python sentiment-service/dashboard/terminal_dashboard.py --once

# 실시간 모드 (기본 5초 갱신)
python sentiment-service/dashboard/terminal_dashboard.py --live

# 사용자 정의 설정
python sentiment-service/dashboard/terminal_dashboard.py \
  --service http://localhost:8000 \
  --theme cyberpunk \
  --refresh 3 \
  --live
```

### 프로그래밍 방식 사용
```python
from utils.enhanced_terminal_formatter import EnhancedTerminalFormatter, ColorTheme

# 포매터 생성
formatter = EnhancedTerminalFormatter(
    width=120, 
    theme=ColorTheme.CYBERPUNK,
    enable_caching=True
)

# 데이터 라인 생성
line = formatter.format_data_line(
    "센티먼트 점수", "5.2% (+0.3)", 
    label_width=20, 
    value_color='success'
)
print(line)

# 프로그레스 바 생성
bar = formatter.format_progress_bar(
    "CPU 사용률", 75.5, 
    bar_width=20, 
    label_width=20
)
print(bar)
```

## ⚡ 성능 최적화

### 캐싱 시스템
- **LRU 캐시**: 최대 1000개 문자열 폭 계산 결과 캐시
- **성능 향상**: 17배 빠른 렌더링 (테스트 기준)
- **메모리 효율**: 자동 캐시 크기 관리

### 비동기 데이터 수집
- **병렬 API 호출**: 모든 엔드포인트 동시 조회
- **타임아웃 처리**: 5초 타임아웃으로 응답성 보장
- **에러 복구**: API 실패 시 기본값 표시

## 🛠️ 개발 및 테스트

### 데모 실행
```bash
# 모든 기능 데모
python test_dashboard_demo.py

# 특정 기능만 테스트
python test_dashboard_demo.py --demo formatter
python test_dashboard_demo.py --demo performance
python test_dashboard_demo.py --demo error
```

### 성능 테스트
```bash
# 성능 벤치마크
python test_dashboard_demo.py --demo performance

# 결과 예시:
# With caching: 0.0001s (400 operations)
# Without caching: 0.0018s (400 operations)  
# Performance improvement: 17.00x faster
```

## 🔍 문제 해결

### 일반적인 문제

1. **wcwidth 모듈 없음**
   ```bash
   pip install wcwidth
   ```

2. **API 연결 실패**
   - 센티먼트 서비스가 실행 중인지 확인
   - 올바른 서비스 URL 사용 (`--service` 옵션)

3. **한글 깨짐**
   - 터미널이 UTF-8 인코딩 지원하는지 확인
   - Windows: `chcp 65001` 실행

4. **색상 표시 안됨**
   - 터미널이 ANSI 색상 지원하는지 확인
   - `--theme minimal` 사용해보기

### 디버그 모드
```python
# 검증 모드 활성화
formatter = EnhancedTerminalFormatter(
    width=120,
    validation_mode=True  # 에러 시 상세 정보 표시
)
```

## 📈 향후 개선 계획

- [ ] 웹 기반 실시간 대시보드 통합
- [ ] 히스토리 데이터 차트 추가
- [ ] 알림/경고 시스템 구현
- [ ] 다중 서비스 모니터링 지원
- [ ] 설정 파일 지원
- [ ] 로그 및 메트릭 저장 기능

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.