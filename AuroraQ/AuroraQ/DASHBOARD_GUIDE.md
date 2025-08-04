# AuroraQ Advanced Dashboard Guide

## 🎯 개요

AuroraQ VPS 시스템용 고도화 대시보드는 PPO와 Rule 전략을 통합한 실시간 모니터링 및 분석 시스템입니다.

## 🚀 실행 방법

### Windows
```bash
# 배치 파일 실행
start_dashboard.bat

# 또는 Python 직접 실행
python start_aurora_dashboard.py
```

### Linux/Unix
```bash
# 셸 스크립트 실행
./start_dashboard.sh

# 또는 Python 직접 실행
python3 start_aurora_dashboard.py
```

## 📊 대시보드 구성 (11개 패널)

### 1. 🎯 Overview (시스템 전체 상태)
- AuroraQ 시스템 전반적 상태 요약
- PPO vs Rule 성과 비교 개요
- 시스템 알림 및 주요 메트릭
- 실시간 성과 지표

### 2. ⚖️ Strategy Battle (전략 대결)
- PPO와 Rule 전략 상세 성과 비교
- 실시간 성과 메트릭 (승률, 수익률, 샤프비율)
- PPO 보상 shaping 분석
- Rule 전략별 특성 분석

### 3. 🤖 PPO Engine (PPO 엔진)
- PPO 모델 상세 정보 (v2.3.1)
- 보상 shaping 컴포넌트 분석
- 학습 진행 상황 및 수렴도
- 동적 가중치 조정 현황

### 4. 📏 Rule Metrics (룰 전략 메트릭)
- 5개 Rule 전략별 상세 성과
- 전략 특성 및 선택 로직
- 성과 시각화 차트
- 전략별 안정성 지표

### 5. 💰 P&L Analysis (손익 분석)
- 7일간 일별 손익 분석
- PPO vs Rule 기여도 분석
- 성과 트렌드 및 통계
- 시각화 차트

### 6. ⚠️ Risk Monitor (위험 관리)
- 실시간 포지션 위험도 모니터링
- 선물 리스크 관리 상태
- 시장 리스크 분석
- 위험도별 권장 액션

### 7. 🔄 Adaptive Flow (적응형 흐름)
- 적응형 전략 선택 흐름도
- 최근 전략 선택 히스토리
- 의사결정 팩터 분석
- 학습 적응 통계

### 8. 📊 Market Intel (시장 정보)
- 실시간 시장 데이터
- 감정 분석 상세
- 기술적 분석 지표
- 소셜 미디어 감정 추적

### 9. ⚡ Live Trading (실시간 거래)
- 실시간 거래 활동
- 주문 관리 상태
- 거래 통계 및 성과
- 거래소 연결 상태

### 10. 📋 System Logs (시스템 로그)
- AuroraQ 통합 로그 시스템
- PPO 로그 분석
- 시스템 로그 통계
- 실시간 이벤트 모니터링

### 11. 🖥️ Infrastructure (인프라)
- VPS 시스템 리소스 모니터링
- 네트워크 연결 상태
- 시스템 성능 지표
- 모니터링 및 알림 설정

## 🎮 대시보드 조작법

- **↑/↓**: 메뉴 탐색
- **ENTER**: 메뉴 선택
- **R**: 새로고침
- **Q**: 종료

## 🔧 주요 기능

### 실시간 데이터 업데이트
- 2초마다 자동 새로고침
- TTL 캐시를 통한 성능 최적화
- 실시간 로그 파싱

### AuroraQ 시스템 통합
- PPO 전략 실시간 모니터링
- Rule 전략 성과 추적
- 적응형 전략 선택 분석
- 보상 shaping 실시간 분석

### 고급 시각화
- Rich TUI 기반 터미널 대시보드
- 컬러 코딩 상태 표시
- 프로그레스 바 및 차트
- 동적 레이아웃

## 📁 파일 구조

```
vps-deployment/
├── dashboard/
│   └── aurora_dashboard_final.py     # 메인 대시보드
├── trading/
│   ├── ppo_strategy.py              # PPO 전략
│   ├── ppo_score_logger.py          # PPO 로거
│   ├── rule_strategy_logger.py      # Rule 로거
│   └── vps_strategy_adapter.py      # 전략 어댑터
├── logs/                            # 로그 디렉토리
├── start_aurora_dashboard.py        # 대시보드 런처
├── start_dashboard.bat              # Windows 실행 스크립트
└── start_dashboard.sh               # Linux 실행 스크립트
```

## 📦 필수 의존성

```bash
pip install rich psutil pandas numpy
```

## 🔍 주요 특징

### PPO vs Rule 전략 통합
- 실시간 성과 비교
- 적응형 전략 선택
- 동적 가중치 조정
- 보상 shaping 최적화

### 고도화된 모니터링
- 11개 전문화된 패널
- 실시간 데이터 처리
- 캐시 기반 성능 최적화
- 로그 통합 분석

### VPS 환경 최적화
- 시스템 리소스 모니터링
- 네트워크 연결 상태 추적
- 성능 지표 실시간 표시
- 알림 시스템 통합

## 🚨 문제 해결

### 모듈 Import 오류
- AuroraQ 모듈이 없어도 시뮬레이션 모드로 실행 가능
- 필수 패키지 자동 설치 안내

### 성능 이슈
- TTL 캐시를 통한 메모리 최적화
- 2초 새로고침 주기로 CPU 부하 최소화
- 로그 파일 자동 정리

### 연결 문제
- 거래소 API 연결 상태 실시간 모니터링
- 자동 재연결 메커니즘
- 장애 시 알림 시스템

## 📈 성과 지표

### PPO 엔진
- 모델 버전: v2.3.1
- 수렴도: 89.3%
- 평균 보상: +0.456
- 신뢰도: 78.4%

### Rule 전략
- 5개 전략 동시 운영
- 평균 점수: 0.687
- 선택률 분산: 18-28%
- 안정성: 79-92%

### 시스템 성능
- 업타임: 99.8%
- 응답시간: <50ms p95
- 메모리 효율: 87.3%
- CPU 효율: 91.2%

## 📞 지원 및 문의

대시보드 관련 문의사항이나 개선 제안은 시스템 관리자에게 연락하시기 바랍니다.

---

**AuroraQ Advanced Dashboard v1.0**  
*PPO + Rule 전략 통합 실시간 모니터링 시스템*