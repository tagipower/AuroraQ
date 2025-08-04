# AuroraQ VPS Deployment - Final Folder Structure

## 📁 최종 정리된 폴더 구조

```
vps-deployment/
├── 📁 config/                     # 설정 파일들
│   ├── env_loader.py              # 환경변수 로더
│   ├── env_validator.py           # 환경변수 검증
│   ├── metrics_integration.yaml   # 메트릭 통합 설정
│   ├── nginx.conf                 # Nginx 설정
│   ├── onnx_settings.py          # ONNX 모델 설정
│   ├── postgresql.conf           # PostgreSQL 설정
│   ├── prometheus.yml            # Prometheus 설정
│   └── redis.conf                # Redis 설정
│
├── 📁 core/                       # 핵심 시스템 모듈
│   ├── 📁 error_recovery/         # 오류 복구 시스템
│   │   ├── api_connection_recovery.py
│   │   └── error_recovery_system.py
│   │
│   ├── 📁 event_management/       # 이벤트 TTL 관리 (P8)
│   │   ├── ttl_event_manager.py
│   │   ├── expiry_processor.py
│   │   └── cleanup_scheduler.py
│   │
│   ├── 📁 logging_management/     # 로그 관리 시스템 (P7)
│   │   ├── log_manager.py
│   │   ├── backup_manager.py
│   │   ├── archive_manager.py
│   │   └── log_orchestrator.py
│   │
│   ├── 📁 model_management/       # 모델 관리 시스템 (P4)
│   │   ├── model_management_system.py
│   │   ├── model_quality_monitor.py
│   │   └── fine_tuning_manager.py
│   │
│   ├── 📁 performance/            # 성능 최적화 (P1, P5)
│   │   ├── dynamic_batch_manager.py
│   │   ├── memory_optimizer.py
│   │   └── performance_optimizer.py
│   │
│   ├── 📁 resource_management/    # 리소스 관리 (P5)
│   │   ├── system_resource_manager.py
│   │   └── process_optimizer.py
│   │
│   └── 📁 strategy_protection/    # 전략 보호 시스템 (P6)
│       ├── signal_validator.py
│       ├── safety_checker.py
│       └── anomaly_detector.py
│
├── 📁 services/                   # 마이크로서비스들
│   ├── 📁 sentiment-service/      # 감정 분석 서비스
│   │   ├── 📁 api/               # API 라우터
│   │   ├── 📁 collectors/        # 데이터 수집기
│   │   ├── 📁 config/           # 서비스 설정
│   │   ├── 📁 deployment/       # 배포 스크립트
│   │   ├── 📁 models/           # AI 모델
│   │   ├── 📁 monitors/         # 모니터링
│   │   ├── 📁 processors/       # 데이터 처리기
│   │   └── 📁 schedulers/       # 스케줄러
│   │
│   └── 📁 trading-service/        # 거래 서비스
│       └── 📁 trading/           # 거래 로직
│           ├── 📁 config/       # 거래 설정
│           ├── 📁 models/       # PPO 모델
│           ├── ppo_agent.py     # PPO 에이전트
│           ├── ppo_strategy.py  # PPO 전략
│           ├── rule_strategies.py # 룰 기반 전략
│           ├── vps_market_data.py # 시장 데이터
│           ├── vps_order_manager.py # 주문 관리
│           ├── vps_position_manager.py # 포지션 관리
│           └── vps_realtime_system.py # 실시간 시스템
│
├── 📁 infrastructure/             # 인프라 관리
│   ├── 📁 monitoring/            # 모니터링 시스템
│   │   ├── monitor_vps_trading.py
│   │   ├── monitoring_alert_system.py
│   │   └── prometheus.yml
│   │
│   ├── 📁 logs/                  # 로그 저장소
│   │   ├── 📁 metrics/          # 메트릭 로그
│   │   └── 📁 summary_logs/     # 요약 로그
│   │
│   └── 📁 vps_logging/          # VPS 로깅 시스템
│       ├── unified_log_manager.py
│       ├── log_retention_policy.py
│       └── vps_integration.py
│
├── 📁 web/                       # 웹 인터페이스
│   ├── 📁 dashboard/            # 대시보드
│   │   ├── aurora_dashboard_final.py
│   │   ├── onnx_dashboard_config.json
│   │   └── start_dashboard.bat
│   ├── start_dashboard.bat      # 대시보드 시작 스크립트
│   └── start_dashboard.sh       # 대시보드 시작 스크립트 (Linux)
│
├── 📁 deployment/               # 배포 관련
│   ├── Dockerfile              # Docker 설정
│   ├── docker-compose-production.yml
│   ├── deploy_vps.sh           # VPS 배포 스크립트
│   ├── api_system.py           # API 시스템
│   ├── vps_standalone_runner.py # 독립 실행기
│   ├── quick_start.bat         # 빠른 시작
│   └── requirements.txt        # Python 의존성
│
├── 📁 scripts/                 # 운영 스크립트
│   ├── deploy_trading_system.sh
│   ├── monitor_trading_system.sh
│   ├── restart_trading_system.sh
│   ├── stop_trading_system.sh
│   ├── test_trading_system.sh
│   └── verify_ppo_integration.py
│
├── 📁 utils/                   # 유틸리티
│   ├── debug_system.py         # 디버그 시스템
│   ├── import_fixer.py         # 임포트 수정기
│   ├── integration_validator.py # 통합 검증기
│   ├── security_system.py      # 보안 시스템
│   ├── system_validator.py     # 시스템 검증기
│   ├── validate_deployment.py  # 배포 검증
│   └── validate_simple.py      # 간단 검증
│
├── 📁 tests/                   # 통합 테스트
│   └── event_management_integration_test.py
│
├── 📁 docs/                    # 문서
│   └── env_endpoint_security_report.md
│
└── 📄 설정 및 문서 파일들
    ├── README.md                # 메인 README
    ├── VPS_DEPLOYMENT_STRUCTURE.md
    ├── DASHBOARD_GUIDE.md
    └── FOLDER_STRUCTURE_FINAL.md (이 파일)
```

## 🏗️ 아키텍처 개요

### 1. 📦 Core Modules (핵심 모듈)
- **Event Management (P8)**: TTL 기반 이벤트 생명주기 관리
- **Logging Management (P7)**: 통합 로그 관리 및 백업 시스템
- **Model Management (P4)**: AI 모델 품질 모니터링 및 Fine-tuning
- **Performance**: 동적 배치 관리 및 메모리 최적화
- **Resource Management (P5)**: 시스템 리소스 관리
- **Strategy Protection (P6)**: 거래 전략 보호 및 검증
- **Error Recovery**: API 연결 실패 복구 시스템

### 2. 🚀 Services (마이크로서비스)
- **Sentiment Service**: 감정 분석 및 뉴스 처리
- **Trading Service**: PPO/룰 기반 거래 실행

### 3. 🔧 Infrastructure (인프라)
- **Monitoring**: Prometheus 기반 모니터링
- **Logging**: 통합 로그 관리
- **VPS Logging**: VPS 특화 로깅 시스템

### 4. 🌐 Web Interface
- **Dashboard**: Streamlit 기반 실시간 대시보드
- **Monitoring UI**: 시스템 상태 및 성능 모니터링

### 5. 🚢 Deployment & Operations
- **Docker**: 컨테이너화된 배포
- **Scripts**: 운영 자동화 스크립트
- **Utils**: 시스템 검증 및 유틸리티

## 🔄 데이터 흐름

1. **Market Data** → Trading Service → Strategy Protection → Order Execution
2. **News/Events** → Sentiment Service → Event Management → TTL Processing
3. **System Logs** → Logging Management → Archive → Cleanup
4. **Performance Metrics** → Resource Management → Optimization
5. **All Activities** → Monitoring → Dashboard → Alerts

## 🛡️ 보안 및 안정성

- **Strategy Protection**: 비정상 신호 감지 및 차단
- **Error Recovery**: 자동 복구 메커니즘
- **Resource Management**: 과부하 방지
- **Logging**: 완전한 감사 추적
- **Event TTL**: 자동 정리 및 메모리 관리

## 📊 모니터링 및 관찰성

- **Real-time Dashboard**: 실시간 시스템 상태
- **Performance Metrics**: 성능 지표 추적
- **Log Aggregation**: 중앙화된 로그 관리
- **Alert System**: 이상 상황 알림
- **Health Checks**: 시스템 상태 점검

이 구조는 확장 가능하고 유지보수가 용이하며, 각 컴포넌트가 독립적으로 작동하면서도 통합된 시스템을 구성합니다.