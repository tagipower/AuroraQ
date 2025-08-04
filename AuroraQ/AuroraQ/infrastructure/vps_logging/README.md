# 🔧 AuroraQ 통합 로깅 시스템 v1.0

VPS deployment용 4가지 범주별 로그 통합 관리 시스템

## 📋 개요

AuroraQ의 방어적 보안 분석을 위한 통합 로그 관리 시스템입니다. 4가지 로그 범주를 차별화하여 처리하며, VPS 환경에 최적화되어 있습니다.

### 🎯 4가지 로그 범주

| 범주 | 목적 | 저장 형식 | 보존 기간 | 용도 |
|------|------|-----------|-----------|------|
| **Raw Logs** | 디버깅/추적용 | `.jsonl` | 3-7일 | 실시간 디버깅, 문제 추적 |
| **Summary Logs** | 분석/리포트용 | `.csv` | 수개월 | 성능 분석, 리포트 생성 |
| **Training Logs** | 학습/검증용 | `.pkl`, `.npz` | 장기보존 | ML 모델 학습, 성능 검증 |
| **Tagged Logs** | 고의미 이벤트 | `.jsonl` | 조건부 영구 | 보안 이벤트, 중요 알림 |

## 🏗️ 아키텍처

```
logging/
├── __init__.py                      # 패키지 초기화
├── unified_log_manager.py           # 통합 로그 관리자 (핵심)
├── vps_integration.py               # VPS deployment 통합 어댑터
├── log_retention_policy.py          # 보존 정책 및 자동 정리
└── README.md                        # 이 문서
```

### 핵심 컴포넌트

1. **UnifiedLogManager**: 4범주 통합 처리
2. **VPSLogIntegrator**: 기존 VPS 시스템과 연동
3. **LogRetentionManager**: 보존 정책 및 자동 정리
4. **LoggingAdapter**: 기존 코드 호환성

## 🚀 사용법

### 1. 기본 설정

```python
from vps_deployment.logging import create_vps_log_manager, get_vps_log_integrator

# VPS 최적화된 로그 관리자 생성
log_manager = create_vps_log_manager("/app/logs")

# 통합 로깅 시작
await log_manager.start_background_tasks()

# VPS 통합기 사용
integrator = get_vps_log_integrator("/app/logs")
```

### 2. 범주별 로깅

```python
from vps_deployment.logging import LogCategory, LogLevel

# Raw 로그 (디버깅용)
await log_manager.log(
    category=LogCategory.RAW,
    level=LogLevel.INFO,
    component="onnx_sentiment", 
    event_type="inference",
    message="ONNX model inference completed",
    metadata={"confidence": 0.85, "processing_time": 0.15}
)

# Training 로그 (학습용)
await log_manager.log(
    category=LogCategory.TRAINING,
    level=LogLevel.INFO,
    component="ml_model",
    event_type="training_data",
    message="Training data recorded",
    metadata={"features": [1.2, 3.4, 5.6], "label": 1}
)

# Tagged 로그 (보안 이벤트)
await log_manager.log(
    category=LogCategory.TAGGED,
    level=LogLevel.CRITICAL,
    component="security",
    event_type="auth_failure", 
    message="Multiple failed login attempts",
    tags=["security", "auth", "suspicious"],
    metadata={"ip": "192.168.1.100", "attempts": 5}
)
```

### 3. VPS 통합 함수 사용

```python
from vps_deployment.logging.vps_integration import (
    log_onnx_event, log_batch_event, log_security_alert
)

# ONNX 추론 이벤트
await log_onnx_event(
    text="Bitcoin price analysis",
    confidence=0.85,
    inference_time=0.15,
    model_version="finbert_onnx_v2"
)

# 배치 처리 이벤트  
await log_batch_event(
    batch_size=100,
    processing_time=5.2,
    success_count=98, 
    error_count=2
)

# 보안 알림
await log_security_alert(
    event_type="anomaly_detected",
    severity="high",
    description="Unusual API access pattern detected"
)
```

### 4. 기존 코드와의 호환성

```python
# 기존 로거 코드를 그대로 사용 가능
integrator = get_vps_log_integrator()
logger = integrator.get_logger("my_component")

logger.info("This works with existing code")
logger.error("Error handling also works", metadata={"error_code": 500})
```

## ⚙️ VPS 최적화 설정

### 메모리 최적화
- 버퍼 크기 제한: 512MB
- 배치 처리: 50개 단위
- 자동 플러시: 1분 간격

### 저장공간 최적화
- 자동 압축: 1일 후 GZIP 압축
- 보존 정책: 범주별 차별화
- 아카이브: 오래된 파일 자동 이동

### 성능 최적화
- 비동기 처리: 논블로킹 I/O
- 병렬 압축: ThreadPoolExecutor 사용
- 스마트 캐싱: 중복 제거

## 📊 보존 정책

### VPS 최적화 모드 (기본)

| 범주 | 활성기간 | 압축시점 | 아카이브 | 삭제 |
|------|----------|----------|----------|------|
| Raw | 3일 | 1일 후 | 7일 | 7일 후 |
| Summary | 30일 | 7일 후 | 90일 | 90일 후 |
| Training | 90일 | 30일 후 | 365일 | 영구보존 |
| Tagged | 365일 | 90일 후 | 5년 | 영구보존 |

### 자동 정리 기능

```python
from vps_deployment.logging.log_retention_policy import create_vps_retention_manager

# 보존 정책 관리자
retention_manager = create_vps_retention_manager("/app/logs")

# 정책 실행
stats = await retention_manager.run_retention_policy()

# 디스크 사용량 기준 긴급 정리
emergency_stats = await retention_manager.cleanup_by_disk_usage(target_usage_percent=80.0)
```

## 🔧 설정 옵션

### docker-compose.yml 설정

```yaml
services:
  onnx-sentiment:
    environment:
      - ONNX_ENABLE_UNIFIED_LOGGING=true
      - ONNX_UNIFIED_LOG_DIR=/app/logs/unified
      - ONNX_LOG_LEVEL=INFO
      - ONNX_LOG_FORMAT=json
    volumes:
      - ./logs:/app/logs
```

### config/onnx_settings.py 설정

```python
# 통합 로깅 설정
enable_unified_logging: bool = True
unified_log_dir: str = "/app/logs/unified"
log_level: str = "INFO"
log_format: str = "json"
```

## 📈 모니터링 및 메트릭

### 로그 통계 확인

```python
# 통합 관리자 통계
stats = log_manager.get_stats()
print(f"버퍼 크기: {stats['buffer_sizes']}")
print(f"메모리 사용량: {stats['memory_usage_mb']:.2f}MB")

# VPS 통합기 통계  
vps_stats = integrator.get_stats()
print(f"어댑터 수: {vps_stats['adapters_count']}")

# 저장소 사용량
storage_stats = await retention_manager.get_storage_stats()
for category, stat in storage_stats.items():
    print(f"{category.value}: {stat.total_size_mb:.2f}MB")
```

### Grafana 대시보드 연동

로그 메트릭은 기존 Grafana 대시보드에서 확인 가능:
- `http://localhost:3000` (admin/admin)
- 패널: "ONNX System Intelligence"에서 로깅 통계 확인

## 🔧 문제 해결

### 1. 로그 파일이 생성되지 않는 경우

```bash
# 디렉토리 권한 확인
ls -la /app/logs/

# 디렉토리 생성
mkdir -p /app/logs/{raw,summary,training,tagged,archive}
chmod 755 /app/logs/

# 컨테이너 재시작
docker-compose restart onnx-sentiment
```

### 2. 메모리 사용량이 높은 경우

```python
# 버퍼 크기 확인
stats = log_manager.get_stats()
print("Buffer sizes:", stats['buffer_sizes'])

# 수동 플러시
for category in LogCategory:
    await log_manager._flush_category(category)
```

### 3. 디스크 공간 부족

```python
# 긴급 정리 실행
emergency_stats = await retention_manager.cleanup_by_disk_usage(70.0)

# 수동 압축
await retention_manager.run_retention_policy()
```

### 4. 로그 레벨 조정

```bash
# 환경변수로 조정
export ONNX_LOG_LEVEL=WARNING

# 또는 docker-compose.yml에서
environment:
  - ONNX_LOG_LEVEL=WARNING
```

## 🧪 테스트

### 단위 테스트 실행

```bash
# 기본 기능 테스트
python -m vps_deployment.logging.unified_log_manager

# VPS 통합 테스트  
python -m vps_deployment.logging.vps_integration

# 보존 정책 테스트
python -m vps_deployment.logging.log_retention_policy
```

### 통합 테스트

```python
import asyncio
from vps_deployment.logging import *

async def integration_test():
    # 관리자 생성
    manager = create_vps_log_manager("/tmp/test_logs")
    integrator = setup_vps_logging("/tmp/test_logs")
    
    # 백그라운드 작업 시작
    await manager.start_background_tasks()
    
    # 다양한 로그 생성
    await log_onnx_event("Test inference", 0.9, 0.1)
    await log_batch_event(50, 2.5, 48, 2)
    await log_security_alert("test_event", "low", "Test security event")
    
    # 통계 확인
    print("Manager stats:", manager.get_stats())
    print("Integrator stats:", integrator.get_stats())
    
    # 정리
    await manager.shutdown()
    await integrator.shutdown()

asyncio.run(integration_test())
```

## 🔄 업그레이드 가이드

### v1.0으로 업그레이드

1. **새 패키지 설치**
   ```bash
   # requirements.txt에 추가
   structlog>=23.1.0
   psutil>=5.9.0
   ```

2. **기존 코드 마이그레이션**
   ```python
   # 기존
   import logging
   logger = logging.getLogger("component")
   
   # 새 방식 (호환성 유지)
   from vps_deployment.logging import get_vps_log_integrator
   integrator = get_vps_log_integrator()
   logger = integrator.get_logger("component")
   ```

3. **설정 업데이트**
   ```yaml
   # docker-compose.yml에 추가
   environment:
     - ONNX_ENABLE_UNIFIED_LOGGING=true
     - ONNX_UNIFIED_LOG_DIR=/app/logs/unified
   volumes:
     - ./logs:/app/logs
   ```

## 📞 지원

### 로그 관련 문제 발생시

1. **로그 상태 확인**
   ```python
   stats = log_manager.get_stats()
   storage_stats = await retention_manager.get_storage_stats()
   ```

2. **디버그 모드 활성화**
   ```bash
   export ONNX_LOG_LEVEL=DEBUG
   ```

3. **수집할 정보**
   - 통합 로그 통계 (get_stats() 결과)
   - 디스크 사용량 (df -h /app/logs)
   - 메모리 사용량 (docker stats)
   - 에러 로그 (/app/logs/raw/*/error_*.jsonl)

---

**🤖 AuroraQ 통합 로깅 시스템 v1.0**  
*4범주 로그 통합 관리 • VPS 최적화 • 방어적 보안 분석*