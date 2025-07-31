# AuroraQ Dashboard - Claude Code Style Typing Effects

클로드 코드와 동일한 타이핑 애니메이션 효과가 구현된 실시간 대시보드

## 🎯 타이핑 효과 특징

### Claude Code 호환 타이핑 시스템
- **기본 속도**: 0.015초 (Claude Code와 동일)
- **고속 타이핑**: 0.008초 (숫자, 구두점)
- **저속 타이핑**: 0.025초 (강조 표시)
- **블록 커서**: █ (타이핑 중 표시)

### 지능형 타이핑 속도 조절
```python
# 문자 유형별 속도 조절
if char.isdigit() or char in '.,+-':
    char_delay *= 0.7  # 숫자/구두점 빠르게
elif char.isupper():
    char_delay *= 1.2  # 대문자 느리게  
elif char == ' ':
    char_delay *= 0.5  # 공백 빠르게
elif char in '\n\t':
    char_delay *= 2.0  # 줄바꿈에서 일시정지
```

### 실시간 데이터 변화 감지
- **변화 감지**: 이전 값과 비교하여 변경사항 자동 탐지
- **선택적 애니메이션**: 변경된 데이터만 타이핑 효과 적용
- **위치 기반 타이핑**: 화면 특정 위치에서 타이핑 애니메이션

## 🎮 사용법

### 기본 실행
```bash
# 타이핑 효과 포함 대시보드
python aurora_dashboard_v3.py

# 타이핑 효과 데모
python demo_typing.py

# 전체 대시보드 데모
python demo_typing.py --dashboard
```

### 키보드 단축키
- **T**: 타이핑 효과 토글 (활성화/비활성화)
- **Q**: 종료
- **R**: 데이터 리셋

### 설정 옵션
```python
# 타이핑 효과 활성화/비활성화
dashboard.enable_typing = True/False

# 타이핑 속도 조절
typing_effect.typing_speed = 0.015  # 기본값
typing_effect.fast_speed = 0.008    # 빠른 속도
typing_effect.slow_speed = 0.025    # 느린 속도
```

## 🎨 타이핑 효과 적용 대상

### 실시간 데이터 업데이트
1. **Sentiment Score**: 감정 점수 변화시 타이핑
2. **CPU/Memory**: 리소스 사용률 변화시 타이핑
3. **API Response**: 응답시간 변화시 타이핑
4. **System Alerts**: 새 알림 발생시 타이핑

### 시스템 메시지
- **시작 메시지**: "Initializing AuroraQ Terminal Dashboard v3.0..."
- **종료 메시지**: "Shutting down dashboard..."
- **연결 상태**: "Services connected successfully"

### 특별 이벤트
- **극한 감정점수**: ±0.8 초과시 깜박임과 함께 타이핑
- **시스템 경고**: CPU/메모리 80% 초과시 경고 메시지 타이핑
- **새 알림**: 실시간 알림 발생시 즉시 타이핑

## 🔧 기술적 구현

### ClaudeCodeTypingEffect 클래스
```python
class ClaudeCodeTypingEffect:
    def __init__(self):
        self.typing_speed = 0.015  # Claude Code 호환 속도
        self.cursor_char = '█'     # 블록 커서
        
    async def type_text(self, text: str, speed_mode: str = 'normal'):
        # 문자별 가변 속도로 타이핑
        # 커서 표시 및 제거
        # ANSI 색상 코드 지원
```

### 데이터 변화 감지 시스템
```python
def detect_data_changes(self, current_data: Dict[str, Any]):
    # 이전 값과 현재 값 비교
    # 변경사항 딕셔너리 반환
    # 타이핑 애니메이션 트리거
```

### 위치 기반 타이핑
```python
async def type_in_position(self, row: int, col: int, text: str):
    # ANSI 커서 이동 코드 사용
    # 특정 위치에서 타이핑 시작
    # 기존 화면 레이아웃 유지
```

## 📊 성능 최적화

### 효율적인 애니메이션
- **선택적 적용**: 3번째 업데이트마다만 타이핑 효과 (과부하 방지)
- **비동기 처리**: 타이핑 중에도 다른 작업 계속 진행
- **메모리 관리**: 이전 데이터 저장소 크기 제한

### 사용자 경험 개선
- **토글 기능**: 타이핑 효과 실시간 활성화/비활성화
- **속도 조절**: 내용에 따른 지능형 속도 조절
- **시각적 피드백**: 타이핑 중 커서 표시

## 🎯 실제 사용 시나리오

### 트레이딩 상황
```
Sentiment Score: +0.742 ▲▲  (타이핑 애니메이션)
Status: EXTREMELY BULLISH    (강조 타이핑)
Alert: Position adjusted     (실시간 타이핑)
```

### 시스템 모니터링
```
CPU: 45.2% ███████████░░░   (수치 빠른 타이핑)
Memory: 62.5% ████████████   (게이지 함께 업데이트)
⚠ 14:23 High CPU usage      (경고 알림 타이핑)
```

## 🚀 확장 가능성

### 향후 개선사항
- **사운드 효과**: 타이핑 소리 추가 (옵션)
- **커스텀 속도**: 사용자 정의 타이핑 속도
- **애니메이션 패턴**: 다양한 타이핑 스타일
- **키보드 입력**: 실시간 명령어 타이핑

이제 AuroraQ 대시보드가 Claude Code와 동일한 생생한 타이핑 애니메이션으로 데이터 변화를 시각적으로 표현합니다!