# AuroraQ Production 설치 가이드

## 📋 시스템 요구사항

### 하드웨어 요구사항
- **CPU**: 4코어 이상 (Intel i5/AMD Ryzen 5 권장)
- **RAM**: 8GB 이상 (16GB 권장)
- **저장공간**: 10GB 이상 여유 공간
- **네트워크**: 안정적인 인터넷 연결 (실시간 데이터용)

### 소프트웨어 요구사항
- **Python**: 3.8 이상 (3.10 권장)
- **운영체제**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Git**: 버전 관리 (선택사항)

## 🚀 설치 과정

### 1. Python 환경 준비

#### 가상환경 생성 (권장)
```bash
# Python 가상환경 생성
python -m venv auroraQ_env

# 가상환경 활성화
# Windows:
auroraQ_env\Scripts\activate
# macOS/Linux:
source auroraQ_env/bin/activate
```

#### Conda 환경 생성 (대안)
```bash
conda create -n auroraQ python=3.10
conda activate auroraQ
```

### 2. AuroraQ Production 설치

#### 방법 1: 소스코드에서 설치
```bash
# 디렉토리 이동
cd AuroraQ_Production

# 의존성 설치
pip install -r requirements.txt

# 패키지 개발 모드 설치
pip install -e .
```

#### 방법 2: setup.py 사용
```bash
# 패키지 설치
python setup.py install
```

### 3. 의존성 확인

#### 필수 패키지 설치 확인
```bash
# 핵심 라이브러리 확인
python -c "import numpy, pandas, torch; print('핵심 라이브러리 설치 완료')"

# 강화학습 라이브러리 확인
python -c "import stable_baselines3; print('SB3 설치 완료')"

# 센티멘트 분석 확인
python -c "import transformers; print('Transformers 설치 완료')"
```

#### GPU 지원 설치 (선택사항)
```bash
# CUDA 지원 PyTorch (GPU 사용 시)
pip install torch[cuda] --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4. 설정 파일 준비

#### 기본 설정 복사
```bash
# 설정 파일이 자동 생성되지 않은 경우
cp config.yaml.example config.yaml
```

#### 설정 파일 편집
```yaml
# config.yaml
trading:
  max_position_size: 0.1          # 본인의 리스크 허용도에 맞게 조정
  max_daily_trades: 10            # 일일 거래 한도

strategy:
  rule_strategies:                # 사용할 전략 선택
    - "RuleStrategyA"
    - "RuleStrategyB" 
  enable_ppo: true               # PPO 사용 여부

notifications:
  enable_notifications: true      # 알림 활성화
  channels:
    - "console"                  # 콘솔 출력
    - "file"                     # 파일 로그
```

### 5. 설치 검증

#### 기본 테스트 실행
```bash
# 전체 테스트 실행
python -m pytest tests/ -v

# 또는 개별 테스트
python tests/test_realtime.py
python tests/test_strategies.py
python tests/test_optimization.py
```

#### 데모 실행
```bash
# 2분 데모 실행
python main.py --mode demo --duration 2

# 또는 직접 실행
python -c "
from core import RealtimeHybridSystem, TradingConfig
config = TradingConfig(max_position_size=0.01, min_data_points=5)
system = RealtimeHybridSystem(config)
print('✅ AuroraQ Production 설치 완료!')
"
```

## 🔧 트러블슈팅

### 일반적인 문제 해결

#### 1. 의존성 설치 실패
```bash
# pip 업그레이드
pip install --upgrade pip setuptools wheel

# 캐시 클리어 후 재설치
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

#### 2. PyTorch 설치 문제
```bash
# CPU 버전으로 설치
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 또는 conda 사용
conda install pytorch cpuonly -c pytorch
```

#### 3. Transformers 설치 문제
```bash
# 기본 버전 설치
pip install transformers[torch]

# 또는 최소 설치
pip install transformers --no-deps
pip install torch tokenizers
```

#### 4. 메모리 부족 오류
- RAM 8GB 미만인 경우: `config.yaml`에서 `min_data_points: 10`으로 설정
- 가상환경 메모리 제한: `--memory-limit 4g` 옵션 사용

#### 5. 네트워크 연결 문제
```bash
# 프록시 환경에서 설치
pip install -r requirements.txt --proxy http://proxy.company.com:8080

# 또는 오프라인 설치
pip download -r requirements.txt -d packages/
pip install --find-links packages/ -r requirements.txt --no-index
```

### 플랫폼별 문제

#### Windows
```bash
# Visual Studio Build Tools 필요한 경우
# https://visualstudio.microsoft.com/visual-cpp-build-tools/ 에서 다운로드

# 또는 미리 컴파일된 패키지 사용
pip install --only-binary=all -r requirements.txt
```

#### macOS
```bash
# Xcode Command Line Tools 설치
xcode-select --install

# M1/M2 Mac의 경우
arch -arm64 pip install -r requirements.txt
```

#### Ubuntu/Debian
```bash
# 시스템 의존성 설치
sudo apt update
sudo apt install python3-dev python3-pip build-essential

# 또는 개발 도구 일괄 설치
sudo apt install python3-full
```

## 📚 다음 단계

설치가 완료되었다면:

1. **[사용자 가이드](USER_GUIDE.md)** - 기본 사용법 학습
2. **[설정 가이드](CONFIGURATION.md)** - 상세 설정 방법
3. **[전략 개발 가이드](STRATEGY_DEVELOPMENT.md)** - 사용자 정의 전략 개발
4. **[API 문서](API_REFERENCE.md)** - 프로그래밍 인터페이스

## 🆘 지원

문제가 지속되는 경우:
- 📧 이메일: support@auroraQ.com
- 🐛 이슈 리포트: GitHub Issues
- 📖 문서: https://docs.auroraQ.com
- 💬 커뮤니티: Discord/Slack 채널