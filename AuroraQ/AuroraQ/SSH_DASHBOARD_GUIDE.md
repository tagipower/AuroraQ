# 📱💻 AuroraQ SSH Dashboard 가이드

## 🎯 완벽 호환: Terminus (모바일) + PC SSH

이 가이드는 VPS에서 AuroraQ 대시보드를 실행하여 모바일(Terminus)과 PC 모두에서 SSH로 접근할 수 있도록 설정합니다.

## 🚀 설치 방법

### 1단계: VPS에 파일 업로드
```bash
# 로컬에서 VPS로 업로드
scp -r vps-deployment/ user@your-vps-ip:/home/user/auroaq/
```

### 2단계: VPS에서 자동 설정 실행
```bash
# VPS에 SSH 접속
ssh user@your-vps-ip

# 설정 디렉토리로 이동
cd /home/user/auroaq/vps-deployment/scripts

# 실행 권한 부여
chmod +x setup_ssh_dashboard.sh dashboard_launcher.sh

# 자동 설정 실행 (모든 필수 구성 요소 설치)
./setup_ssh_dashboard.sh
```

### 3단계: 환경 적용
```bash
# 로그아웃 후 재접속하여 환경 변수 적용
exit
ssh user@your-vps-ip
```

## 📱 모바일 접근 (Terminus)

### Terminus 앱 설정
1. **App Store에서 Terminus 다운로드**
2. **새 호스트 추가**:
   - Host: `your-vps-ip`
   - Username: `your-username`
   - Authentication: SSH Key 또는 Password
3. **SSH Key 설정** (권장):
   ```bash
   # VPS에서 공개키 확인
   cat ~/.ssh/id_rsa.pub
   # 이 키를 Terminus에 입력
   ```

### 대시보드 실행
```bash
# 연결 후 바로 실행
auroaq

# 또는 직접 실행
auroaq-run
```

## 💻 PC 접근

### Windows (PuTTY, Windows Terminal)
```bash
# PowerShell 또는 CMD
ssh user@your-vps-ip

# 대시보드 실행
auroaq
```

### Mac/Linux
```bash
# 터미널에서
ssh user@your-vps-ip

# 대시보드 실행
auroaq
```

### VSCode SSH Extension
1. **Remote-SSH 확장 설치**
2. **SSH 호스트 추가**: `user@your-vps-ip`
3. **터미널에서 실행**: `auroaq`

## 🎮 사용법

### 대화형 런처 메뉴
```
╔══════════════════════════════════════╗
║        🚀 AuroraQ Dashboard         ║
║     SSH Compatible Launcher         ║
╚══════════════════════════════════════╝

Terminal: mobile | Size: 80x24

1. 📊 Run Dashboard (Foreground)
2. 🔄 Run Dashboard (Background)
3. 📋 Attach to Background Session  
4. ⏹️  Stop Background Session
5. 🔍 Check System Status
6. ⚙️  Install/Update Dependencies
q. 🚪 Exit

🟢 Background session: RUNNING
```

### 자동 화면 최적화
- **모바일** (80열 미만): 자동으로 컴팩트 모드
- **PC** (80열 이상): 풀 레이아웃 모드
- **Terminus**: 가로 회전 권장

### 키보드 조작
- **↑↓**: 메뉴 이동
- **Enter**: 선택
- **q**: 종료
- **r**: 새로고침

## 🔧 고급 기능

### 백그라운드 실행
```bash
# 백그라운드에서 실행 (세션 유지)
auroaq
# 메뉴에서 "2" 선택

# 세션에 다시 연결
auroaq  
# 메뉴에서 "3" 선택
```

### 자동 시작 서비스
```bash
# 서비스 시작
systemctl --user start auroaq-dashboard

# 서비스 중지
systemctl --user stop auroaq-dashboard

# 서비스 상태 확인
systemctl --user status auroaq-dashboard
```

### 시스템 상태 확인
```bash
# 대시보드에서 시스템 상태 확인
auroaq
# 메뉴에서 "5" 선택
```

## 🛠️ 문제 해결

### 연결 문제
```bash
# SSH 연결 테스트
ssh -v user@your-vps-ip

# 방화벽 확인
sudo ufw status

# SSH 서비스 상태
sudo systemctl status ssh
```

### Python 라이브러리 문제
```bash
# 라이브러리 재설치
auroaq
# 메뉴에서 "6" 선택

# 또는 수동 설치
pip3 install --user rich psutil
```

### 화면 크기 문제
```bash
# 터미널 크기 확인
echo "Columns: $(tput cols), Lines: $(tput lines)"

# 강제 크기 설정
export COLUMNS=80 LINES=24
stty cols 80 rows 24
```

### 세션 관리 문제
```bash
# 모든 세션 종료
pkill -f "auroaq\|aurora_dashboard"

# Screen 세션 확인
screen -list

# Tmux 세션 확인  
tmux list-sessions
```

## 📊 성능 최적화

### SSH 연결 최적화
SSH 설정이 자동으로 최적화됩니다:
```
# ~/.ssh/config
ServerAliveInterval 60
ServerAliveCountMax 3
TCPKeepAlive yes
Compression yes
```

### 대시보드 성능
- **새로고침 주기**: 1초 (실시간)
- **메모리 사용량**: ~50MB
- **CPU 사용률**: ~5%

## 🔒 보안 설정

### SSH 키 인증 권장
```bash
# SSH 키 생성 (없는 경우)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 공개키 복사
ssh-copy-id user@your-vps-ip
```

### 방화벽 설정
```bash
# SSH 포트만 허용
sudo ufw allow 22/tcp
sudo ufw enable
```

## 📈 모니터링

### 실시간 로그 확인
```bash
# 서비스 로그
journalctl --user -u auroaq-dashboard -f

# 시스템 리소스
htop
```

### 알림 설정 (선택사항)
Telegram Bot API를 통한 모바일 알림도 설정 가능합니다.

## ✅ 완료!

이제 다음과 같이 사용할 수 있습니다:

### 📱 모바일 (Terminus)
1. Terminus 앱 열기
2. VPS 연결
3. `auroaq` 실행
4. 가로 모드로 회전하여 최적 화면

### 💻 PC (모든 SSH 클라이언트)  
1. SSH 클라이언트로 VPS 연결
2. `auroaq` 실행
3. 풀 해상도로 대시보드 확인

**두 방식 모두 완벽하게 호환되며, 동시 접속도 가능합니다!** 🎉