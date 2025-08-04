# 📱 AuroraQ 모바일 접근 가이드

## 🎯 목표
VPS에서 실행되는 AuroraQ 대시보드를 핸드폰으로 언제든 확인하고 조작하기

## 🚀 설치 방법

### 1단계: VPS에 파일 업로드
```bash
# 로컬에서 VPS로 전체 폴더 업로드
scp -r vps-deployment/ user@your-vps-ip:/home/user/auroaq/
```

### 2단계: VPS에서 설치 실행
```bash
# VPS에 SSH 접속
ssh user@your-vps-ip

# 설치 스크립트 실행
cd /home/user/auroaq
chmod +x vps-deployment/scripts/setup_web_terminal.sh
./vps-deployment/scripts/setup_web_terminal.sh
```

### 3단계: 서비스 시작
```bash
# HTTP 버전 시작
sudo systemctl start auroaq-dashboard
sudo systemctl enable auroaq-dashboard

# 상태 확인
sudo systemctl status auroaq-dashboard
```

## 📱 핸드폰 접근 방법

### 웹 브라우저 접근
1. **Chrome/Safari 실행**
2. **주소창에 입력**: `http://VPS_IP:7681`
3. **화면을 가로로 회전** (더 잘 보임)
4. **풀스크린 모드** 사용

### 접근 URL 예시
- HTTP: `http://123.456.789.012:7681`
- HTTPS: `https://yourdomain.com:7681` (도메인 있는 경우)

## ⌨️ 모바일에서 키보드 조작

### 터치 키보드 사용
- **위/아래 화살표**: 메뉴 이동
- **Enter**: 선택
- **q**: 종료
- **r**: 새로고침

### 모바일 앱 추천
- **Android**: 
  - JuiceSSH (SSH 클라이언트)
  - Termux (터미널 에뮬레이터)
- **iOS**: 
  - Terminus (SSH 클라이언트)
  - Prompt 3

## 🔒 보안 설정

### 방화벽 설정
```bash
# 포트 7681만 열기
sudo ufw allow 7681/tcp
sudo ufw enable
```

### SSL 인증서 (권장)
```bash
# Let's Encrypt 인증서 설치
sudo certbot certonly --standalone -d yourdomain.com

# HTTPS 서비스 시작
sudo systemctl start auroaq-dashboard-ssl
```

### IP 제한 (선택사항)
```bash
# 특정 IP만 접근 허용
sudo ufw allow from YOUR_PHONE_IP to any port 7681
```

## 📊 실시간 알림 설정

### Telegram Bot 설정
1. **@BotFather**에게 `/newbot` 메시지 전송
2. **봇 토큰** 받기
3. **본인 Chat ID** 확인 (`@userinfobot` 사용)

### 알림 스크립트 설정
```bash
# mobile_notifications.py 편집
BOT_TOKEN = "your_bot_token_here"
CHAT_ID = "your_chat_id_here"

# 알림 테스트
python3 vps-deployment/scripts/mobile_notifications.py
```

## 🔧 Docker 사용 (선택사항)

### Docker Compose로 실행
```bash
cd vps-deployment/docker
docker-compose -f docker-compose.webterminal.yml up -d

# 상태 확인
docker-compose ps
```

## 🚨 문제 해결

### 접속이 안 될 때
1. **방화벽 확인**: `sudo ufw status`
2. **서비스 상태**: `sudo systemctl status auroaq-dashboard`
3. **포트 확인**: `netstat -tlnp | grep 7681`

### 화면이 깨질 때
1. **화면을 가로로 회전**
2. **브라우저 풀스크린 모드**
3. **터미널 크기 조정**: `stty cols 80 rows 24`

### 성능이 느릴 때
1. **VPS 리소스 확인**: `htop`
2. **네트워크 지연**: `ping your-vps-ip`
3. **가까운 VPS 지역** 사용 권장

## 💡 사용 팁

### 효율적인 사용법
- **즐겨찾기에 추가**: 빠른 접근
- **홈 화면에 바로가기**: PWA처럼 사용
- **알림 설정**: 중요한 거래 알림

### 배터리 절약
- **화면 밝기 조절**
- **불필요할 때 연결 해제**
- **WiFi 사용 권장**

## 🎯 완료!

이제 언제 어디서든 핸드폰으로 AuroraQ 대시보드를 확인하고 거래를 모니터링할 수 있습니다!

**접속 URL**: `http://YOUR_VPS_IP:7681`