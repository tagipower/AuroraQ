#!/bin/bash
# VPS Web Terminal 설정 스크립트

echo "🚀 AuroraQ Web Terminal 설정"

# ttyd 설치 (Ubuntu/Debian)
install_ttyd() {
    echo "📦 ttyd 설치 중..."
    
    # GitHub에서 최신 바이너리 다운로드
    TTYD_VERSION="1.7.3"
    wget -O /tmp/ttyd https://github.com/tsl0922/ttyd/releases/download/${TTYD_VERSION}/ttyd.x86_64
    chmod +x /tmp/ttyd
    sudo mv /tmp/ttyd /usr/local/bin/
    
    echo "✅ ttyd 설치 완료"
}

# SSL 인증서 설정 (Let's Encrypt)
setup_ssl() {
    echo "🔒 SSL 인증서 설정..."
    
    # certbot 설치
    sudo apt update
    sudo apt install -y certbot
    
    # 도메인이 있는 경우
    read -p "도메인이 있나요? (y/n): " has_domain
    if [ "$has_domain" = "y" ]; then
        read -p "도메인을 입력하세요: " domain
        sudo certbot certonly --standalone -d $domain
        CERT_PATH="/etc/letsencrypt/live/$domain"
        echo "DOMAIN=$domain" > /tmp/ttyd_config
        echo "CERT_PATH=$CERT_PATH" >> /tmp/ttyd_config
    else
        echo "⚠️ SSL 없이 진행 (HTTP만 사용)"
    fi
}

# 방화벽 설정
setup_firewall() {
    echo "🛡️ 방화벽 설정..."
    
    # ufw 사용
    sudo ufw allow 7681/tcp  # ttyd 기본 포트
    sudo ufw allow 8080/tcp  # 대안 포트
    
    echo "✅ 방화벽 설정 완료"
}

# 자동 시작 서비스 생성
create_service() {
    echo "⚙️ 시스템 서비스 생성..."
    
    # 현재 사용자 정보
    CURRENT_USER=$(whoami)
    DASHBOARD_PATH="$(pwd)/vps-deployment/dashboard"
    
    cat > /tmp/auroaq-dashboard.service << EOF
[Unit]
Description=AuroraQ Web Dashboard
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$DASHBOARD_PATH
ExecStart=/usr/local/bin/ttyd -p 7681 -i 0.0.0.0 --writable python3 aurora_dashboard_final.py
Restart=always
RestartSec=10
Environment=TERM=xterm-256color

[Install]
WantedBy=multi-user.target
EOF

    sudo mv /tmp/auroaq-dashboard.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable auroaq-dashboard
    
    echo "✅ 서비스 생성 완료"
}

# SSL이 있는 경우 HTTPS 서비스
create_ssl_service() {
    if [ -f "/tmp/ttyd_config" ]; then
        source /tmp/ttyd_config
        
        cat > /tmp/auroaq-dashboard-ssl.service << EOF
[Unit]
Description=AuroraQ Web Dashboard (HTTPS)
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$DASHBOARD_PATH
ExecStart=/usr/local/bin/ttyd -p 7681 -i 0.0.0.0 --writable --ssl --ssl-cert $CERT_PATH/fullchain.pem --ssl-key $CERT_PATH/privkey.pem python3 aurora_dashboard_final.py
Restart=always
RestartSec=10
Environment=TERM=xterm-256color

[Install]
WantedBy=multi-user.target
EOF

        sudo mv /tmp/auroaq-dashboard-ssl.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable auroaq-dashboard-ssl
        
        echo "✅ HTTPS 서비스 생성 완료"
    fi
}

# 메인 실행
main() {
    echo "======================================"
    echo "🚀 AuroraQ Web Terminal 설치 시작"
    echo "======================================"
    
    install_ttyd
    setup_ssl
    setup_firewall
    create_service
    create_ssl_service
    
    echo ""
    echo "======================================"
    echo "✅ 설치 완료!"
    echo "======================================"
    echo ""
    echo "📱 모바일 접근 방법:"
    echo "1. HTTP:  http://VPS_IP:7681"
    echo "2. HTTPS: https://도메인:7681 (SSL 설정시)"
    echo ""
    echo "🚀 서비스 시작:"
    echo "sudo systemctl start auroaq-dashboard"
    echo ""
    echo "📊 상태 확인:"
    echo "sudo systemctl status auroaq-dashboard"
    echo ""
}

main