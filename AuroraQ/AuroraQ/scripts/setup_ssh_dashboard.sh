#!/bin/bash
# AuroraQ SSH Dashboard 자동 설정 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 로그 함수
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 시스템 감지
detect_system() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VERSION=$VERSION_ID
    else
        OS="Unknown"
    fi
    
    log_info "Detected OS: $OS $VERSION"
}

# 패키지 설치
install_packages() {
    log_info "Installing required packages..."
    
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install -y python3 python3-pip screen tmux curl wget
    elif command -v yum &> /dev/null; then
        sudo yum update -y
        sudo yum install -y python3 python3-pip screen tmux curl wget
    elif command -v dnf &> /dev/null; then
        sudo dnf update -y
        sudo dnf install -y python3 python3-pip screen tmux curl wget
    else
        log_error "Unsupported package manager"
        exit 1
    fi
    
    log_success "System packages installed"
}

# Python 라이브러리 설치
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # 사용자 레벨 설치
    pip3 install --user --upgrade pip
    pip3 install --user rich psutil
    
    # 시스템 레벨 백업 설치 (권한이 있는 경우)
    if sudo -n true 2>/dev/null; then
        sudo pip3 install rich psutil
    fi
    
    log_success "Python dependencies installed"
}

# SSH 설정 최적화
optimize_ssh() {
    log_info "Optimizing SSH configuration..."
    
    local ssh_config="$HOME/.ssh/config"
    
    # SSH 클라이언트 설정 생성/업데이트
    mkdir -p ~/.ssh
    
    if [ ! -f "$ssh_config" ] || ! grep -q "ServerAliveInterval" "$ssh_config"; then
        cat >> "$ssh_config" << EOF

# AuroraQ Dashboard optimizations
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    Compression yes

EOF
        log_success "SSH client configuration optimized"
    fi
    
    # SSH 서버 설정 권장사항 표시
    log_info "SSH server optimization recommendations:"
    echo "Add to /etc/ssh/sshd_config:"
    echo "  ClientAliveInterval 60"
    echo "  ClientAliveCountMax 3"
    echo "  TCPKeepAlive yes"
}

# 대시보드 환경 설정
setup_dashboard_env() {
    log_info "Setting up dashboard environment..."
    
    local script_dir="$(dirname "$(realpath "$0")")"
    local project_root="$(dirname "$script_dir")"
    
    # 실행 권한 설정
    chmod +x "$script_dir/dashboard_launcher.sh"
    
    # 환경 변수 설정 스크립트 생성
    cat > "$HOME/.auroaq_env" << EOF
# AuroraQ Dashboard Environment
export AUROAQ_ROOT="$project_root"
export AUROAQ_DASHBOARD_PATH="$project_root/dashboard"
export AUROAQ_SCRIPTS_PATH="$project_root/scripts"
export TERM=xterm-256color
export PYTHONUNBUFFERED=1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
EOF

    # bashrc에 자동 로드 추가
    if ! grep -q ".auroaq_env" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# AuroraQ Dashboard" >> ~/.bashrc
        echo "[ -f ~/.auroaq_env ] && source ~/.auroaq_env" >> ~/.bashrc
    fi
    
    # 현재 세션에 적용
    source ~/.auroaq_env
    
    log_success "Dashboard environment configured"
}

# 자동 시작 서비스 생성
create_autostart_service() {
    log_info "Creating auto-start service..."
    
    local script_dir="$(dirname "$(realpath "$0")")"
    local launcher_path="$script_dir/dashboard_launcher.sh"
    local current_user=$(whoami)
    
    # systemd 사용자 서비스 생성
    mkdir -p ~/.config/systemd/user
    
    cat > ~/.config/systemd/user/auroaq-dashboard.service << EOF
[Unit]
Description=AuroraQ SSH Dashboard Auto-start
After=network.target

[Service]
Type=forking
ExecStart=$launcher_path run
Restart=always
RestartSec=10
Environment=HOME=$HOME
Environment=USER=$current_user
WorkingDirectory=$script_dir

[Install]
WantedBy=default.target
EOF

    # 서비스 활성화
    systemctl --user daemon-reload
    systemctl --user enable auroaq-dashboard.service
    
    log_success "Auto-start service created"
    log_info "Service commands:"
    echo "  Start: systemctl --user start auroaq-dashboard"
    echo "  Stop:  systemctl --user stop auroaq-dashboard"
    echo "  Status: systemctl --user status auroaq-dashboard"
}

# 바로가기 명령어 생성
create_shortcuts() {
    log_info "Creating command shortcuts..."
    
    local script_dir="$(dirname "$(realpath "$0")")"
    local bin_dir="$HOME/.local/bin"
    
    mkdir -p "$bin_dir"
    
    # 메인 런처 바로가기
    cat > "$bin_dir/auroaq" << EOF
#!/bin/bash
cd "$script_dir"
./dashboard_launcher.sh "\$@"
EOF
    
    # 직접 실행 바로가기
    cat > "$bin_dir/auroaq-run" << EOF
#!/bin/bash
cd "$script_dir"
./dashboard_launcher.sh run
EOF
    
    chmod +x "$bin_dir/auroaq" "$bin_dir/auroaq-run"
    
    # PATH에 추가 (없는 경우)
    if ! echo "$PATH" | grep -q "$bin_dir"; then
        echo "" >> ~/.bashrc
        echo "# AuroraQ shortcuts" >> ~/.bashrc
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
    fi
    
    log_success "Command shortcuts created"
    log_info "Available commands:"
    echo "  auroaq      - Interactive dashboard launcher"
    echo "  auroaq-run  - Direct dashboard execution"
}

# SSH 키 설정 도움말
show_ssh_key_help() {
    log_info "SSH Key Setup for Mobile/PC access:"
    echo ""
    echo "1. Generate SSH key (if not exists):"
    echo "   ssh-keygen -t rsa -b 4096 -C \"your_email@example.com\""
    echo ""
    echo "2. Copy public key to server:"
    echo "   ssh-copy-id user@your-vps-ip"
    echo ""
    echo "3. For Terminus (iOS):"
    echo "   - Export private key: cat ~/.ssh/id_rsa"
    echo "   - Import in Terminus app"
    echo ""
    echo "4. For PC SSH clients:"
    echo "   - Use existing private key"
    echo "   - Configure connection: user@your-vps-ip"
}

# 방화벽 설정
setup_firewall() {
    log_info "Configuring firewall..."
    
    if command -v ufw &> /dev/null; then
        sudo ufw allow 22/tcp  # SSH
        log_success "UFW firewall configured for SSH"
    elif command -v firewall-cmd &> /dev/null; then
        sudo firewall-cmd --permanent --add-service=ssh
        sudo firewall-cmd --reload
        log_success "Firewalld configured for SSH"
    else
        log_warning "No supported firewall found. Ensure SSH (port 22) is accessible."
    fi
}

# 테스트 및 검증
run_tests() {
    log_info "Running system tests..."
    
    # Python 가져오기 테스트
    if python3 -c "import rich, psutil" 2>/dev/null; then
        log_success "Python dependencies test passed"
    else
        log_error "Python dependencies test failed"
        exit 1
    fi
    
    # 스크립트 실행 테스트
    local script_dir="$(dirname "$(realpath "$0")")"
    if [ -x "$script_dir/dashboard_launcher.sh" ]; then
        log_success "Dashboard launcher is executable"
    else
        log_error "Dashboard launcher is not executable"
        exit 1
    fi
    
    # 터미널 환경 테스트
    if [ -n "$TERM" ]; then
        log_success "Terminal environment is configured"
    else
        log_warning "Terminal environment may need configuration"
    fi
    
    log_success "All tests passed"
}

# 최종 설정 요약
show_summary() {
    log_success "🎉 AuroraQ SSH Dashboard setup completed!"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📱 Mobile Access (Terminus):"
    echo "  1. Open Terminus app"
    echo "  2. Connect: user@your-vps-ip"
    echo "  3. Run: auroaq"
    echo ""
    echo "💻 PC Access (Any SSH client):"
    echo "  1. SSH connect: ssh user@your-vps-ip"
    echo "  2. Run: auroaq"
    echo ""
    echo "🚀 Quick Commands:"
    echo "  auroaq      - Interactive menu"
    echo "  auroaq-run  - Direct dashboard"
    echo ""
    echo "⚙️ Service Management:"
    echo "  systemctl --user start auroaq-dashboard"
    echo "  systemctl --user status auroaq-dashboard"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    log_info "Logout and login again to apply all changes"
}

# 메인 실행
main() {
    echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     🚀 AuroraQ SSH Dashboard        ║${NC}"
    echo -e "${CYAN}║        Auto Setup Script            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
    echo ""
    
    detect_system
    install_packages
    install_python_deps
    optimize_ssh
    setup_dashboard_env
    create_autostart_service
    create_shortcuts
    setup_firewall
    run_tests
    show_ssh_key_help
    show_summary
}

# 스크립트 실행
main "$@"