#!/bin/bash
# AuroraQ Systemd Services Setup Script
# 실거래 환경용 서비스 등록 및 설정

set -e

# 색깔 출력 함수
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
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

# 설정 변수
AUROAQ_USER="auroaq"
AUROAQ_GROUP="auroaq" 
AUROAQ_HOME="/opt/AuroraQ"
SYSTEMD_DIR="/etc/systemd/system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Root 권한 확인
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# 사용자 및 그룹 생성
create_user_group() {
    log_info "Creating AuroraQ user and group..."
    
    # 그룹 생성
    if ! getent group "$AUROAQ_GROUP" > /dev/null 2>&1; then
        groupadd --system "$AUROAQ_GROUP"
        log_success "Created group: $AUROAQ_GROUP"
    else
        log_info "Group $AUROAQ_GROUP already exists"
    fi
    
    # 사용자 생성
    if ! getent passwd "$AUROAQ_USER" > /dev/null 2>&1; then
        useradd --system --gid "$AUROAQ_GROUP" --home-dir "$AUROAQ_HOME" \
                --shell /bin/bash --comment "AuroraQ Trading System" "$AUROAQ_USER"
        log_success "Created user: $AUROAQ_USER"
    else
        log_info "User $AUROAQ_USER already exists"
    fi
}

# 디렉토리 설정
setup_directories() {
    log_info "Setting up directories..."
    
    # 필요한 디렉토리들
    directories=(
        "$AUROAQ_HOME"
        "$AUROAQ_HOME/logs"
        "$AUROAQ_HOME/backups"
        "$AUROAQ_HOME/SharedCore/data_storage"
        "$AUROAQ_HOME/sentiment-service/logs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        fi
        
        # 권한 설정
        chown -R "$AUROAQ_USER:$AUROAQ_GROUP" "$dir"
        chmod 755 "$dir"
    done
    
    # AuroraQ 홈 디렉토리 권한 설정
    chown -R "$AUROAQ_USER:$AUROAQ_GROUP" "$AUROAQ_HOME"
    
    # 로그 및 백업 디렉토리는 쓰기 권한 추가
    chmod 775 "$AUROAQ_HOME/logs" "$AUROAQ_HOME/backups"
}

# 서비스 파일 설치
install_service_files() {
    log_info "Installing systemd service files..."
    
    service_files=(
        "auroaq-sentiment.service"
        "auroaq-trading.service"
        "auroaq-backup.service"
        "auroaq-backup.timer"
        "auroaq-monitor.service"
    )
    
    for service_file in "${service_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/systemd/$service_file" ]]; then
            cp "$SCRIPT_DIR/systemd/$service_file" "$SYSTEMD_DIR/"
            chmod 644 "$SYSTEMD_DIR/$service_file"
            log_success "Installed: $service_file"
        else
            log_error "Service file not found: $service_file"
            exit 1
        fi
    done
}

# 환경 파일 생성
create_environment_file() {
    log_info "Creating environment file..."
    
    env_file="$AUROAQ_HOME/.env"
    
    if [[ ! -f "$env_file" ]]; then
        cat > "$env_file" << 'EOF'
# AuroraQ Production Environment Variables
# Copy this file and fill in your actual values

# Binance API (Required)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Notifications (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Database Settings
DATABASE_URL=sqlite:///SharedCore/data_storage/auroaq.db

# Logging Level
LOG_LEVEL=INFO

# Trading Settings
TRADING_MODE=live
RISK_LEVEL=medium
MAX_POSITION_SIZE=0.25

# Service URLs
SENTIMENT_SERVICE_URL=http://localhost:8000
EOF
        
        chown "$AUROAQ_USER:$AUROAQ_GROUP" "$env_file"
        chmod 600 "$env_file"  # 보안을 위해 소유자만 읽기 가능
        
        log_success "Created environment file: $env_file"
        log_warning "Please edit $env_file and add your API keys and configuration"
    else
        log_info "Environment file already exists: $env_file"
    fi
}

# 가상환경 설정
setup_python_venv() {
    log_info "Setting up Python virtual environment..."
    
    venv_dir="$AUROAQ_HOME/venv"
    
    if [[ ! -d "$venv_dir" ]]; then
        # Python3과 venv 모듈 확인
        if ! command -v python3 &> /dev/null; then
            log_error "Python3 is not installed"
            exit 1
        fi
        
        # 가상환경 생성
        sudo -u "$AUROAQ_USER" python3 -m venv "$venv_dir"
        log_success "Created Python virtual environment: $venv_dir"
        
        # 기본 패키지 설치
        sudo -u "$AUROAQ_USER" "$venv_dir/bin/pip" install --upgrade pip
        
        # requirements.txt가 있다면 설치
        if [[ -f "$AUROAQ_HOME/requirements.txt" ]]; then
            log_info "Installing Python dependencies..."
            sudo -u "$AUROAQ_USER" "$venv_dir/bin/pip" install -r "$AUROAQ_HOME/requirements.txt"
            log_success "Python dependencies installed"
        fi
    else
        log_info "Python virtual environment already exists: $venv_dir"
    fi
}

# Docker 권한 설정
setup_docker_permissions() {
    log_info "Setting up Docker permissions..."
    
    # Docker 그룹에 사용자 추가
    if command -v docker &> /dev/null; then
        usermod -aG docker "$AUROAQ_USER"
        log_success "Added $AUROAQ_USER to docker group"
        log_warning "User needs to log out and back in for docker group membership to take effect"
    else
        log_warning "Docker is not installed - install Docker if you plan to use sentiment service"
    fi
}

# Systemd 서비스 활성화
enable_services() {
    log_info "Enabling systemd services..."
    
    # 시스템 데몬 리로드
    systemctl daemon-reload
    
    # 서비스 활성화
    services_to_enable=(
        "auroaq-sentiment.service"
        "auroaq-backup.timer"
        "auroaq-monitor.service"
    )
    
    for service in "${services_to_enable[@]}"; do
        systemctl enable "$service"
        log_success "Enabled: $service"
    done
    
    # 트레이딩 서비스는 수동으로 시작하도록 권장
    log_warning "auroaq-trading.service is not auto-enabled for safety"
    log_info "To enable auto-start: sudo systemctl enable auroaq-trading.service"
}

# 서비스 상태 확인
check_services() {
    log_info "Checking service status..."
    
    services=(
        "auroaq-sentiment.service"
        "auroaq-trading.service"
        "auroaq-backup.service"
        "auroaq-backup.timer"
        "auroaq-monitor.service"
    )
    
    for service in "${services[@]}"; do
        if systemctl is-enabled "$service" &> /dev/null; then
            status="enabled"
        else
            status="disabled"
        fi
        
        if systemctl is-active "$service" &> /dev/null; then
            active="active"
            icon="✅"
        else
            active="inactive"
            icon="⭕"
        fi
        
        echo -e "$icon $service: $status, $active"
    done
}

# 방화벽 설정
setup_firewall() {
    log_info "Configuring firewall..."
    
    if command -v ufw &> /dev/null; then
        # UFW가 활성화되어있는 경우에만 설정
        if ufw status | grep -q "Status: active"; then
            # Sentiment service port
            ufw allow 8000/tcp comment "AuroraQ Sentiment Service"
            log_success "Firewall rule added for port 8000"
        else
            log_info "UFW is not active, skipping firewall configuration"
        fi
    elif command -v firewall-cmd &> /dev/null; then
        # Firewalld 설정
        firewall-cmd --permanent --add-port=8000/tcp
        firewall-cmd --reload
        log_success "Firewall rule added for port 8000 (firewalld)"
    else
        log_warning "No firewall management tool found (ufw/firewalld)"
    fi
}

# 메인 설치 함수
main() {
    log_info "Starting AuroraQ Systemd Services Setup..."
    log_info "================================================"
    
    # Root 권한 확인
    check_root
    
    # 설치 단계들
    create_user_group
    setup_directories
    install_service_files
    create_environment_file
    setup_python_venv
    setup_docker_permissions
    setup_firewall
    enable_services
    
    log_info ""
    log_info "================================================"
    log_success "AuroraQ Systemd Services Setup Complete!"
    log_info "================================================"
    
    echo ""
    log_info "Next steps:"
    echo "1. Edit $AUROAQ_HOME/.env and add your API keys"
    echo "2. Copy your AuroraQ code to $AUROAQ_HOME"
    echo "3. Install Python dependencies if needed"
    echo "4. Start services:"
    echo "   sudo systemctl start auroaq-sentiment"
    echo "   sudo systemctl start auroaq-monitor"
    echo "   sudo systemctl start auroaq-trading  # When ready"
    echo ""
    echo "Service management commands:"
    echo "   sudo systemctl status auroaq-sentiment"
    echo "   sudo systemctl logs -f auroaq-trading"
    echo "   sudo systemctl restart auroaq-monitor"
    echo ""
    
    log_info "Current service status:"
    check_services
}

# 옵션 처리
case "${1:-install}" in
    "install")
        main
        ;;
    "status")
        check_services
        ;;
    "uninstall")
        log_info "Uninstalling AuroraQ systemd services..."
        
        # 서비스 중지 및 비활성화
        for service in auroaq-sentiment auroaq-trading auroaq-backup auroaq-monitor; do
            systemctl stop "${service}.service" 2>/dev/null || true
            systemctl disable "${service}.service" 2>/dev/null || true
        done
        
        systemctl stop auroaq-backup.timer 2>/dev/null || true
        systemctl disable auroaq-backup.timer 2>/dev/null || true
        
        # 서비스 파일 삭제
        rm -f /etc/systemd/system/auroaq-*.service
        rm -f /etc/systemd/system/auroaq-*.timer
        
        systemctl daemon-reload
        
        log_success "AuroraQ systemd services uninstalled"
        log_warning "User and directories were not removed. Remove manually if needed:"
        echo "   sudo userdel $AUROAQ_USER"
        echo "   sudo rm -rf $AUROAQ_HOME"
        ;;
    "help"|"-h"|"--help")
        echo "AuroraQ Systemd Services Setup"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  install     Install and configure services (default)"
        echo "  status      Show service status"
        echo "  uninstall   Remove services"
        echo "  help        Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac