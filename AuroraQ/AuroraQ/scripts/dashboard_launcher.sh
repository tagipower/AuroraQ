#!/bin/bash
# AuroraQ Dashboard SSH Launcher
# PC/Mobile SSH 클라이언트 완벽 호환

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 터미널 환경 감지
detect_terminal() {
    local width=$(tput cols 2>/dev/null || echo 80)
    local height=$(tput lines 2>/dev/null || echo 24)
    
    # 모바일 감지 (작은 화면)
    if [ "$width" -lt 90 ] || [ "$height" -lt 25 ]; then
        echo "mobile"
    else
        echo "desktop"
    fi
}

# 환경 설정
setup_environment() {
    export TERM=xterm-256color
    export PYTHONUNBUFFERED=1
    export LANG=en_US.UTF-8
    export LC_ALL=en_US.UTF-8
    
    # 터미널 크기 설정
    local terminal_type=$(detect_terminal)
    
    if [ "$terminal_type" = "mobile" ]; then
        # 모바일 최적화 (Terminus)
        export COLUMNS=80
        export LINES=24
        stty cols 80 rows 24 2>/dev/null || true
        echo -e "${CYAN}📱 Mobile mode detected (Terminus optimized)${NC}"
    else
        # PC 최적화
        export COLUMNS=120
        export LINES=35
        stty cols 120 rows 35 2>/dev/null || true
        echo -e "${BLUE}💻 Desktop mode detected${NC}"
    fi
}

# 대시보드 상태 확인
check_dashboard_status() {
    local dashboard_path="$1"
    
    if [ ! -f "$dashboard_path/aurora_dashboard_final.py" ]; then
        echo -e "${RED}❌ Dashboard file not found: $dashboard_path/aurora_dashboard_final.py${NC}"
        return 1
    fi
    
    # Python 및 필수 라이브러리 확인
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 not found${NC}"
        return 1
    fi
    
    # Rich 라이브러리 확인
    if ! python3 -c "import rich" 2>/dev/null; then
        echo -e "${YELLOW}⚠️ Installing required libraries...${NC}"
        pip3 install rich psutil --user
    fi
    
    echo -e "${GREEN}✅ Dashboard ready${NC}"
    return 0
}

# 세션 관리
manage_sessions() {
    local action="$1"
    local session_name="auroaq_dashboard"
    
    case "$action" in
        "check")
            if command -v screen &> /dev/null; then
                screen -list | grep -q "$session_name" && echo "running" || echo "stopped"
            elif command -v tmux &> /dev/null; then
                tmux list-sessions 2>/dev/null | grep -q "$session_name" && echo "running" || echo "stopped"
            else
                echo "no_multiplexer"
            fi
            ;;
        "start")
            if command -v screen &> /dev/null; then
                screen -dmS "$session_name" bash -c "cd $(pwd) && ./dashboard_launcher.sh run"
                echo -e "${GREEN}✅ Dashboard started in background (screen)${NC}"
            elif command -v tmux &> /dev/null; then
                tmux new-session -d -s "$session_name" "cd $(pwd) && ./dashboard_launcher.sh run"
                echo -e "${GREEN}✅ Dashboard started in background (tmux)${NC}"
            else
                echo -e "${YELLOW}⚠️ No screen/tmux available. Running in foreground...${NC}"
                ./dashboard_launcher.sh run
            fi
            ;;
        "attach")
            if command -v screen &> /dev/null && screen -list | grep -q "$session_name"; then
                screen -r "$session_name"
            elif command -v tmux &> /dev/null && tmux list-sessions 2>/dev/null | grep -q "$session_name"; then
                tmux attach-session -t "$session_name"
            else
                echo -e "${RED}❌ No active dashboard session found${NC}"
                return 1
            fi
            ;;
        "stop")
            if command -v screen &> /dev/null; then
                screen -S "$session_name" -X quit 2>/dev/null
            fi
            if command -v tmux &> /dev/null; then
                tmux kill-session -t "$session_name" 2>/dev/null
            fi
            echo -e "${GREEN}✅ Dashboard sessions stopped${NC}"
            ;;
    esac
}

# 메인 메뉴
show_menu() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║        🚀 AuroraQ Dashboard         ║${NC}"
    echo -e "${CYAN}║     SSH Compatible Launcher         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Terminal: $(detect_terminal) | Size: $(tput cols)x$(tput lines)${NC}"
    echo ""
    echo -e "${GREEN}1.${NC} 📊 Run Dashboard (Foreground)"
    echo -e "${GREEN}2.${NC} 🔄 Run Dashboard (Background)"
    echo -e "${GREEN}3.${NC} 📋 Attach to Background Session"
    echo -e "${GREEN}4.${NC} ⏹️  Stop Background Session"
    echo -e "${GREEN}5.${NC} 🔍 Check System Status"
    echo -e "${GREEN}6.${NC} ⚙️  Install/Update Dependencies"
    echo -e "${GREEN}q.${NC} 🚪 Exit"
    echo ""
    
    # 세션 상태 표시
    local session_status=$(manage_sessions "check")
    if [ "$session_status" = "running" ]; then
        echo -e "${GREEN}🟢 Background session: RUNNING${NC}"
    else
        echo -e "${RED}🔴 Background session: STOPPED${NC}"
    fi
    echo ""
}

# 대시보드 실행
run_dashboard() {
    local dashboard_path="$(dirname "$(realpath "$0")")/../dashboard"
    
    echo -e "${CYAN}🚀 Starting AuroraQ Dashboard...${NC}"
    echo -e "${YELLOW}📱 Mobile users: Rotate to landscape for better view${NC}"
    echo -e "${YELLOW}⌨️  Controls: ↑↓ = Navigate, Enter = Select, q = Quit${NC}"
    echo ""
    echo -e "${GREEN}Press any key to continue...${NC}"
    read -n 1 -s
    
    # 대시보드 실행
    cd "$dashboard_path"
    python3 aurora_dashboard_final.py
}

# 시스템 상태 확인
check_system() {
    echo -e "${CYAN}🔍 System Status Check${NC}"
    echo "================================"
    
    # Python 버전
    echo -n "Python3: "
    if command -v python3 &> /dev/null; then
        python3 --version
    else
        echo -e "${RED}Not installed${NC}"
    fi
    
    # 필수 라이브러리
    echo -n "Rich library: "
    if python3 -c "import rich" 2>/dev/null; then
        echo -e "${GREEN}Installed${NC}"
    else
        echo -e "${RED}Not installed${NC}"
    fi
    
    echo -n "psutil library: "
    if python3 -c "import psutil" 2>/dev/null; then
        echo -e "${GREEN}Installed${NC}"
    else
        echo -e "${RED}Not installed${NC}"
    fi
    
    # 터미널 멀티플렉서
    echo -n "Screen: "
    if command -v screen &> /dev/null; then
        echo -e "${GREEN}Available${NC}"
    else
        echo -e "${YELLOW}Not available${NC}"
    fi
    
    echo -n "Tmux: "
    if command -v tmux &> /dev/null; then
        echo -e "${GREEN}Available${NC}"
    else
        echo -e "${YELLOW}Not available${NC}"
    fi
    
    # 시스템 리소스
    echo ""
    echo "System Resources:"
    echo "CPU: $(grep 'cpu ' /proc/stat | awk '{usage=($2+$4)*100/($2+$3+$4+$5)} END {print usage "%"}')"
    echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2*100.0)}')"
    echo "Disk: $(df -h / | awk 'NR==2{print $5}')"
    
    echo ""
    echo -e "${GREEN}Press any key to continue...${NC}"
    read -n 1 -s
}

# 의존성 설치
install_dependencies() {
    echo -e "${CYAN}📦 Installing/Updating Dependencies${NC}"
    echo "=================================="
    
    # 패키지 관리자 감지
    if command -v apt &> /dev/null; then
        echo "Updating system packages..."
        sudo apt update
        sudo apt install -y python3 python3-pip screen tmux
    elif command -v yum &> /dev/null; then
        echo "Updating system packages..."
        sudo yum update -y
        sudo yum install -y python3 python3-pip screen tmux
    fi
    
    # Python 라이브러리 설치
    echo "Installing Python libraries..."
    pip3 install --user --upgrade rich psutil
    
    echo -e "${GREEN}✅ Dependencies installation completed${NC}"
    echo ""
    echo -e "${GREEN}Press any key to continue...${NC}"
    read -n 1 -s
}

# 메인 함수
main() {
    # 환경 설정
    setup_environment
    
    # 인자가 있으면 직접 실행
    if [ "$1" = "run" ]; then
        run_dashboard
        return
    fi
    
    # 대시보드 경로 확인
    local dashboard_path="$(dirname "$(realpath "$0")")/../dashboard"
    if ! check_dashboard_status "$dashboard_path"; then
        echo -e "${RED}❌ Dashboard setup incomplete. Please check installation.${NC}"
        exit 1
    fi
    
    # 메인 루프
    while true; do
        show_menu
        read -p "Select option: " choice
        
        case $choice in
            1)
                run_dashboard
                ;;
            2)
                manage_sessions "start"
                sleep 2
                ;;
            3)
                manage_sessions "attach"
                ;;
            4)
                manage_sessions "stop"
                sleep 1
                ;;
            5)
                check_system
                ;;
            6)
                install_dependencies
                ;;
            q|Q)
                echo -e "${GREEN}👋 Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Invalid option${NC}"
                sleep 1
                ;;
        esac
    done
}

# 스크립트 실행
main "$@"