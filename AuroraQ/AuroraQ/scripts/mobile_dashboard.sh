#!/bin/bash
# 모바일 최적화 대시보드 실행 스크립트

export TERM=xterm-256color
export COLUMNS=80
export LINES=24

# 모바일 화면 크기에 맞춰 터미널 크기 조정
stty cols 80 rows 24

echo "📱 Mobile Optimized AuroraQ Dashboard"
echo "화면을 가로로 돌려서 보세요!"
echo "Press any key to start..."
read -n 1

# 대시보드 실행
cd "$(dirname "$0")/../dashboard"
python3 aurora_dashboard_final.py