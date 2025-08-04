@echo off
REM AuroraQ VPS 배포 - Windows 빠른 시작 스크립트
chcp 65001 > nul

echo.
echo ==========================================
echo AuroraQ VPS 배포 - 빠른 시작
echo ==========================================
echo.

set VPS_HOST=109.123.239.30
set VPS_USER=root

echo VPS 정보:
echo - 호스트: %VPS_HOST%
echo - 사용자: %VPS_USER%
echo.

REM SSH 연결 테스트
echo 1. SSH 연결 테스트...
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null %VPS_USER%@%VPS_HOST% "echo 'SSH 연결 성공'"

if errorlevel 1 (
    echo [ERROR] SSH 연결 실패
    echo 해결 방법:
    echo 1. VPS 서버가 실행 중인지 확인
    echo 2. SSH 키 설정 또는 비밀번호 확인
    echo 3. 방화벽에서 SSH 포트(22) 열기
    pause
    exit /b 1
)

echo.
echo 2. 배포 스크립트 전송 및 실행...

REM 배포 스크립트를 VPS로 전송
scp -o StrictHostKeyChecking=no deploy_vps.sh %VPS_USER%@%VPS_HOST%:~/

REM VPS에서 배포 스크립트 실행
ssh -o StrictHostKeyChecking=no %VPS_USER%@%VPS_HOST% "chmod +x ~/deploy_vps.sh && sudo ~/deploy_vps.sh"

echo.
echo 3. 배포 완료 확인...
ssh -o StrictHostKeyChecking=no %VPS_USER%@%VPS_HOST% "curl -s http://localhost:8004/health | head -10"

echo.
echo ==========================================
echo 배포 완료!
echo ==========================================
echo.
echo 접속 주소:
echo   API 서버: http://%VPS_HOST%:8004
echo   헬스체크: http://%VPS_HOST%:8004/health
echo   API 문서: http://%VPS_HOST%:8004/docs
echo.

pause