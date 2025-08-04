#!/usr/bin/env python3
"""
간단한 VPS 배포 검증 스크립트
배포 후 시스템 상태를 빠르게 확인합니다.
"""

import requests
import time
import sys
from datetime import datetime

def test_api_endpoint(url, timeout=10):
    """API 엔드포인트 테스트"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    server_ip = "109.123.239.30"
    base_url = f"http://{server_ip}:8004"
    
    print("🔍 AuroraQ VPS 배포 검증")
    print("=" * 40)
    print(f"서버: {server_ip}")
    print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 테스트할 엔드포인트들
    endpoints = [
        ("/", "메인 페이지"),
        ("/health", "헬스체크"),
        ("/api/system/stats", "시스템 통계"),
        ("/api/trading/status", "트레이딩 상태"),
    ]
    
    success_count = 0
    total_count = len(endpoints)
    
    for endpoint, description in endpoints:
        url = base_url + endpoint
        print(f"테스트: {description} ({endpoint})")
        
        success, result = test_api_endpoint(url)
        
        if success:
            print(f"✅ 성공")
            if isinstance(result, dict):
                if 'status' in result:
                    print(f"   상태: {result['status']}")
                if 'timestamp' in result:
                    print(f"   시간: {result['timestamp']}")
            success_count += 1
        else:
            print(f"❌ 실패: {result}")
        
        print()
    
    # 결과 요약
    print("=" * 40)
    print(f"검증 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트 통과! 시스템이 정상 동작합니다.")
        print(f"접속 주소: {base_url}")
        sys.exit(0)
    else:
        print("⚠️ 일부 테스트 실패. 시스템 상태를 확인하세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()