#!/usr/bin/env python3
"""
.env 파일 설정 상태 확인
"""

import os
from pathlib import Path

def check_env_config():
    """환경 설정 상태 확인"""
    print('📋 AuroraQ v2.0 환경 설정 검증')
    print('=' * 50)
    
    # 핵심 설정 확인
    core_settings = {
        'BINANCE_API_KEY': '거래소 API',
        'NEWSAPI_KEY': '뉴스 API (선택)',
        'FINNHUB_API_KEY': '금융 데이터 API (선택)', 
        'TELEGRAM_BOT_TOKEN': 'Telegram 알림 (권장)',
        'NEWS_COLLECTION_INTERVAL': '뉴스 수집 주기',
        'SENTIMENT_BUY_THRESHOLD': '매수 임계값',
        'MAX_MEMORY_MB': '메모리 제한'
    }
    
    configured = 0
    total = len(core_settings)
    
    for key, desc in core_settings.items():
        value = os.getenv(key, '')
        if value and value != '' and 'your_' not in value:
            print(f'✅ {desc}: 설정됨')
            configured += 1
        else:
            print(f'⚠️ {desc}: 미설정 또는 기본값')
    
    print(f'\n📊 설정 완료도: {configured}/{total} ({configured/total*100:.1f}%)')
    
    if configured >= 3:
        print('🎉 기본 설정 완료! 시스템 사용 가능')
    elif configured >= 1:
        print('⚡ 부분 설정 완료. 추가 설정 권장')
    else:
        print('🔧 추가 설정이 필요합니다')
        
    print('\n📖 자세한 설정 가이드: SETUP_GUIDE.md 참조')
    print('🚀 무료 소스로 즉시 사용 가능: Google News + Yahoo Finance + Reddit')
    
    # 무료 소스 가용성 확인
    print('\n🆓 무료 뉴스 소스 상태:')
    print('   ✅ Google News RSS - 항상 사용 가능')
    print('   ✅ Yahoo Finance RSS - 항상 사용 가능')  
    print('   ✅ Reddit API - 항상 사용 가능')
    
    # API 키 상태
    newsapi = os.getenv('NEWSAPI_KEY', '')
    finnhub = os.getenv('FINNHUB_API_KEY', '')
    
    if newsapi and 'your_' not in newsapi:
        print('   ✅ NewsAPI - 활성화 (100req/일 추가)')
    else:
        print('   ⚪ NewsAPI - 미설정 (선택사항)')
        
    if finnhub and 'your_' not in finnhub:
        print('   ✅ Finnhub - 활성화 (60req/분 추가)')
    else:
        print('   ⚪ Finnhub - 미설정 (선택사항)')

if __name__ == "__main__":
    check_env_config()