#!/usr/bin/env python3
"""
실전 API 연결 테스트 스크립트
Binance + Feedly 실제 데이터 수집 및 센티멘트 분석 검증
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from SharedCore.data_layer.unified_data_provider import UnifiedDataProvider
from SharedCore.sentiment_engine.sentiment_aggregator import SentimentAggregator
from SharedCore.data_layer.market_data.binance_collector import create_binance_collector
from SharedCore.sentiment_engine.news_collectors.feedly_collector import create_feedly_collector


class RealDataTester:
    """실전 데이터 연결 테스트"""
    
    def __init__(self):
        self.results = {
            'binance': {'status': 'pending', 'data': None, 'error': None},
            'feedly': {'status': 'pending', 'data': None, 'error': None},
            'unified_provider': {'status': 'pending', 'data': None, 'error': None},
            'sentiment': {'status': 'pending', 'data': None, 'error': None}
        }
    
    async def test_binance_connection(self):
        """Binance API 연결 테스트"""
        print("🔗 Testing Binance API Connection...")
        
        try:
            # 환경 변수 확인
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            if not api_key or api_key == 'your_binance_api_key_here':
                self.results['binance'] = {
                    'status': 'skipped',
                    'error': 'API keys not configured in .env file'
                }
                print("  ⚠️ Binance API keys not configured - using dummy data")
                return
            
            # Binance 수집기 생성 및 연결
            collector = create_binance_collector(api_key, api_secret, testnet)
            await collector.connect()
            
            # 계정 정보 조회
            account_info = await collector.get_account_info()
            
            # 시장 데이터 조회 (BTC/USDT)
            market_data = await collector.get_market_data(
                symbol='BTCUSDT',
                interval='1h',
                limit=10
            )
            
            # 현재 가격 조회
            current_price = await collector.get_ticker_price('BTCUSDT')
            
            # 잔고 조회 (USDT)
            free_balance, locked_balance = await collector.get_balance('USDT')
            
            await collector.close()
            
            self.results['binance'] = {
                'status': 'success',
                'data': {
                    'account_type': account_info.get('accountType', 'Unknown'),
                    'can_trade': account_info.get('canTrade', False),
                    'market_data_points': len(market_data),
                    'current_btc_price': current_price,
                    'usdt_balance': {'free': free_balance, 'locked': locked_balance},
                    'testnet': testnet
                }
            }
            
            print(f"  ✅ Binance connection successful ({'testnet' if testnet else 'mainnet'})")
            print(f"     Account: {account_info.get('accountType', 'Unknown')}")
            print(f"     BTC/USDT Price: ${current_price:,.2f}")
            print(f"     Market data: {len(market_data)} candles")
            print(f"     USDT Balance: {free_balance:.2f}")
            
        except Exception as e:
            self.results['binance'] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"  ❌ Binance connection failed: {e}")
    
    async def test_feedly_connection(self):
        """Feedly API 연결 테스트"""
        print("\n📰 Testing Feedly API Connection...")
        
        try:
            # Feedly 수집기 생성 및 연결
            feedly_token = os.getenv('FEEDLY_ACCESS_TOKEN')
            collector = create_feedly_collector(feedly_token)
            await collector.connect()
            
            # 암호화폐 관련 뉴스 수집
            articles = await collector.get_latest_crypto_news(
                hours_back=12,
                max_articles=10
            )
            
            # 감정 분석 요약
            sentiment_summary = await collector.get_sentiment_summary(articles)
            
            await collector.close()
            
            self.results['feedly'] = {
                'status': 'success',
                'data': {
                    'articles_count': len(articles),
                    'sentiment_summary': sentiment_summary,
                    'sample_titles': [a.title for a in articles[:3]],
                    'sources': list(set(a.source for a in articles[:10]))
                }
            }
            
            print(f"  ✅ Feedly connection successful")
            print(f"     Articles collected: {len(articles)}")
            print(f"     Overall sentiment: {sentiment_summary['overall_sentiment']:.2f}")
            print(f"     Positive ratio: {sentiment_summary['sentiment_distribution']['positive']:.1%}")
            print(f"     Top sources: {', '.join(list(set(a.source for a in articles[:5])))}")
            
        except Exception as e:
            self.results['feedly'] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"  ❌ Feedly connection failed: {e}")
    
    async def test_unified_provider(self):
        """UnifiedDataProvider 통합 테스트"""
        print("\n🔄 Testing Unified Data Provider...")
        
        try:
            # AuroraQ 모드로 데이터 프로바이더 초기화
            provider = UnifiedDataProvider(
                use_crypto=True,
                use_macro=False
            )
            await provider.connect()
            
            # 암호화폐 시장 데이터 조회
            market_data = await provider.get_market_data(
                asset_type="crypto",
                symbol="BTC/USDT",
                timeframe="1h"
            )
            
            # 감정 점수 조회
            sentiment_data = await provider.get_sentiment_score("BTC")
            
            await provider.close()
            
            self.results['unified_provider'] = {
                'status': 'success',
                'data': {
                    'market_data_points': len(market_data),
                    'latest_price': float(market_data['close'].iloc[-1]) if not market_data.empty else 0,
                    'sentiment_overall': sentiment_data.get('overall', 0),
                    'loaded_types': list(provider.loaded_data_types)
                }
            }
            
            print(f"  ✅ Unified provider working")
            print(f"     Data points: {len(market_data)}")
            print(f"     Latest price: ${float(market_data['close'].iloc[-1]):,.2f}" if not market_data.empty else "     No price data")
            print(f"     Sentiment: {sentiment_data.get('overall', 0):.2f}")
            print(f"     Loaded types: {', '.join(provider.loaded_data_types)}")
            
        except Exception as e:
            self.results['unified_provider'] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"  ❌ Unified provider failed: {e}")
    
    async def test_sentiment_aggregator(self):
        """SentimentAggregator 통합 테스트"""
        print("\n🧠 Testing Sentiment Aggregator...")
        
        try:
            # 감정 분석 엔진 초기화
            aggregator = SentimentAggregator()
            
            # 실시간 감정 분석
            real_time_sentiment = await aggregator.get_real_time_sentiment("BTC")
            
            # 통합 감정 점수
            aggregate_sentiment = await aggregator.aggregate_sentiment(
                asset="BTC",
                lookback_hours=12
            )
            
            await aggregator.close()
            
            self.results['sentiment'] = {
                'status': 'success',
                'data': {
                    'real_time': real_time_sentiment,
                    'aggregate': aggregate_sentiment
                }
            }
            
            print(f"  ✅ Sentiment aggregator working")
            print(f"     Real-time sentiment: {real_time_sentiment['sentiment_score']:.2f}")
            print(f"     Confidence: {real_time_sentiment['confidence']:.2f}")
            print(f"     Articles analyzed: {real_time_sentiment['article_count']}")
            print(f"     Aggregate sentiment: {aggregate_sentiment['overall']:.2f}")
            print(f"     Trend: {aggregate_sentiment['trend']}")
            
        except Exception as e:
            self.results['sentiment'] = {
                'status': 'failed',
                'error': str(e)
            }
            print(f"  ❌ Sentiment aggregator failed: {e}")
    
    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed_tests = sum(1 for r in self.results.values() if r['status'] == 'failed')
        skipped_tests = sum(1 for r in self.results.values() if r['status'] == 'skipped')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️ Skipped: {skipped_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status_icon = {
                'success': '✅',
                'failed': '❌',
                'skipped': '⚠️',
                'pending': '⏳'
            }[result['status']]
            
            print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result['status']}")
            if result['error']:
                print(f"     Error: {result['error']}")
        
        # 권장사항
        print("\n💡 Recommendations:")
        
        if self.results['binance']['status'] == 'skipped':
            print("  • Configure Binance API keys in .env file for real trading data")
            print("  • Get testnet keys from https://testnet.binance.vision/")
        
        if self.results['feedly']['status'] == 'failed':
            print("  • Feedly works without token but with rate limits")
            print("  • Get free token from https://developer.feedly.com/")
        
        if failed_tests == 0 and skipped_tests == 0:
            print("  🎉 All systems ready for production!")
        elif passed_tests >= 2:
            print("  ✨ Core systems functional - ready for testing")
        else:
            print("  🔧 Check API configurations and network connectivity")


async def main():
    """메인 테스트 실행"""
    print("🚀 AuroraQ Real Connections Test Suite")
    print("Testing live API connections and data collection")
    print("="*60)
    
    # 환경 변수 로드 체크
    from dotenv import load_dotenv
    load_dotenv()
    
    tester = RealDataTester()
    
    # 테스트 실행
    await tester.test_binance_connection()
    await tester.test_feedly_connection()
    await tester.test_unified_provider()
    await tester.test_sentiment_aggregator()
    
    # 결과 요약
    tester.print_summary()
    
    print("\n🏁 Test suite completed!")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())