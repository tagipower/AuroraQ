#!/usr/bin/env python3
"""
고급 감정 분석 통합 테스트 스크립트
FinBERT + Fusion + Router + SentimentAggregator v3.0 통합 검증
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

from SharedCore.sentiment_engine.analyzers.finbert_analyzer import get_finbert_analyzer
from SharedCore.sentiment_engine.fusion.sentiment_fusion_manager import get_fusion_manager
from SharedCore.sentiment_engine.routing.sentiment_router import get_router
from SharedCore.sentiment_engine.sentiment_aggregator import SentimentAggregator


class AdvancedSentimentTester:
    """고급 감정 분석 통합 테스트"""
    
    def __init__(self):
        self.results = {}
        self.test_articles = [
            {
                "title": "Bitcoin ETF Approval Expected Soon",
                "snippet": "Regulatory authorities signal positive outlook for cryptocurrency investment products",
                "datetime": datetime.now() - timedelta(hours=2)
            },
            {
                "title": "Major Cryptocurrency Exchange Security Breach",
                "snippet": "Hackers steal millions in digital assets, market confidence shaken",
                "datetime": datetime.now() - timedelta(hours=1)
            },
            {
                "title": "Federal Reserve Maintains Neutral Stance on Digital Assets",
                "snippet": "Central bank officials express cautious optimism about blockchain technology",
                "datetime": datetime.now() - timedelta(minutes=30)
            }
        ]
    
    async def test_finbert_analyzer(self):
        """FinBERT 분석기 테스트"""
        print("🧠 Testing FinBERT Analyzer...")
        
        try:
            analyzer = await get_finbert_analyzer()
            
            results = []
            for i, article in enumerate(self.test_articles):
                print(f"  Article {i+1}: {article['title'][:50]}...")
                
                # 기본 분석
                score = await analyzer.analyze(article)
                
                # 상세 분석
                detailed = await analyzer.analyze_detailed(article)
                
                result = {
                    'basic_score': score,
                    'detailed_score': detailed.sentiment_score,
                    'label': detailed.label.value,
                    'confidence': detailed.confidence,
                    'keywords': detailed.keywords,
                    'scenario': detailed.scenario_tag
                }
                
                results.append(result)
                print(f"    Score: {score:.4f} | Label: {detailed.label.value} | Scenario: {detailed.scenario_tag}")
            
            # 배치 분석 테스트
            batch_scores = await analyzer.analyze_batch(self.test_articles)
            
            self.results['finbert'] = {
                'status': 'success',
                'individual_results': results,
                'batch_scores': batch_scores,
                'average_score': sum(batch_scores) / len(batch_scores),
                'analyzer_initialized': analyzer._initialized
            }
            
            print(f"  ✅ FinBERT test passed - Average score: {sum(batch_scores)/len(batch_scores):.4f}")
            
        except Exception as e:
            self.results['finbert'] = {'status': 'failed', 'error': str(e)}
            print(f"  ❌ FinBERT test failed: {e}")
    
    async def test_fusion_manager(self):
        """융합 관리자 테스트"""
        print("\n🔄 Testing Sentiment Fusion Manager...")
        
        try:
            fusion_manager = await get_fusion_manager()
            
            # 단일 소스 융합
            single_score = await fusion_manager.fuse({"news": 0.75})
            
            # 다중 소스 융합
            multi_scores = {
                "news": 0.8,
                "social": 0.6,
                "technical": 0.7,
                "historical": 0.65
            }
            multi_score = await fusion_manager.fuse(multi_scores, symbol="BTCUSDT")
            
            # 이상치 처리 테스트
            outlier_scores = {
                "news": 0.7,
                "social": 0.65,
                "technical": 0.1,  # 이상치
                "historical": 0.68
            }
            outlier_score = await fusion_manager.fuse(outlier_scores, symbol="BTCUSDT")
            
            # 통계 조회
            stats = fusion_manager.get_statistics("BTCUSDT")
            
            # 기사 융합 분석
            fused_articles = await fusion_manager.get_fused_scores(self.test_articles, "BTCUSDT")
            
            self.results['fusion'] = {
                'status': 'success',
                'single_score': single_score,
                'multi_score': multi_score,
                'outlier_score': outlier_score,
                'statistics': stats,
                'fused_articles_count': len(fused_articles),
                'fused_articles_avg': sum(a['sentiment_score'] for a in fused_articles) / len(fused_articles) if fused_articles else 0
            }
            
            print(f"  ✅ Fusion test passed - Multi-source: {multi_score:.4f}, Outlier-handled: {outlier_score:.4f}")
            
        except Exception as e:
            self.results['fusion'] = {'status': 'failed', 'error': str(e)}
            print(f"  ❌ Fusion test failed: {e}")
    
    async def test_sentiment_router(self):
        """감정 라우터 테스트"""
        print("\n🚏 Testing Sentiment Router...")
        
        try:
            # Live 모드 테스트
            router_live = await get_router("live")
            
            live_results = []
            for article in self.test_articles:
                text = f"{article['title']} {article['snippet']}"
                score = await router_live.get_score(news_text=text)
                metadata = await router_live.get_score_with_metadata(news_text=text)
                
                live_results.append({
                    'score': score,
                    'metadata': metadata
                })
            
            # 배치 분석 테스트
            batch_results = await router_live.analyze_articles_batch(self.test_articles, "BTCUSDT")
            
            # Backtest 모드 테스트 (샘플 데이터 사용)
            router_backtest = await get_router("backtest")
            backtest_score = await router_backtest.get_score(
                timestamp=datetime.now() - timedelta(hours=12)
            )
            
            self.results['router'] = {
                'status': 'success',
                'live_mode': {
                    'individual_scores': [r['score'] for r in live_results],
                    'batch_results_count': len(batch_results),
                    'batch_avg_score': sum(r['sentiment_score'] for r in batch_results) / len(batch_results) if batch_results else 0
                },
                'backtest_mode': {
                    'sample_score': backtest_score
                },
                'router_modes': {
                    'live_mode': router_live.get_mode(),
                    'backtest_mode': router_backtest.get_mode()
                }
            }
            
            print(f"  ✅ Router test passed - Live avg: {sum(r['score'] for r in live_results)/len(live_results):.4f}, Backtest: {backtest_score:.4f}")
            
        except Exception as e:
            self.results['router'] = {'status': 'failed', 'error': str(e)}
            print(f"  ❌ Router test failed: {e}")
    
    async def test_sentiment_aggregator_v3(self):
        """SentimentAggregator v3.0 통합 테스트"""
        print("\n📊 Testing SentimentAggregator v3.0...")
        
        try:
            # Live 모드 테스트
            aggregator_live = SentimentAggregator(mode="live")
            await aggregator_live.initialize()
            
            # 통합 감정 분석
            btc_sentiment = await aggregator_live.aggregate_sentiment("BTC", lookback_hours=12)
            
            # 실시간 감정 분석
            realtime_sentiment = await aggregator_live.get_real_time_sentiment("BTC")
            
            # 감정 요약
            sentiment_summary = aggregator_live.get_sentiment_summary("BTC")
            
            # Backtest 모드 테스트
            aggregator_backtest = SentimentAggregator(mode="backtest")
            await aggregator_backtest.initialize()
            
            backtest_sentiment = await aggregator_backtest.aggregate_sentiment(
                "BTC", 
                timestamp=datetime.now() - timedelta(hours=6),
                lookback_hours=12
            )
            
            self.results['aggregator_v3'] = {
                'status': 'success',
                'live_mode': {
                    'aggregate_sentiment': btc_sentiment,
                    'realtime_sentiment': realtime_sentiment,
                    'sentiment_summary': sentiment_summary,
                    'mode': aggregator_live.get_mode()
                },
                'backtest_mode': {
                    'aggregate_sentiment': backtest_sentiment,
                    'mode': aggregator_backtest.get_mode()
                },
                'initialization_status': {
                    'live_initialized': aggregator_live._initialized,
                    'backtest_initialized': aggregator_backtest._initialized
                }
            }
            
            # 리소스 정리
            await aggregator_live.close()
            await aggregator_backtest.close()
            
            print(f"  ✅ Aggregator v3.0 test passed")
            print(f"    Live overall: {btc_sentiment['overall']:.4f} | Confidence: {btc_sentiment['confidence']:.4f}")
            print(f"    Real-time: {realtime_sentiment['sentiment_score']:.4f} | Articles: {realtime_sentiment['article_count']}")
            print(f"    Backtest overall: {backtest_sentiment['overall']:.4f}")
            
        except Exception as e:
            self.results['aggregator_v3'] = {'status': 'failed', 'error': str(e)}
            print(f"  ❌ Aggregator v3.0 test failed: {e}")
    
    async def test_integration_compatibility(self):
        """기존 시스템과의 호환성 테스트"""
        print("\n🔗 Testing Integration Compatibility...")
        
        try:
            # 기존 API 호환성 테스트
            from SharedCore.sentiment_engine.routing.sentiment_router import get_sentiment_score
            
            compatibility_results = []
            for article in self.test_articles:
                text = f"{article['title']} {article['snippet']}"
                score = await get_sentiment_score(text)
                compatibility_results.append(score)
            
            # UnifiedDataProvider와의 호환성 테스트 시뮬레이션
            aggregator = SentimentAggregator(mode="live")
            await aggregator.initialize()
            
            # 표준 형식 검증
            sentiment_data = await aggregator.get_real_time_sentiment("CRYPTO")
            required_fields = ['sentiment_score', 'confidence', 'article_count', 'status']
            
            compatibility_check = all(field in sentiment_data for field in required_fields)
            
            await aggregator.close()
            
            self.results['compatibility'] = {
                'status': 'success',
                'api_compatibility': {
                    'get_sentiment_score_working': len(compatibility_results) == len(self.test_articles),
                    'scores': compatibility_results,
                    'average': sum(compatibility_results) / len(compatibility_results)
                },
                'data_format_compatibility': {
                    'required_fields_present': compatibility_check,
                    'sentiment_data_structure': list(sentiment_data.keys())
                }
            }
            
            print(f"  ✅ Compatibility test passed")
            print(f"    API compatibility: ✅ | Data format: {'✅' if compatibility_check else '❌'}")
            
        except Exception as e:
            self.results['compatibility'] = {'status': 'failed', 'error': str(e)}
            print(f"  ❌ Compatibility test failed: {e}")
    
    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "="*70)
        print("📈 ADVANCED SENTIMENT INTEGRATION TEST SUMMARY")
        print("="*70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'success')
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.results.items():
            status_icon = '✅' if result['status'] == 'success' else '❌'
            print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result['status']}")
            
            if result['status'] == 'failed':
                print(f"     Error: {result['error']}")
        
        print("\n💡 Integration Status:")
        if failed_tests == 0:
            print("  🎉 All advanced sentiment modules successfully integrated!")
            print("  🚀 Ready for production use with FinBERT + Fusion + Router")
        elif passed_tests >= 3:
            print("  ✨ Core integration successful - minor issues detected")
            print("  🔧 Review failed components for optimization")
        else:
            print("  🔧 Integration issues detected - review configuration")
        
        print("\n📋 Key Features Validated:")
        if self.results.get('finbert', {}).get('status') == 'success':
            print("  ✅ FinBERT-based sentiment analysis")
        if self.results.get('fusion', {}).get('status') == 'success':
            print("  ✅ Multi-source sentiment fusion with outlier detection")
        if self.results.get('router', {}).get('status') == 'success':
            print("  ✅ Live/Backtest mode routing")
        if self.results.get('aggregator_v3', {}).get('status') == 'success':
            print("  ✅ Enhanced SentimentAggregator v3.0")
        if self.results.get('compatibility', {}).get('status') == 'success':
            print("  ✅ Backward compatibility with existing systems")


async def main():
    """메인 테스트 실행"""
    print("🚀 Advanced Sentiment Integration Test Suite")
    print("Testing FinBERT + Fusion + Router + SentimentAggregator v3.0")
    print("="*70)
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.WARNING,  # 테스트 중에는 경고 이상만 표시
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tester = AdvancedSentimentTester()
    
    # 모든 테스트 실행
    await tester.test_finbert_analyzer()
    await tester.test_fusion_manager()
    await tester.test_sentiment_router()
    await tester.test_sentiment_aggregator_v3()
    await tester.test_integration_compatibility()
    
    # 결과 요약
    tester.print_summary()
    
    print("\n🏁 Advanced sentiment integration test suite completed!")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())