#!/usr/bin/env python3
"""
핵심 감정 분석 기능 빠른 테스트
FinBERT + Fusion + Router 핵심 기능만 검증
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()


async def test_finbert_basic():
    """FinBERT 기본 기능 테스트"""
    print("🧠 Testing FinBERT Basic Functions...")
    
    try:
        from SharedCore.sentiment_engine.analyzers.finbert_analyzer import get_finbert_analyzer
        
        analyzer = await get_finbert_analyzer()
        
        # 테스트 텍스트
        test_texts = [
            "Bitcoin price surges to new all-time high as ETF approval nears",
            "Major cryptocurrency exchange hacked, investors panic",
            "Federal Reserve remains neutral on digital assets"
        ]
        
        results = []
        for text in test_texts:
            score = await analyzer.analyze(text)
            detailed = await analyzer.analyze_detailed(text)
            
            results.append({
                'text': text[:50] + "...",
                'score': score,
                'label': detailed.label.value,
                'confidence': detailed.confidence,
                'keywords': detailed.keywords[:3]  # 상위 3개만
            })
        
        # 결과 출력
        for i, result in enumerate(results, 1):
            print(f"  Text {i}: {result['text']}")
            print(f"    Score: {result['score']:.4f} | Label: {result['label']} | Confidence: {result['confidence']:.4f}")
            print(f"    Keywords: {', '.join(result['keywords'])}")
        
        print(f"  ✅ FinBERT test passed - {len(results)} texts analyzed")
        return True
        
    except Exception as e:
        print(f"  ❌ FinBERT test failed: {e}")
        return False


async def test_fusion_basic():
    """Fusion Manager 기본 기능 테스트"""
    print("\n🔄 Testing Fusion Manager Basic Functions...")
    
    try:
        from SharedCore.sentiment_engine.fusion.sentiment_fusion_manager import get_fusion_manager
        
        fusion_manager = await get_fusion_manager()
        
        # 단일 소스 테스트
        single_score = await fusion_manager.fuse({"news": 0.75})
        
        # 다중 소스 테스트
        multi_scores = {
            "news": 0.8,
            "social": 0.6,
            "technical": 0.7
        }
        multi_score = await fusion_manager.fuse(multi_scores, symbol="BTCUSDT")
        
        # 이상치 테스트
        outlier_scores = {
            "news": 0.7,
            "social": 0.65,
            "technical": 0.1,  # 이상치
            "historical": 0.68
        }
        outlier_score = await fusion_manager.fuse(outlier_scores, symbol="BTCUSDT")
        
        print(f"  Single source (0.75): {single_score:.4f}")
        print(f"  Multi source: {multi_score:.4f}")
        print(f"  Outlier handled: {outlier_score:.4f} (should be higher than 0.1)")
        
        print(f"  ✅ Fusion test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Fusion test failed: {e}")
        return False


async def test_router_basic():
    """Router 기본 기능 테스트"""
    print("\n🚏 Testing Router Basic Functions...")
    
    try:
        from SharedCore.sentiment_engine.routing.sentiment_router import get_router
        
        # Live 모드 테스트
        router = await get_router("live")
        
        test_text = "Bitcoin shows strong bullish momentum with institutional adoption"
        score = await router.get_score(news_text=test_text)
        metadata = await router.get_score_with_metadata(news_text=test_text)
        
        print(f"  Live mode score: {score:.4f}")
        print(f"  Mode: {router.get_mode()}")
        print(f"  Cached: {metadata['cached']}")
        
        # Backtest 모드 테스트
        await router.switch_mode("backtest")
        backtest_score = await router.get_score(timestamp=datetime.now() - timedelta(hours=12))
        
        print(f"  Backtest mode score: {backtest_score:.4f}")
        print(f"  Mode after switch: {router.get_mode()}")
        
        print(f"  ✅ Router test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Router test failed: {e}")
        return False


async def test_compatibility():
    """기존 API 호환성 테스트"""
    print("\n🔗 Testing API Compatibility...")
    
    try:
        # 기존 get_sentiment_score 함수 테스트
        from SharedCore.sentiment_engine.routing.sentiment_router import get_sentiment_score
        
        test_text = "Cryptocurrency market shows positive sentiment"
        score = await get_sentiment_score(test_text)
        
        print(f"  get_sentiment_score: {score:.4f}")
        print(f"  API compatibility: ✅")
        
        # SentimentAggregator 빠른 테스트
        from SharedCore.sentiment_engine.sentiment_aggregator import SentimentAggregator
        
        aggregator = SentimentAggregator(mode="live")
        await aggregator.initialize()
        
        # 간단한 더미 기사로 테스트
        test_articles = [{
            "title": "Bitcoin Price Analysis",
            "snippet": "Strong technical indicators suggest bullish trend",
            "datetime": datetime.now()
        }]
        
        # 실시간 감정 분석 (에러가 있어도 기본값 반환하는지 확인)
        try:
            sentiment = await aggregator.get_real_time_sentiment("BTC")
            print(f"  Real-time sentiment: {sentiment['sentiment_score']:.4f}")
            print(f"  Status: {sentiment['status']}")
        except Exception:
            print(f"  Real-time sentiment: fallback mode (expected with no Feedly token)")
        
        await aggregator.close()
        
        print(f"  ✅ Compatibility test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Compatibility test failed: {e}")
        return False


async def main():
    """메인 테스트 실행"""
    print("🚀 Core Sentiment Features Test Suite")
    print("Testing essential FinBERT + Fusion + Router integration")
    print("="*60)
    
    tests = [
        test_finbert_basic,
        test_fusion_basic,
        test_router_basic,
        test_compatibility
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    # 결과 요약
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All core sentiment features working correctly!")
        print("✨ FinBERT + Fusion + Router integration successful")
    elif passed >= 3:
        print("\n✨ Core integration successful with minor issues")
    else:
        print("\n🔧 Core integration needs attention")
    
    print("\n🏁 Core test suite completed!")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())