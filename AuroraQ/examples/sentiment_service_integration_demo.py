#!/usr/bin/env python3
"""
AuroraQ/MacroQ Sentiment Service Integration Demo
Shows how to migrate from local sentiment modules to the independent sentiment service
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from SharedCore.sentiment_engine.sentiment_client import (
    SentimentServiceClient,
    SentimentServiceConfig,
    get_analyzer_proxy,
    get_fusion_manager_proxy
)


class AuroraQSentimentIntegration:
    """
    Example of how AuroraQ would integrate with the sentiment service
    Demonstrates resource efficiency and caching optimization
    """
    
    def __init__(self, sentiment_service_url: str = "http://localhost:8000"):
        # Configure sentiment service client for optimal performance
        config = SentimentServiceConfig(
            base_url=sentiment_service_url,
            timeout=30.0,
            retry_attempts=3,
            retry_delay=1.0,
            enable_caching=True,
            cache_ttl=300  # 5 minutes cache for real-time trading
        )
        self.sentiment_client = SentimentServiceClient(config)
        self.cache = {}  # Local cache for trading decisions
    
    async def get_trading_sentiment(self, symbol: str, news_articles: List[Dict]) -> Dict[str, Any]:
        """
        Get comprehensive sentiment for trading decisions
        Optimized for AuroraQ's real-time trading needs
        """
        try:
            # Extract text content from news articles
            news_texts = []
            for article in news_articles:
                text = article.get('title', '') + ' ' + article.get('content', '')
                if text.strip():
                    news_texts.append(text.strip())
            
            if not news_texts:
                return self._get_fallback_sentiment(symbol)
            
            # Batch analyze news for efficiency
            batch_result = await self.sentiment_client.analyze_batch_sentiment(
                texts=news_texts,
                symbol=symbol,
                include_detailed=True
            )
            
            # Calculate news sentiment
            news_scores = [r['sentiment_score'] for r in batch_result['results']]
            news_sentiment = sum(news_scores) / len(news_scores)
            
            # Get additional sentiment sources (would be implemented separately)
            social_sentiment = await self._get_social_sentiment(symbol)
            technical_sentiment = await self._get_technical_sentiment(symbol)
            
            # Fuse all sentiment sources
            fusion_scores = {
                "news": news_sentiment,
                "social": social_sentiment,
                "technical": technical_sentiment
            }
            
            fusion_result = await self.sentiment_client.fuse_sentiment_scores(
                sentiment_scores=fusion_scores,
                symbol=symbol,
                timestamp=datetime.now()
            )
            
            # Extract key insights for trading
            trading_sentiment = {
                "symbol": symbol,
                "fused_sentiment": fusion_result['fused_score'],
                "confidence": fusion_result['confidence'],
                "trend": fusion_result['trend'],
                "volatility": fusion_result['volatility'],
                "sources": {
                    "news": {
                        "sentiment": news_sentiment,
                        "article_count": len(news_articles),
                        "keywords": self._extract_top_keywords(batch_result['results'])
                    },
                    "social": {"sentiment": social_sentiment},
                    "technical": {"sentiment": technical_sentiment}
                },
                "trading_signal": self._generate_trading_signal(fusion_result),
                "timestamp": datetime.now().isoformat()
            }
            
            return trading_sentiment
            
        except Exception as e:
            print(f"Error getting trading sentiment: {e}")
            return self._get_fallback_sentiment(symbol)
    
    async def _get_social_sentiment(self, symbol: str) -> float:
        """Placeholder for social media sentiment (would integrate with Twitter API, etc.)"""
        # In real implementation, this would call social media APIs
        return 0.6  # Mock positive social sentiment
    
    async def _get_technical_sentiment(self, symbol: str) -> float:
        """Placeholder for technical analysis sentiment"""
        # In real implementation, this would analyze price patterns, indicators
        return 0.65  # Mock technical sentiment
    
    def _extract_top_keywords(self, sentiment_results: List[Dict]) -> List[str]:
        """Extract most common keywords from sentiment analysis"""
        keyword_counts = {}
        for result in sentiment_results:
            for keyword in result.get('keywords', []):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Return top 5 keywords
        return sorted(keyword_counts.keys(), key=keyword_counts.get, reverse=True)[:5]
    
    def _generate_trading_signal(self, fusion_result: Dict) -> str:
        """Generate trading signal based on fused sentiment"""
        score = fusion_result['fused_score']
        confidence = fusion_result['confidence']
        
        if confidence < 0.6:
            return "HOLD"  # Low confidence, avoid trading
        elif score > 0.7:
            return "BUY"
        elif score < 0.3:
            return "SELL"
        else:
            return "HOLD"
    
    def _get_fallback_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fallback sentiment when service is unavailable"""
        return {
            "symbol": symbol,
            "fused_sentiment": 0.5,
            "confidence": 0.0,
            "trend": "neutral",
            "volatility": 0.0,
            "sources": {},
            "trading_signal": "HOLD",
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        }
    
    async def close(self):
        """Clean up resources"""
        await self.sentiment_client.close()


class MacroQSentimentIntegration:
    """
    Example of how MacroQ would integrate with the sentiment service
    Focused on macro economic sentiment analysis
    """
    
    def __init__(self, sentiment_service_url: str = "http://localhost:8000"):
        config = SentimentServiceConfig(
            base_url=sentiment_service_url,
            cache_ttl=3600  # 1 hour cache for macro analysis (less frequent updates)
        )
        self.sentiment_client = SentimentServiceClient(config)
    
    async def analyze_fed_sentiment(self, fed_minutes: str) -> Dict[str, Any]:
        """
        Analyze Federal Reserve meeting minutes for monetary policy sentiment
        """
        try:
            # Analyze Fed minutes with detailed results
            detailed_result = await self.sentiment_client.analyze_sentiment_detailed(
                text=fed_minutes,
                symbol="SPY"  # Use S&P 500 as proxy for market sentiment
            )
            
            # Extract policy-relevant information
            policy_keywords = [
                kw for kw in detailed_result.get('keywords', [])
                if kw.lower() in ['inflation', 'rates', 'employment', 'growth', 'policy', 'outlook']
            ]
            
            # Determine hawkish/dovish stance
            sentiment_score = detailed_result['sentiment_score']
            if sentiment_score < 0.4:
                policy_stance = "hawkish"
                market_impact = "negative"
            elif sentiment_score > 0.6:
                policy_stance = "dovish" 
                market_impact = "positive"
            else:
                policy_stance = "neutral"
                market_impact = "mixed"
            
            return {
                "sentiment_score": sentiment_score,
                "confidence": detailed_result['confidence'],
                "policy_stance": policy_stance,
                "market_impact": market_impact,
                "policy_keywords": policy_keywords,
                "scenario_tag": detailed_result.get('scenario_tag', ''),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing Fed sentiment: {e}")
            return {
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "policy_stance": "neutral",
                "market_impact": "unknown",
                "policy_keywords": [],
                "error": str(e)
            }
    
    async def analyze_economic_news_sentiment(self, economic_indicators: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment from economic news and data releases
        """
        try:
            # Prepare texts for batch analysis
            texts = []
            indicator_types = []
            
            for indicator in economic_indicators:
                text = f"{indicator['title']} {indicator.get('description', '')}"
                texts.append(text)
                indicator_types.append(indicator.get('type', 'unknown'))
            
            # Batch analyze economic indicators
            batch_result = await self.sentiment_client.analyze_batch_sentiment(
                texts=texts,
                symbol="DXY",  # Dollar index for macro sentiment
                include_detailed=True
            )
            
            # Categorize sentiment by indicator type
            sentiment_by_category = {}
            for i, result in enumerate(batch_result['results']):
                category = indicator_types[i]
                if category not in sentiment_by_category:
                    sentiment_by_category[category] = []
                sentiment_by_category[category].append(result['sentiment_score'])
            
            # Calculate category averages
            category_sentiment = {
                category: sum(scores) / len(scores)
                for category, scores in sentiment_by_category.items()
            }
            
            # Fuse economic sentiment
            if category_sentiment:
                fusion_result = await self.sentiment_client.fuse_sentiment_scores(
                    sentiment_scores=category_sentiment,
                    symbol="ECONOMY",
                    timestamp=datetime.now()
                )
                
                overall_sentiment = fusion_result['fused_score']
                confidence = fusion_result['confidence']
            else:
                overall_sentiment = 0.5
                confidence = 0.0
            
            return {
                "overall_sentiment": overall_sentiment,
                "confidence": confidence,
                "category_sentiment": category_sentiment,
                "total_indicators": len(economic_indicators),
                "average_score": batch_result['average_score'],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing economic sentiment: {e}")
            return {
                "overall_sentiment": 0.5,
                "confidence": 0.0,
                "category_sentiment": {},
                "error": str(e)
            }
    
    async def close(self):
        """Clean up resources"""
        await self.sentiment_client.close()


async def demonstrate_migration_compatibility():
    """
    Demonstrate backward compatibility with existing code
    Shows how existing AuroraQ/MacroQ code can work with minimal changes
    """
    print("🔄 Testing backward compatibility...")
    
    try:
        # Original interface (now using service proxy)
        analyzer = await get_analyzer_proxy()
        fusion_manager = await get_fusion_manager_proxy()
        
        # Test original analyze method
        sentiment = await analyzer.analyze("Bitcoin shows strong bullish momentum")
        print(f"✅ Original analyze() method: {sentiment:.3f}")
        
        # Test original detailed analysis
        detailed = await analyzer.analyze_detailed("Fed maintains hawkish stance on inflation")
        print(f"✅ Original detailed analysis: {detailed.sentiment_score:.3f} ({detailed.label.value})")
        
        # Test original fusion
        scores = {"news": 0.7, "social": 0.6, "technical": 0.8}
        fused = await fusion_manager.fuse(scores, symbol="BTC")
        print(f"✅ Original fusion: {fused:.3f}")
        
        # Cleanup
        await analyzer.close()
        await fusion_manager.close()
        
        print("✅ Backward compatibility test passed!")
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")


async def demonstrate_performance_optimization():
    """
    Demonstrate performance optimizations and resource efficiency
    """
    print("\n⚡ Testing performance optimizations...")
    
    config = SentimentServiceConfig(
        base_url="http://localhost:8000",
        enable_caching=True,
        cache_ttl=300,
        retry_attempts=2
    )
    
    client = SentimentServiceClient(config)
    
    try:
        # Test batch processing for efficiency
        news_texts = [
            "Cryptocurrency adoption accelerates among institutions",
            "Central bank digital currencies gain momentum globally", 
            "DeFi protocols show resilience despite market volatility",
            "Regulatory clarity improves crypto market sentiment",
            "Bitcoin ETF approval sparks institutional interest"
        ]
        
        start_time = datetime.now()
        
        # Efficient batch processing
        batch_result = await client.analyze_batch_sentiment(
            texts=news_texts,
            symbol="CRYPTO",
            include_detailed=False
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Batch processed {len(news_texts)} texts in {processing_time:.2f}s")
        print(f"✅ Average sentiment: {batch_result['average_score']:.3f}")
        print(f"✅ Throughput: {len(news_texts)/processing_time:.1f} texts/second")
        
        # Test caching efficiency
        start_time = datetime.now()
        
        # Same request should be cached
        cached_result = await client.analyze_sentiment(
            "Cryptocurrency adoption accelerates among institutions",
            symbol="CRYPTO"
        )
        
        cache_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Cached request completed in {cache_time:.3f}s")
        
        # Test fusion efficiency
        fusion_scores = {
            "news": batch_result['average_score'],
            "social": 0.65,
            "technical": 0.7,
            "historical": 0.6
        }
        
        fusion_result = await client.fuse_sentiment_scores(fusion_scores, "CRYPTO")
        print(f"✅ Fusion result: {fusion_result['fused_score']:.3f} (trend: {fusion_result['trend']})")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    finally:
        await client.close()


async def main():
    """
    Main demonstration of sentiment service integration
    """
    print("🚀 AuroraQ/MacroQ Sentiment Service Integration Demo")
    print("=" * 60)
    
    # Mock data for demonstration
    sample_news = [
        {
            "title": "Bitcoin Reaches New All-Time High",
            "content": "Bitcoin surged past $70,000 as institutional adoption continues to grow"
        },
        {
            "title": "Fed Signals Potential Rate Cuts",
            "content": "Federal Reserve hints at dovish policy shift amid economic uncertainty"
        },
        {
            "title": "Crypto Regulation Clarity Improves",
            "content": "New regulatory framework provides clearer guidelines for cryptocurrency operations"
        }
    ]
    
    sample_economic_indicators = [
        {
            "title": "Inflation Rate Drops to 2.1%",
            "description": "Consumer price index shows continued decline in inflationary pressures",
            "type": "inflation"
        },
        {
            "title": "Employment Data Shows Strength", 
            "description": "Job growth exceeds expectations with unemployment at historic lows",
            "type": "employment"
        },
        {
            "title": "GDP Growth Accelerates",
            "description": "Economic expansion continues with strong consumer spending",
            "type": "growth"
        }
    ]
    
    fed_minutes_sample = """
    The Federal Reserve decided to maintain the current federal funds rate at 5.25-5.50%. 
    Committee members noted that inflation has shown signs of moderating but remains above 
    the 2% target. Employment conditions remain strong with continued job growth. 
    The Committee will continue to assess incoming data and adjust policy as appropriate.
    """
    
    # Test AuroraQ integration
    print("\n📈 Testing AuroraQ Integration...")
    aurora_integration = AuroraQSentimentIntegration()
    
    try:
        trading_sentiment = await aurora_integration.get_trading_sentiment("BTC", sample_news)
        
        print(f"Symbol: {trading_sentiment['symbol']}")
        print(f"Fused Sentiment: {trading_sentiment['fused_sentiment']:.3f}")
        print(f"Confidence: {trading_sentiment['confidence']:.3f}")
        print(f"Trading Signal: {trading_sentiment['trading_signal']}")
        print(f"Trend: {trading_sentiment['trend']}")
        print(f"Top Keywords: {', '.join(trading_sentiment['sources']['news']['keywords'])}")
        
    except Exception as e:
        print(f"AuroraQ integration error: {e}")
    
    finally:
        await aurora_integration.close()
    
    # Test MacroQ integration
    print("\n🌍 Testing MacroQ Integration...")
    macro_integration = MacroQSentimentIntegration()
    
    try:
        # Test Fed minutes analysis
        fed_analysis = await macro_integration.analyze_fed_sentiment(fed_minutes_sample)
        print(f"Fed Policy Stance: {fed_analysis['policy_stance']}")
        print(f"Market Impact: {fed_analysis['market_impact']}")
        print(f"Confidence: {fed_analysis['confidence']:.3f}")
        print(f"Policy Keywords: {', '.join(fed_analysis['policy_keywords'])}")
        
        # Test economic indicators
        economic_analysis = await macro_integration.analyze_economic_news_sentiment(sample_economic_indicators)
        print(f"Overall Economic Sentiment: {economic_analysis['overall_sentiment']:.3f}")
        print(f"Category Breakdown: {economic_analysis['category_sentiment']}")
        
    except Exception as e:
        print(f"MacroQ integration error: {e}")
    
    finally:
        await macro_integration.close()
    
    # Test backward compatibility
    await demonstrate_migration_compatibility()
    
    # Test performance optimizations
    await demonstrate_performance_optimization()
    
    print("\n✅ Integration demonstration complete!")
    print("\nKey Benefits:")
    print("• 🔧 Resource efficiency through centralized model loading")
    print("• ⚡ Improved performance with Redis caching")
    print("• 🔄 Backward compatibility with existing code")
    print("• 📊 Enhanced monitoring and observability")
    print("• 🚀 Horizontal scalability")
    print("• 🛡️ Built-in rate limiting and error handling")


if __name__ == "__main__":
    print("Starting sentiment service integration demo...")
    print("Note: Make sure the sentiment service is running on http://localhost:8000")
    print("Run: docker-compose up -d in the sentiment-service directory")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure the sentiment service is running and accessible")