#!/usr/bin/env python3
"""
AuroraQ 거래 시스템과 새로운 뉴스 시스템 연동 예제
"""

import asyncio
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

# 기존 방식으로 뉴스 수집기 import (코드 수정 없음)
from SharedCore.sentiment_engine.news_collectors.news_collector import NewsCollector

class AuroraQTradingBot:
    """AuroraQ 거래 봇 (기존 인터페이스 유지)"""
    
    def __init__(self):
        # 기존 방식으로 뉴스 수집기 초기화
        self.news_collector = NewsCollector()
        self.sentiment_threshold = 0.6  # 감정 임계값
        
    async def initialize(self):
        """초기화"""
        await self.news_collector.connect()
        print("🤖 AuroraQ 거래 봇 초기화 완료")
    
    async def analyze_market_sentiment(self):
        """시장 감정 분석 (기존 로직 그대로 사용)"""
        print("📊 시장 감정 분석 중...")
        
        # 1. 최신 암호화폐 뉴스 수집 (기존 방식)
        crypto_news = await self.news_collector.get_latest_crypto_news(
            hours_back=6, 
            max_articles=50
        )
        print(f"   수집된 뉴스: {len(crypto_news)}개")
        
        if not crypto_news:
            print("   ⚠️ 수집된 뉴스가 없습니다")
            return {"sentiment": 0.5, "confidence": 0.0, "action": "hold", "strength": 0.0, "article_count": 0}
        
        # 2. 감정 분석 (기존 방식)
        sentiment_summary = await self.news_collector.get_sentiment_summary(crypto_news)
        
        overall_sentiment = sentiment_summary['overall_sentiment']
        confidence = sentiment_summary['confidence']
        
        print(f"   전체 감정: {overall_sentiment:.3f}")
        print(f"   신뢰도: {confidence:.3f}")
        print(f"   기사 분포: 긍정 {sentiment_summary['positive_count']}, "
              f"부정 {sentiment_summary['negative_count']}, "
              f"중립 {sentiment_summary['neutral_count']}")
        
        # 3. 거래 신호 생성
        if overall_sentiment > self.sentiment_threshold and confidence > 0.5:
            action = "buy"
            strength = min(1.0, (overall_sentiment - 0.5) * 2)
        elif overall_sentiment < (1 - self.sentiment_threshold) and confidence > 0.5:
            action = "sell"  
            strength = min(1.0, (0.5 - overall_sentiment) * 2)
        else:
            action = "hold"
            strength = 0.0
        
        return {
            "sentiment": overall_sentiment,
            "confidence": confidence,
            "action": action,
            "strength": strength,
            "article_count": len(crypto_news)
        }
    
    async def check_breaking_news(self):
        """속보 확인 및 긴급 거래 신호"""
        print("🚨 속보 확인 중...")
        
        # 최근 30분 속보 확인 (기존 방식)
        breaking_news = await self.news_collector.get_breaking_news(minutes=30)
        
        if breaking_news:
            print(f"   🔥 발견된 속보: {len(breaking_news)}개")
            
            # 속보들의 감정 분석
            breaking_sentiment = await self.news_collector.get_sentiment_summary(breaking_news)
            
            for news in breaking_news[:3]:  # 상위 3개만 출력
                print(f"      - {news.title}")
                print(f"        소스: {news.source}, 시간: {news.published}")
            
            return {
                "has_breaking": True,
                "count": len(breaking_news),
                "sentiment": breaking_sentiment['overall_sentiment'],
                "articles": breaking_news
            }
        else:
            print("   📰 최근 속보 없음")
            return {"has_breaking": False}
    
    async def get_trading_signals(self):
        """통합 거래 신호 생성"""
        print("\n🎯 거래 신호 생성")
        print("=" * 40)
        
        # 1. 일반적인 시장 감정 분석
        market_analysis = await self.analyze_market_sentiment()
        
        # 2. 속보 확인
        breaking_analysis = await self.check_breaking_news()
        
        # 3. 통합 신호 계산
        final_action = market_analysis['action']
        final_strength = market_analysis['strength']
        
        # 속보가 있으면 가중치 증가
        if breaking_analysis.get('has_breaking'):
            breaking_sentiment = breaking_analysis.get('sentiment', 0.5)
            
            # 속보 감정이 기존 감정과 같은 방향이면 강화
            if (market_analysis['sentiment'] > 0.5 and breaking_sentiment > 0.5) or \
               (market_analysis['sentiment'] < 0.5 and breaking_sentiment < 0.5):
                final_strength = min(1.0, final_strength * 1.5)
                print("   📈 속보가 기존 신호를 강화합니다")
            
            # 속보 감정이 기존 감정과 반대면 약화
            elif (market_analysis['sentiment'] > 0.5 and breaking_sentiment < 0.5) or \
                 (market_analysis['sentiment'] < 0.5 and breaking_sentiment > 0.5):
                final_strength = max(0.0, final_strength * 0.5)
                print("   📉 속보가 기존 신호와 상충합니다")
        
        # 4. 최종 결과
        result = {
            "timestamp": datetime.now().isoformat(),
            "market_sentiment": market_analysis['sentiment'],
            "confidence": market_analysis['confidence'],
            "action": final_action,
            "strength": final_strength,
            "article_count": market_analysis['article_count'],
            "has_breaking_news": breaking_analysis.get('has_breaking', False),
            "breaking_count": breaking_analysis.get('count', 0)
        }
        
        # 결과 출력
        print(f"\n📊 최종 거래 신호:")
        print(f"   액션: {final_action.upper()}")
        print(f"   강도: {final_strength:.2f}")
        print(f"   시장 감정: {market_analysis['sentiment']:.3f}")
        print(f"   신뢰도: {market_analysis['confidence']:.3f}")
        
        return result
    
    async def close(self):
        """리소스 정리"""
        await self.news_collector.close()
        print("🤖 거래 봇 종료")

async def main():
    """메인 실행"""
    bot = AuroraQTradingBot()
    
    try:
        await bot.initialize()
        
        # 거래 신호 생성
        signals = await bot.get_trading_signals()
        
        # 실제 거래 로직은 여기에 추가
        if signals['action'] == 'buy' and signals['strength'] > 0.3:
            print(f"\n💰 매수 신호! 강도: {signals['strength']:.2f}")
            # 실제 매수 로직 호출
        elif signals['action'] == 'sell' and signals['strength'] > 0.3:
            print(f"\n💸 매도 신호! 강도: {signals['strength']:.2f}")
            # 실제 매도 로직 호출
        else:
            print(f"\n⏳ 관망 신호. 강도: {signals['strength']:.2f}")
        
    finally:
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())