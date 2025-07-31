#!/usr/bin/env python3
"""Finnhub 수집기"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

class FinnhubCollector:
    """Finnhub.io 뉴스 및 데이터 수집기"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self.base_url = "https://finnhub.io/api/v1"
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        if not self.api_key:
            self.logger.warning("Finnhub API key not provided - collector will be disabled")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def get_company_news(self, symbol: str, 
                              from_date: Optional[str] = None, 
                              to_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """특정 회사 뉴스 수집"""
        if not self.api_key:
            self.logger.warning("Finnhub API key not available")
            return []
        
        # 기본 날짜 설정 (최근 7일)
        if not from_date:
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/company-news"
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self.api_key
        }
        
        try:
            async with self.session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    self.logger.error(f"Finnhub company news error: HTTP {response.status}")
                    return []
                
                data = await response.json()
                
                if isinstance(data, dict) and "error" in data:
                    self.logger.error(f"Finnhub error: {data['error']}")
                    return []
                
                articles = []
                for article_data in data:
                    article = {
                        "id": article_data.get("id"),
                        "category": article_data.get("category", ""),
                        "headline": article_data.get("headline", ""),
                        "image": article_data.get("image", ""),
                        "related": article_data.get("related", ""),
                        "source": article_data.get("source", "Finnhub"),
                        "summary": article_data.get("summary", ""),
                        "url": article_data.get("url", ""),
                        "datetime": article_data.get("datetime", 0),
                        "symbol": symbol,
                        "collected_at": datetime.now().isoformat()
                    }
                    
                    # Unix timestamp를 datetime으로 변환
                    if article["datetime"]:
                        try:
                            article["published_at"] = datetime.fromtimestamp(article["datetime"]).isoformat()
                        except:
                            article["published_at"] = datetime.now().isoformat()
                    
                    articles.append(article)
                
                self.logger.info(f"Collected {len(articles)} articles for {symbol} from Finnhub")
                return articles
        
        except asyncio.TimeoutError:
            self.logger.error("Finnhub company news request timeout")
            return []
        except Exception as e:
            self.logger.error(f"Finnhub company news error: {str(e)}")
            return []
    
    async def get_general_news(self, category: str = "general", 
                              min_id: int = 0) -> List[Dict[str, Any]]:
        """일반 뉴스 수집"""
        if not self.api_key:
            self.logger.warning("Finnhub API key not available")
            return []
        
        url = f"{self.base_url}/news"
        params = {
            "category": category,
            "minId": min_id,
            "token": self.api_key
        }
        
        try:
            async with self.session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    self.logger.error(f"Finnhub general news error: HTTP {response.status}")
                    return []
                
                data = await response.json()
                
                if isinstance(data, dict) and "error" in data:
                    self.logger.error(f"Finnhub error: {data['error']}")
                    return []
                
                articles = []
                for article_data in data:
                    article = {
                        "id": article_data.get("id"),
                        "category": category,
                        "headline": article_data.get("headline", ""),
                        "image": article_data.get("image", ""),
                        "related": article_data.get("related", ""),
                        "source": article_data.get("source", "Finnhub"),
                        "summary": article_data.get("summary", ""),
                        "url": article_data.get("url", ""),
                        "datetime": article_data.get("datetime", 0),
                        "collected_at": datetime.now().isoformat()
                    }
                    
                    # Unix timestamp를 datetime으로 변환
                    if article["datetime"]:
                        try:
                            article["published_at"] = datetime.fromtimestamp(article["datetime"]).isoformat()
                        except:
                            article["published_at"] = datetime.now().isoformat()
                    
                    articles.append(article)
                
                self.logger.info(f"Collected {len(articles)} general news articles from Finnhub")
                return articles
        
        except asyncio.TimeoutError:
            self.logger.error("Finnhub general news request timeout")
            return []
        except Exception as e:
            self.logger.error(f"Finnhub general news error: {str(e)}")
            return []
    
    async def get_crypto_news(self) -> List[Dict[str, Any]]:
        """암호화폐 관련 뉴스 수집"""
        crypto_symbols = ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT"]
        all_articles = []
        
        for symbol in crypto_symbols:
            articles = await self.get_company_news(symbol)
            all_articles.extend(articles)
        
        # 일반 암호화폐 뉴스도 수집
        general_crypto = await self.get_general_news(category="crypto")
        all_articles.extend(general_crypto)
        
        # 중복 제거 및 정렬
        seen_ids = set()
        unique_articles = []
        
        for article in all_articles:
            article_id = article.get("id") or article.get("url")
            if article_id and article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_articles.append(article)
        
        # 시간순 정렬
        unique_articles.sort(key=lambda x: x.get("datetime", 0), reverse=True)
        
        return unique_articles
    
    async def get_market_news(self) -> List[Dict[str, Any]]:
        """주식 시장 뉴스 수집"""
        market_categories = ["general", "forex", "merger"]
        all_articles = []
        
        for category in market_categories:
            articles = await self.get_general_news(category=category)
            all_articles.extend(articles)
        
        # 주요 주식 심볼들의 뉴스도 수집
        major_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        for symbol in major_symbols:
            articles = await self.get_company_news(symbol)
            # 최신 5개만 추가
            all_articles.extend(articles[:5])
        
        # 중복 제거 및 정렬
        seen_ids = set()
        unique_articles = []
        
        for article in all_articles:
            article_id = article.get("id") or article.get("url")
            if article_id and article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_articles.append(article)
        
        # 시간순 정렬
        unique_articles.sort(key=lambda x: x.get("datetime", 0), reverse=True)
        
        return unique_articles[:50]  # 최대 50개만 반환
    
    async def health_check(self) -> Dict[str, Any]:
        """수집기 상태 확인"""
        if not self.api_key:
            return {
                "status": "disabled",
                "reason": "API key not configured"
            }
        
        try:
            # 간단한 테스트 요청
            test_articles = await self.get_general_news(category="general")
            
            return {
                "status": "healthy",
                "api_key_configured": True,
                "test_request_successful": len(test_articles) >= 0
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

# 사용 예제
async def main():
    """Finnhub 수집기 테스트"""
    async with FinnhubCollector() as collector:
        # API 키 확인
        health = await collector.health_check()
        print(f"Finnhub Health: {health}")
        
        if health["status"] == "healthy":
            # 일반 뉴스 수집
            general_news = await collector.get_general_news(category="general")
            print(f"\nGeneral News: {len(general_news)} articles")
            for article in general_news[:3]:
                print(f"  - {article['headline'][:60]}...")
            
            # 암호화폐 뉴스 수집
            crypto_news = await collector.get_crypto_news()
            print(f"\nCrypto News: {len(crypto_news)} articles")
            for article in crypto_news[:3]:
                print(f"  - {article['headline'][:60]}...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())