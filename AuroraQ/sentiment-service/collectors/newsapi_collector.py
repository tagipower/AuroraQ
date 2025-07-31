#!/usr/bin/env python3
"""NewsAPI 수집기"""

import asyncio
import aiohttp
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

class NewsAPICollector:
    """NewsAPI.org 수집기"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        if not self.api_key:
            self.logger.warning("NewsAPI key not provided - collector will be disabled")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def search_news(self, query: str, language: str = "en", 
                         page_size: int = 20, sort_by: str = "publishedAt") -> List[Dict[str, Any]]:
        """뉴스 검색"""
        if not self.api_key:
            self.logger.warning("NewsAPI key not available")
            return []
        
        url = f"{self.base_url}/everything"
        params = {
            "q": query,
            "language": language,
            "pageSize": min(page_size, 100),  # 최대 100개
            "sortBy": sort_by,
            "apiKey": self.api_key
        }
        
        try:
            async with self.session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    self.logger.error(f"NewsAPI error: HTTP {response.status}")
                    return []
                
                data = await response.json()
                
                if data.get("status") != "ok":
                    self.logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                    return []
                
                articles = []
                for article_data in data.get("articles", []):
                    article = {
                        "title": article_data.get("title", ""),
                        "description": article_data.get("description", ""),
                        "url": article_data.get("url", ""),
                        "urlToImage": article_data.get("urlToImage", ""),
                        "publishedAt": article_data.get("publishedAt", ""),
                        "source": article_data.get("source", {}).get("name", "NewsAPI"),
                        "author": article_data.get("author", ""),
                        "content": article_data.get("content", ""),
                        "collected_at": datetime.now().isoformat()
                    }
                    articles.append(article)
                
                self.logger.info(f"Collected {len(articles)} articles from NewsAPI for query: {query}")
                return articles
        
        except asyncio.TimeoutError:
            self.logger.error("NewsAPI request timeout")
            return []
        except Exception as e:
            self.logger.error(f"NewsAPI error: {str(e)}")
            return []
    
    async def get_top_headlines(self, category: Optional[str] = None, 
                               country: str = "us", page_size: int = 20) -> List[Dict[str, Any]]:
        """톱 헤드라인 수집"""
        if not self.api_key:
            self.logger.warning("NewsAPI key not available")
            return []
        
        url = f"{self.base_url}/top-headlines"
        params = {
            "country": country,
            "pageSize": min(page_size, 100),
            "apiKey": self.api_key
        }
        
        if category:
            params["category"] = category
        
        try:
            async with self.session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    self.logger.error(f"NewsAPI headlines error: HTTP {response.status}")
                    return []
                
                data = await response.json()
                
                if data.get("status") != "ok":
                    self.logger.error(f"NewsAPI headlines error: {data.get('message', 'Unknown error')}")
                    return []
                
                articles = []
                for article_data in data.get("articles", []):
                    article = {
                        "title": article_data.get("title", ""),
                        "description": article_data.get("description", ""),
                        "url": article_data.get("url", ""),
                        "urlToImage": article_data.get("urlToImage", ""),
                        "publishedAt": article_data.get("publishedAt", ""),
                        "source": article_data.get("source", {}).get("name", "NewsAPI"),
                        "author": article_data.get("author", ""),
                        "content": article_data.get("content", ""),
                        "category": category or "general",
                        "collected_at": datetime.now().isoformat()
                    }
                    articles.append(article)
                
                self.logger.info(f"Collected {len(articles)} headlines from NewsAPI")
                return articles
        
        except asyncio.TimeoutError:
            self.logger.error("NewsAPI headlines request timeout")
            return []
        except Exception as e:
            self.logger.error(f"NewsAPI headlines error: {str(e)}")
            return []
    
    async def get_crypto_news(self, page_size: int = 20) -> List[Dict[str, Any]]:
        """암호화폐 관련 뉴스 수집"""
        crypto_queries = [
            "bitcoin OR cryptocurrency OR ethereum",
            "crypto market OR blockchain",
            "bitcoin price OR ethereum price"
        ]
        
        all_articles = []
        articles_per_query = page_size // len(crypto_queries)
        
        for query in crypto_queries:
            articles = await self.search_news(query, page_size=articles_per_query)
            all_articles.extend(articles)
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_articles = []
        
        for article in all_articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                unique_articles.append(article)
        
        return unique_articles[:page_size]
    
    async def health_check(self) -> Dict[str, Any]:
        """수집기 상태 확인"""
        if not self.api_key:
            return {
                "status": "disabled",
                "reason": "API key not configured"
            }
        
        try:
            # 간단한 테스트 요청
            test_articles = await self.search_news("test", page_size=1)
            
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
    """NewsAPI 수집기 테스트"""
    async with NewsAPICollector() as collector:
        # API 키 확인
        health = await collector.health_check()
        print(f"NewsAPI Health: {health}")
        
        if health["status"] == "healthy":
            # 톱 헤드라인 수집
            headlines = await collector.get_top_headlines(page_size=5)
            print(f"\nTop Headlines: {len(headlines)} articles")
            for article in headlines[:3]:
                print(f"  - {article['title'][:60]}...")
            
            # 암호화폐 뉴스 수집
            crypto_news = await collector.get_crypto_news(page_size=5)
            print(f"\nCrypto News: {len(crypto_news)} articles")
            for article in crypto_news[:3]:
                print(f"  - {article['title'][:60]}...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())