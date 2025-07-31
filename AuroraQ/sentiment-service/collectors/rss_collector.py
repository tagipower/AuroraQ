#!/usr/bin/env python3
"""RSS 피드 수집기"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

class RSSCollector:
    """RSS 피드 수집기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        
        # RSS 피드 소스
        self.rss_feeds = {
            "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "cointelegraph": "https://cointelegraph.com/rss",
            "decrypt": "https://decrypt.co/feed",
            "google_crypto": "https://news.google.com/rss/search?q=cryptocurrency+bitcoin&hl=en&gl=US&ceid=US:en",
            "yahoo_finance": "https://finance.yahoo.com/rss/topfinstories"
        }
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    async def collect_feed(self, feed_name: str, max_articles: int = 20) -> List[Dict[str, Any]]:
        """단일 RSS 피드 수집"""
        if feed_name not in self.rss_feeds:
            raise ValueError(f"Unknown feed: {feed_name}")
        
        url = self.rss_feeds[feed_name]
        articles = []
        
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to fetch {feed_name}: HTTP {response.status}")
                    return []
                
                content = await response.text()
                
            # 피드 파싱
            feed = feedparser.parse(content)
            
            if feed.bozo:
                self.logger.warning(f"Feed parsing warning for {feed_name}: {feed.bozo_exception}")
            
            # 기사 추출
            for entry in feed.entries[:max_articles]:
                article = {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "source": feed_name,
                    "collected_at": datetime.now().isoformat()
                }
                
                # 발행 날짜 파싱
                if entry.get("published_parsed"):
                    published_time = datetime(*entry.published_parsed[:6])
                    article["published_parsed"] = published_time.isoformat()
                
                articles.append(article)
            
            self.logger.info(f"Collected {len(articles)} articles from {feed_name}")
            return articles
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout collecting from {feed_name}")
            return []
        except Exception as e:
            self.logger.error(f"Error collecting from {feed_name}: {str(e)}")
            return []
    
    async def collect_all_feeds(self, max_articles_per_feed: int = 15) -> Dict[str, List[Dict[str, Any]]]:
        """모든 RSS 피드 수집"""
        results = {}
        
        # 병렬로 모든 피드 수집
        tasks = []
        for feed_name in self.rss_feeds.keys():
            task = self.collect_feed(feed_name, max_articles_per_feed)
            tasks.append((feed_name, task))
        
        # 결과 수집
        for feed_name, task in tasks:
            try:
                articles = await task
                results[feed_name] = articles
            except Exception as e:
                self.logger.error(f"Failed to collect {feed_name}: {str(e)}")
                results[feed_name] = []
        
        total_articles = sum(len(articles) for articles in results.values())
        self.logger.info(f"Collected total {total_articles} articles from {len(results)} feeds")
        
        return results
    
    def get_recent_articles(self, all_articles: Dict[str, List[Dict[str, Any]]], 
                          hours_back: int = 24) -> List[Dict[str, Any]]:
        """최근 기사만 필터링"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_articles = []
        
        for feed_name, articles in all_articles.items():
            for article in articles:
                # 발행 시간 확인
                if "published_parsed" in article:
                    try:
                        published_time = datetime.fromisoformat(article["published_parsed"])
                        if published_time >= cutoff_time:
                            recent_articles.append(article)
                    except:
                        # 파싱 실패 시 일단 포함
                        recent_articles.append(article)
                else:
                    # 발행 시간 정보가 없으면 일단 포함
                    recent_articles.append(article)
        
        # 발행 시간 역순 정렬
        recent_articles.sort(key=lambda x: x.get("published_parsed", ""), reverse=True)
        
        return recent_articles

# 사용 예제
async def main():
    """RSS 수집기 테스트"""
    async with RSSCollector() as collector:
        # 모든 피드 수집
        all_articles = await collector.collect_all_feeds(max_articles_per_feed=10)
        
        # 결과 출력
        for feed_name, articles in all_articles.items():
            print(f"\n{feed_name.upper()}: {len(articles)} articles")
            for article in articles[:3]:
                print(f"  - {article['title'][:60]}...")
        
        # 최근 24시간 기사만 필터링
        recent_articles = collector.get_recent_articles(all_articles, hours_back=24)
        print(f"\nRecent articles (24h): {len(recent_articles)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())