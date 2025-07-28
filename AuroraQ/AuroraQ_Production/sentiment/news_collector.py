#!/usr/bin/env python3
"""
뉴스 수집기
다양한 소스에서 뉴스 데이터를 수집하고 전처리
"""

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger("NewsCollector")

@dataclass
class NewsItem:
    """뉴스 아이템"""
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    category: str = "general"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class NewsCollector:
    """뉴스 수집기"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.collected_urls = set()  # 중복 방지
        
    def collect_crypto_news(self, hours_back: int = 24) -> List[NewsItem]:
        """암호화폐 뉴스 수집"""
        news_items = []
        
        # CoinDesk 뉴스
        try:
            coindesk_news = self._collect_coindesk_news(hours_back)
            news_items.extend(coindesk_news)
        except Exception as e:
            logger.error(f"CoinDesk 뉴스 수집 실패: {e}")
        
        # CoinTelegraph 뉴스
        try:
            cointelegraph_news = self._collect_cointelegraph_news(hours_back)
            news_items.extend(cointelegraph_news)
        except Exception as e:
            logger.error(f"CoinTelegraph 뉴스 수집 실패: {e}")
        
        # 중복 제거
        unique_news = self._remove_duplicates(news_items)
        logger.info(f"암호화폐 뉴스 수집 완료: {len(unique_news)}건")
        
        return unique_news
    
    def collect_financial_news(self, hours_back: int = 24) -> List[NewsItem]:
        """금융 뉴스 수집"""
        news_items = []
        
        # Yahoo Finance 뉴스
        try:
            yahoo_news = self._collect_yahoo_finance_news(hours_back)
            news_items.extend(yahoo_news)
        except Exception as e:
            logger.error(f"Yahoo Finance 뉴스 수집 실패: {e}")
        
        # Reuters 뉴스
        try:
            reuters_news = self._collect_reuters_news(hours_back)
            news_items.extend(reuters_news)
        except Exception as e:
            logger.error(f"Reuters 뉴스 수집 실패: {e}")
        
        unique_news = self._remove_duplicates(news_items)
        logger.info(f"금융 뉴스 수집 완료: {len(unique_news)}건")
        
        return unique_news
    
    def _collect_coindesk_news(self, hours_back: int) -> List[NewsItem]:
        """CoinDesk 뉴스 수집"""
        news_items = []
        
        # RSS 피드 사용
        rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        
        try:
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for item in items[:20]:  # 최대 20개
                try:
                    title = item.title.text.strip()
                    link = item.link.text.strip()
                    pub_date = item.pubDate.text.strip()
                    description = item.description.text.strip() if item.description else ""
                    
                    # 시간 파싱
                    pub_datetime = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                    pub_datetime = pub_datetime.replace(tzinfo=None)  # 시간대 제거
                    
                    if pub_datetime < cutoff_time:
                        continue
                    
                    if link not in self.collected_urls:
                        news_item = NewsItem(
                            title=title,
                            content=description,
                            url=link,
                            source="CoinDesk",
                            timestamp=pub_datetime,
                            category="crypto",
                            tags=["bitcoin", "cryptocurrency", "blockchain"]
                        )
                        news_items.append(news_item)
                        self.collected_urls.add(link)
                        
                except Exception as e:
                    logger.error(f"CoinDesk 아이템 파싱 실패: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"CoinDesk RSS 수집 실패: {e}")
        
        return news_items
    
    def _collect_cointelegraph_news(self, hours_back: int) -> List[NewsItem]:
        """CoinTelegraph 뉴스 수집"""
        news_items = []
        
        # API 엔드포인트 (실제로는 웹 스크래핑)
        base_url = "https://cointelegraph.com"
        
        try:
            response = self.session.get(f"{base_url}/news", timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 뉴스 아티클 찾기 (실제 HTML 구조에 맞게 조정 필요)
            articles = soup.find_all('article', class_='post-card-inline')[:10]
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for article in articles:
                try:
                    title_elem = article.find('h2') or article.find('h3')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text().strip()
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        continue
                    
                    relative_url = link_elem.get('href')
                    full_url = f"{base_url}{relative_url}" if relative_url.startswith('/') else relative_url
                    
                    if full_url not in self.collected_urls:
                        news_item = NewsItem(
                            title=title,
                            content="",  # 상세 내용은 별도 요청 필요
                            url=full_url,
                            source="CoinTelegraph",
                            timestamp=datetime.now(),  # 정확한 시간은 별도 파싱 필요
                            category="crypto",
                            tags=["cryptocurrency", "blockchain", "defi"]
                        )
                        news_items.append(news_item)
                        self.collected_urls.add(full_url)
                        
                except Exception as e:
                    logger.error(f"CoinTelegraph 아이템 파싱 실패: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"CoinTelegraph 수집 실패: {e}")
        
        return news_items
    
    def _collect_yahoo_finance_news(self, hours_back: int) -> List[NewsItem]:
        """Yahoo Finance 뉴스 수집"""
        news_items = []
        
        # Yahoo Finance RSS
        rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
        
        try:
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for item in items[:15]:
                try:
                    title = item.title.text.strip()
                    link = item.link.text.strip()
                    pub_date = item.pubDate.text.strip()
                    description = item.description.text.strip() if item.description else ""
                    
                    # 시간 파싱
                    pub_datetime = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z')
                    pub_datetime = pub_datetime.replace(tzinfo=None)
                    
                    if pub_datetime < cutoff_time:
                        continue
                    
                    if link not in self.collected_urls:
                        news_item = NewsItem(
                            title=title,
                            content=description,
                            url=link,
                            source="Yahoo Finance",
                            timestamp=pub_datetime,
                            category="finance",
                            tags=["stock", "market", "finance"]
                        )
                        news_items.append(news_item)
                        self.collected_urls.add(link)
                        
                except Exception as e:
                    logger.error(f"Yahoo Finance 아이템 파싱 실패: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Yahoo Finance RSS 수집 실패: {e}")
        
        return news_items
    
    def _collect_reuters_news(self, hours_back: int) -> List[NewsItem]:
        """Reuters 뉴스 수집"""
        news_items = []
        
        # Reuters Business RSS
        rss_url = "https://www.reuters.com/business/finance"
        
        try:
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Reuters 구조에 맞게 파싱 (실제 구조 확인 필요)
            articles = soup.find_all('div', class_='story-card')[:10]
            
            for article in articles:
                try:
                    title_elem = article.find('h3') or article.find('h2')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text().strip()
                    link_elem = title_elem.find('a') or article.find('a')
                    if not link_elem:
                        continue
                    
                    relative_url = link_elem.get('href')
                    full_url = f"https://www.reuters.com{relative_url}" if relative_url.startswith('/') else relative_url
                    
                    if full_url not in self.collected_urls:
                        news_item = NewsItem(
                            title=title,
                            content="",
                            url=full_url,
                            source="Reuters",
                            timestamp=datetime.now(),
                            category="finance",
                            tags=["business", "finance", "markets"]
                        )
                        news_items.append(news_item)
                        self.collected_urls.add(full_url)
                        
                except Exception as e:
                    logger.error(f"Reuters 아이템 파싱 실패: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Reuters 수집 실패: {e}")
        
        return news_items
    
    def _remove_duplicates(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """중복 뉴스 제거"""
        seen_titles = set()
        unique_items = []
        
        for item in news_items:
            title_key = item.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)
        
        return unique_items
    
    async def collect_news_async(self, hours_back: int = 24) -> List[NewsItem]:
        """비동기 뉴스 수집"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._collect_async_source(session, "crypto", hours_back),
                self._collect_async_source(session, "finance", hours_back)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_news = []
            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)
                else:
                    logger.error(f"비동기 수집 오류: {result}")
            
            return self._remove_duplicates(all_news)
    
    async def _collect_async_source(self, session: aiohttp.ClientSession, source_type: str, hours_back: int) -> List[NewsItem]:
        """비동기 소스별 수집"""
        # 비동기 구현 (추후 확장)
        return []