#!/usr/bin/env python3
"""
Power Search Engine for AuroraQ Sentiment Service
고급 쿼리 기능으로 시장 관련성 높은 뉴스 수집 (최대 15개 쿼리 제한)
"""

import asyncio
import aiohttp
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class SearchOperator(Enum):
    """검색 연산자"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    QUOTE = "\""
    WILDCARD = "*"
    SITE = "site:"
    FILETYPE = "filetype:"
    INTITLE = "intitle:"
    INTEXT = "intext:"

@dataclass
class SearchQuery:
    """검색 쿼리 데이터 클래스"""
    terms: List[str]
    required_terms: List[str] = field(default_factory=list)
    excluded_terms: List[str] = field(default_factory=list)
    exact_phrases: List[str] = field(default_factory=list)
    site_filters: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, datetime]] = None
    language: str = "en"
    region: str = "US"
    
    def build_query_string(self) -> str:
        """쿼리 문자열 생성"""
        query_parts = []
        
        # 기본 검색어 (OR 연결)
        if self.terms:
            query_parts.append("(" + " OR ".join(self.terms) + ")")
        
        # 필수 포함 단어 (AND 연결)
        for term in self.required_terms:
            query_parts.append(f"+{term}")
        
        # 제외 단어
        for term in self.excluded_terms:
            query_parts.append(f"-{term}")
        
        # 정확한 구문
        for phrase in self.exact_phrases:
            query_parts.append(f'"{phrase}"')
        
        # 사이트 필터
        for site in self.site_filters:
            query_parts.append(f"site:{site}")
        
        return " ".join(query_parts)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    title: str
    snippet: str
    url: str
    source: str
    published_date: Optional[datetime]
    relevance_score: float
    market_relevance: float
    search_query: str
    hash_id: str = field(init=False)
    
    def __post_init__(self):
        """해시 ID 생성"""
        content_for_hash = f"{self.title}{self.url}"
        self.hash_id = hashlib.md5(content_for_hash.encode()).hexdigest()

class PowerSearchEngine:
    """고급 검색 엔진"""
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        초기화
        
        Args:
            api_keys: API 키 딕셔너리
                - google_search_key: Google Custom Search API 키
                - google_cx: Google Custom Search Engine ID
                - bing_search_key: Bing Search API 키 (선택사항)
        """
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        self.query_count = 0
        self.max_queries = 15  # 최대 쿼리 제한
        self.collected_hashes: Set[str] = set()
        
        # 시장 관련성 키워드
        self.market_keywords = {
            "high": ["price", "trading", "market", "investment", "finance", 
                    "economic", "monetary", "fiscal", "regulation", "policy"],
            "medium": ["business", "company", "corporate", "industry", "sector",
                      "growth", "earnings", "revenue", "profit", "loss"],
            "low": ["news", "update", "report", "analysis", "opinion", "view"]
        }
        
        # 신뢰할 수 있는 뉴스 사이트
        self.trusted_sites = [
            "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
            "cnbc.com", "marketwatch.com", "yahoo.com", "google.com",
            "coindesk.com", "cointelegraph.com", "decrypt.co"
        ]
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'AuroraQ-PowerSearch/1.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self.session:
            await self.session.close()
    
    def _check_query_limit(self) -> bool:
        """쿼리 한도 확인"""
        if self.query_count >= self.max_queries:
            logger.warning(f"Query limit reached: {self.query_count}/{self.max_queries}")
            return False
        return True
    
    def _calculate_market_relevance(self, text: str) -> float:
        """시장 관련성 점수 계산"""
        text_lower = text.lower()
        score = 0.0
        
        # 가중치별 키워드 매칭
        for level, keywords in self.market_keywords.items():
            weight = {"high": 1.0, "medium": 0.6, "low": 0.3}[level]
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score += matches * weight
        
        # 정규화 (0.0 ~ 1.0)
        max_possible_score = sum(len(keywords) for keywords in self.market_keywords.values())
        return min(1.0, score / max_possible_score * 2)  # 2배 승수로 민감도 증가
    
    def _extract_date_from_snippet(self, snippet: str) -> Optional[datetime]:
        """스니펫에서 날짜 추출 (간단한 패턴 매칭)"""
        
        # 날짜 패턴들
        date_patterns = [
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
            r'(\d{4})-(\d{2})-(\d{2})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                try:
                    if "Jan|Feb|Mar" in pattern:
                        # 월 이름 형식
                        day, month_str, year = match.groups()
                        month_map = {
                            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                        }
                        month = month_map.get(month_str, 1)
                        return datetime(int(year), month, int(day))
                    
                    elif "-" in pattern:
                        # YYYY-MM-DD 형식
                        year, month, day = match.groups()
                        return datetime(int(year), int(month), int(day))
                    
                    else:
                        # MM/DD/YYYY 형식
                        month, day, year = match.groups()
                        return datetime(int(year), int(month), int(day))
                        
                except ValueError:
                    continue
        
        return None
    
    async def search_google_custom(self, query: SearchQuery, num_results: int = 10) -> List[SearchResult]:
        """Google Custom Search API 검색"""
        
        if not self._check_query_limit():
            return []
        
        if not self.api_keys.get("google_search_key") or not self.api_keys.get("google_cx"):
            logger.warning("Google Custom Search API credentials not provided")
            return []
        
        logger.info(f"Performing Google Custom Search: {query.build_query_string()}")
        
        try:
            self.query_count += 1
            
            # API 요청 파라미터
            params = {
                'key': self.api_keys["google_search_key"],
                'cx': self.api_keys["google_cx"],
                'q': query.build_query_string(),
                'num': min(num_results, 10),  # Google API 최대 10개
                'lr': f'lang_{query.language}',
                'gl': query.region.lower(),
                'sort': 'date'  # 날짜순 정렬
            }
            
            # 날짜 범위 추가
            if query.date_range:
                start_date = query.date_range.get('start', datetime.now() - timedelta(days=7))
                end_date = query.date_range.get('end', datetime.now())
                params['dateRestrict'] = f"d{(end_date - start_date).days}"
            
            async with self.session.get(
                'https://www.googleapis.com/customsearch/v1',
                params=params
            ) as response:
                response.raise_for_status()
                data = await response.json()
            
            results = []
            for item in data.get('items', []):
                try:
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')
                    url = item.get('link', '')
                    source = item.get('displayLink', '')
                    
                    # 신뢰할 수 있는 사이트인지 확인
                    is_trusted = any(site in url.lower() for site in self.trusted_sites)
                    
                    # 시장 관련성 점수 계산
                    market_relevance = self._calculate_market_relevance(f"{title} {snippet}")
                    
                    # 관련성이 낮거나 신뢰할 수 없는 사이트는 필터링
                    if market_relevance < 0.3 and not is_trusted:
                        continue
                    
                    # 날짜 추출
                    published_date = self._extract_date_from_snippet(snippet)
                    
                    # 기본 관련성 점수 (Google 검색 순위 기반)
                    relevance_score = 1.0 - (len(results) * 0.1)  # 순위가 낮을수록 점수 감소
                    
                    search_result = SearchResult(
                        title=title,
                        snippet=snippet,
                        url=url,
                        source=source,
                        published_date=published_date,
                        relevance_score=relevance_score,
                        market_relevance=market_relevance,
                        search_query=query.build_query_string()
                    )
                    
                    if search_result.hash_id not in self.collected_hashes:
                        results.append(search_result)
                        self.collected_hashes.add(search_result.hash_id)
                
                except Exception as e:
                    logger.error(f"Error parsing Google search result: {e}")
                    continue
            
            logger.info(f"Google Custom Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Google Custom Search failed: {e}")
            return []
    
    def create_crypto_queries(self, symbol: str = "BTC") -> List[SearchQuery]:
        """암호화폐 관련 고급 쿼리 생성"""
        
        # 심볼별 키워드 매핑
        symbol_keywords = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth", "ether"],
            "CRYPTO": ["cryptocurrency", "crypto", "blockchain"]
        }
        
        base_keywords = symbol_keywords.get(symbol.upper(), [symbol.lower()])
        
        queries = []
        
        # 1. 가격 관련 뉴스
        queries.append(SearchQuery(
            terms=base_keywords,
            required_terms=["price", "surge", "drop", "rally", "crash"],
            site_filters=["coindesk.com", "cointelegraph.com", "decrypt.co"],
            date_range={"start": datetime.now() - timedelta(days=1), "end": datetime.now()}
        ))
        
        # 2. 기관 투자 관련
        queries.append(SearchQuery(
            terms=base_keywords,
            required_terms=["institutional", "investment", "fund", "etf"],
            excluded_terms=["scam", "hack"],
            site_filters=["bloomberg.com", "reuters.com", "wsj.com"]
        ))
        
        # 3. 규제 관련
        queries.append(SearchQuery(
            terms=base_keywords,
            required_terms=["regulation", "sec", "government", "policy"],
            exact_phrases=["regulatory approval", "legal framework"],
            site_filters=["reuters.com", "cnbc.com"]
        ))
        
        # 4. 기술적 발전
        if symbol.upper() in ["ETH", "CRYPTO"]:
            queries.append(SearchQuery(
                terms=base_keywords,
                required_terms=["upgrade", "update", "development", "defi"],
                site_filters=["coindesk.com", "decrypt.co"]
            ))
        
        return queries[:4]  # 최대 4개 쿼리
    
    def create_finance_queries(self, category: str = "market") -> List[SearchQuery]:
        """금융 관련 고급 쿼리 생성"""
        
        queries = []
        
        # 1. 시장 분석
        queries.append(SearchQuery(
            terms=["market", "stock", "trading"],
            required_terms=["analysis", "outlook", "forecast"],
            site_filters=["marketwatch.com", "yahoo.com", "bloomberg.com"],
            date_range={"start": datetime.now() - timedelta(days=1), "end": datetime.now()}
        ))
        
        # 2. 경제 정책
        queries.append(SearchQuery(
            terms=["federal", "fed", "interest", "rate"],
            required_terms=["policy", "decision", "meeting"],
            exact_phrases=["interest rate", "monetary policy"],
            site_filters=["reuters.com", "wsj.com", "ft.com"]
        ))
        
        # 3. 기업 실적
        queries.append(SearchQuery(
            terms=["earnings", "revenue", "profit"],
            required_terms=["report", "results", "quarterly"],
            excluded_terms=["crypto", "bitcoin"],
            site_filters=["bloomberg.com", "cnbc.com"]
        ))
        
        return queries[:3]  # 최대 3개 쿼리
    
    async def power_search(self,
                         queries: List[SearchQuery],
                         max_results_per_query: int = 5) -> List[SearchResult]:
        """고급 검색 실행"""
        
        logger.info(f"Starting power search with {len(queries)} queries...")
        start_time = time.time()
        
        # 쿼리 수 제한 확인
        if len(queries) > self.max_queries:
            logger.warning(f"Too many queries ({len(queries)}), limiting to {self.max_queries}")
            queries = queries[:self.max_queries]
        
        # 병렬 검색 실행
        tasks = []
        for query in queries:
            if self._check_query_limit():
                tasks.append(
                    self.search_google_custom(query, max_results_per_query)
                )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 병합
        all_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search query {i} failed: {result}")
            elif isinstance(result, list):
                all_results.extend(result)
        
        # 중복 제거 및 정렬
        unique_results = []
        seen_hashes = set()
        
        for result in all_results:
            if result.hash_id not in seen_hashes:
                unique_results.append(result)
                seen_hashes.add(result.hash_id)
        
        # 시장 관련성과 관련성 점수로 정렬
        unique_results.sort(
            key=lambda x: (x.market_relevance * 0.6 + x.relevance_score * 0.4),
            reverse=True
        )
        
        search_time = time.time() - start_time
        
        logger.info(f"Power search completed: {len(unique_results)} unique results "
                   f"from {len(all_results)} total in {search_time:.2f}s "
                   f"({self.query_count}/{self.max_queries} queries used)")
        
        return unique_results
    
    async def search_crypto_news(self, symbol: str = "BTC", max_results: int = 20) -> List[SearchResult]:
        """암호화폐 뉴스 전용 검색"""
        
        queries = self.create_crypto_queries(symbol)
        results = await self.power_search(queries, max_results // len(queries))
        
        return results[:max_results]
    
    async def search_finance_news(self, category: str = "market", max_results: int = 15) -> List[SearchResult]:
        """금융 뉴스 전용 검색"""
        
        queries = self.create_finance_queries(category)
        results = await self.power_search(queries, max_results // len(queries))
        
        return results[:max_results]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        return {
            "queries_used": self.query_count,
            "queries_remaining": self.max_queries - self.query_count,
            "total_results_collected": len(self.collected_hashes),
            "trusted_sites": len(self.trusted_sites),
            "market_keyword_categories": len(self.market_keywords),
            "api_keys_configured": {
                "google_custom_search": bool(self.api_keys.get("google_search_key")),
                "bing_search": bool(self.api_keys.get("bing_search_key"))
            }
        }


# 테스트 코드
if __name__ == "__main__":
    import json
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def test_power_search():
        """파워 서치 테스트"""
        
        # API 키 설정 (실제 키로 교체 필요)
        api_keys = {
            "google_search_key": "your_google_api_key_here",
            "google_cx": "your_custom_search_engine_id_here"
        }
        
        if api_keys["google_search_key"] == "your_google_api_key_here":
            print("❌ Google Custom Search API 키를 설정해주세요!")
            return
        
        async with PowerSearchEngine(api_keys) as search_engine:
            print("=== 암호화폐 파워 서치 테스트 ===")
            
            # 비트코인 뉴스 검색
            btc_results = await search_engine.search_crypto_news("BTC", max_results=10)
            
            print(f"\n비트코인 검색 결과: {len(btc_results)}개")
            
            for i, result in enumerate(btc_results[:3], 1):
                print(f"\n{i}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   소스: {result.source}")
                print(f"   시장 관련성: {result.market_relevance:.2f}")
                print(f"   관련성: {result.relevance_score:.2f}")
                print(f"   스니펫: {result.snippet[:100]}...")
            
            print(f"\n=== 금융 시장 파워 서치 테스트 ===")
            
            # 금융 시장 뉴스 검색
            finance_results = await search_engine.search_finance_news("market", max_results=8)
            
            print(f"\n금융 시장 검색 결과: {len(finance_results)}개")
            
            for i, result in enumerate(finance_results[:2], 1):
                print(f"\n{i}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   소스: {result.source}")
                print(f"   시장 관련성: {result.market_relevance:.2f}")
            
            # 검색 통계
            stats = search_engine.get_search_stats()
            print(f"\n=== 검색 통계 ===")
            print(json.dumps(stats, indent=2))
    
    # 테스트 실행
    asyncio.run(test_power_search())