#!/usr/bin/env python3
"""
Advanced News Query Engine
Feedly Power Search Style 고급 뉴스 검색 엔진
"""

import re
from typing import List, Dict, Any, Set, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

from .base_collector import NewsArticle, NewsCategory


class QueryOperator(Enum):
    """검색 연산자"""
    AND = "AND"
    OR = "OR"
    NOT = "-"
    PHRASE = "\""
    WILDCARD = "*"
    GROUPING = "()"


class MarketImpactCategory(Enum):
    """시장 영향 카테고리"""
    PRICE_MOVEMENT = "price_movement"
    REGULATORY_POLICY = "regulatory_policy"
    INSTITUTIONAL_FLOW = "institutional_flow"
    TECHNOLOGY_SECURITY = "technology_security"
    MACRO_ECONOMIC = "macro_economic"
    CELEBRITY_INFLUENCE = "celebrity_influence"
    GEOPOLITICAL = "geopolitical"


@dataclass
class QueryToken:
    """검색 쿼리 토큰"""
    text: str
    operator: Optional[QueryOperator] = None
    is_phrase: bool = False
    is_negated: bool = False
    is_wildcard: bool = False
    weight: float = 1.0


@dataclass
class SearchQuery:
    """구조화된 검색 쿼리"""
    raw_query: str
    tokens: List[QueryToken]
    categories: List[MarketImpactCategory]
    priority_score: float
    estimated_impact: str  # HIGH, MEDIUM, LOW


class AdvancedNewsQueryEngine:
    """고급 뉴스 검색 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 시장 영향도별 키워드 매트릭스
        self.market_keywords = self._build_market_keywords()
        
        # 주요 인물 데이터베이스 (확장)
        self.key_figures = self._build_extended_key_figures()
        
        # 기관 투자자 및 거대 기업
        self.institutions = self._build_extended_institutions()
        
        # 규제 기관 및 정책 인물
        self.regulatory_figures = self._build_regulatory_figures()
        
        # 국제/지정학 인물
        self.geopolitical_figures = self._build_geopolitical_figures()
        
        # 사전 정의된 고영향 쿼리 패턴
        self.predefined_queries = self._build_predefined_queries()
    
    def _build_market_keywords(self) -> Dict[MarketImpactCategory, Dict[str, Dict[str, float]]]:
        """시장 영향도별 키워드 매트릭스 구축"""
        return {
            MarketImpactCategory.PRICE_MOVEMENT: {
                "bullish": {
                    "surge": 3.0, "soar": 3.0, "rally": 2.5, "bullish": 3.0,
                    "breakout": 2.5, "moon": 3.0, "pump": 2.5, "climb": 2.0,
                    "gain": 2.0, "rise": 1.5, "jump": 2.5, "spike": 2.5,
                    "ath": 3.0, "all-time high": 3.0, "new high": 2.5
                },
                "bearish": {
                    "crash": -3.0, "plunge": -3.0, "dump": -2.5, "bearish": -3.0,
                    "collapse": -3.0, "tumble": -2.5, "slump": -2.5, "drop": -2.0,
                    "fall": -2.0, "decline": -2.0, "correction": -1.5, "dip": -1.0
                }
            },
            
            MarketImpactCategory.REGULATORY_POLICY: {
                "negative": {
                    "ban": -3.0, "crackdown": -3.0, "lawsuit": -2.5, "prohibit": -3.0,
                    "illegal": -3.0, "fraud investigation": -3.0, "SEC lawsuit": -3.0,
                    "criminal charges": -3.0, "ponzi scheme": -3.0, "restrict": -2.5,
                    "regulate": -2.0, "compliance": -1.5
                },
                "positive": {
                    "approval": 3.0, "approved": 3.0, "ETF approval": 3.0,
                    "regulatory clarity": 2.5, "legal framework": 2.0,
                    "compliance framework": 2.0, "endorsement": 2.5
                }
            },
            
            MarketImpactCategory.INSTITUTIONAL_FLOW: {
                "inflow": {
                    "institutional adoption": 3.0, "corporate treasury": 3.0,
                    "pension fund": 2.5, "sovereign wealth": 3.0, "etf inflow": 2.5,
                    "whale accumulation": 2.0, "institutional buying": 2.5,
                    "corporate purchase": 2.5, "treasury allocation": 3.0
                },
                "outflow": {
                    "institutional exodus": -3.0, "grayscale liquidation": -3.0,
                    "tesla sells": -3.0, "microstrategy sells": -3.0,
                    "etf outflow": -2.5, "whale selling": -2.0, "mass liquidation": -3.0
                }
            },
            
            MarketImpactCategory.TECHNOLOGY_SECURITY: {
                "positive": {
                    "network upgrade": 2.0, "protocol improvement": 2.0,
                    "lightning network": 1.5, "taproot": 2.0, "scalability": 1.5,
                    "hash rate ath": 2.5, "mining innovation": 2.0
                },
                "negative": {
                    "51% attack": -3.0, "blockchain attack": -3.0, "hack": -3.0,
                    "exchange hack": -3.0, "wallet breach": -3.0, "smart contract exploit": -3.0,
                    "rug pull": -3.0, "exit scam": -3.0, "vulnerability": -2.5
                }
            },
            
            MarketImpactCategory.MACRO_ECONOMIC: {
                "monetary_policy": {
                    "fed rate hike": -2.0, "emergency rate cut": 2.5, "quantitative easing": 2.5,
                    "tapering": -2.0, "hawkish": -2.0, "dovish": 2.0,
                    "inflation hedge": 2.0, "currency debasement": 2.5, "dollar weakness": 2.0
                },
                "economic_indicators": {
                    "high inflation": 2.0, "recession": 1.5, "economic uncertainty": 1.5,
                    "market volatility": -1.0, "risk-off": -2.0, "risk-on": 2.0,
                    "safe haven": 2.5, "store of value": 2.0
                }
            }
        }
    
    def _build_extended_key_figures(self) -> Dict[str, Dict[str, Any]]:
        """확장된 주요 인물 데이터베이스"""
        return {
            # 미국 정치·경제 주요 인물
            "donald trump": {
                "role": "US President", "impact_weight": 3.0,
                "keywords": ["donald trump", "president trump", "trump administration", "trump", "white house"],
                "category": MarketImpactCategory.REGULATORY_POLICY
            },
            "jerome powell": {
                "role": "Fed Chairman", "impact_weight": 3.0,
                "keywords": ["jerome powell", "fed chairman", "federal reserve chair", "powell"],
                "category": MarketImpactCategory.MACRO_ECONOMIC
            },
            "janet yellen": {
                "role": "Treasury Secretary", "impact_weight": 2.5,
                "keywords": ["janet yellen", "treasury secretary", "yellen"],
                "category": MarketImpactCategory.REGULATORY_POLICY
            },
            "gary gensler": {
                "role": "SEC Chairman", "impact_weight": 3.0,
                "keywords": ["gary gensler", "sec chairman", "gensler", "securities exchange commission"],
                "category": MarketImpactCategory.REGULATORY_POLICY
            },
            "rostin behnam": {
                "role": "CFTC Chairman", "impact_weight": 2.5,
                "keywords": ["rostin behnam", "cftc chairman", "behnam", "commodity futures"],
                "category": MarketImpactCategory.REGULATORY_POLICY
            },
            
            # 기관 투자 및 거대 기업 인물
            "elon musk": {
                "role": "Tesla/SpaceX CEO", "impact_weight": 3.0,
                "keywords": ["elon musk", "tesla ceo", "spacex", "musk"],
                "category": MarketImpactCategory.CELEBRITY_INFLUENCE
            },
            "michael saylor": {
                "role": "MicroStrategy Chairman", "impact_weight": 3.0,
                "keywords": ["michael saylor", "microstrategy", "saylor", "mstr"],
                "category": MarketImpactCategory.INSTITUTIONAL_FLOW
            },
            "larry fink": {
                "role": "BlackRock CEO", "impact_weight": 2.5,
                "keywords": ["larry fink", "blackrock ceo", "fink", "blackrock"],
                "category": MarketImpactCategory.INSTITUTIONAL_FLOW
            },
            "cathie wood": {
                "role": "ARK Invest CEO", "impact_weight": 2.0,
                "keywords": ["cathie wood", "ark invest", "cathie", "arkk"],
                "category": MarketImpactCategory.INSTITUTIONAL_FLOW
            },
            "brian armstrong": {
                "role": "Coinbase CEO", "impact_weight": 2.0,
                "keywords": ["brian armstrong", "coinbase ceo", "armstrong"],
                "category": MarketImpactCategory.TECHNOLOGY_SECURITY
            },
            "changpeng zhao": {
                "role": "Former Binance CEO", "impact_weight": 2.5,
                "keywords": ["changpeng zhao", "cz", "binance ceo", "zhao"],
                "category": MarketImpactCategory.REGULATORY_POLICY
            }
        }
    
    def _build_extended_institutions(self) -> Dict[str, Dict[str, Any]]:
        """확장된 기관 데이터베이스"""
        return {
            # 코퍼레이트
            "tesla": {"impact_weight": 3.0, "type": "corporate", "btc_holding": True},
            "microstrategy": {"impact_weight": 3.0, "type": "corporate", "btc_holding": True},
            "square": {"impact_weight": 2.0, "type": "corporate", "btc_holding": True},
            
            # 자산 운용사
            "blackrock": {"impact_weight": 3.0, "type": "asset_manager", "etf_provider": True},
            "fidelity": {"impact_weight": 2.5, "type": "asset_manager", "etf_provider": True},
            "ark invest": {"impact_weight": 2.0, "type": "asset_manager", "etf_provider": True},
            "grayscale": {"impact_weight": 2.5, "type": "crypto_fund", "btc_holding": True},
            
            # 투자은행
            "goldman sachs": {"impact_weight": 2.5, "type": "investment_bank"},
            "jpmorgan": {"impact_weight": 2.5, "type": "investment_bank"},
            "morgan stanley": {"impact_weight": 2.0, "type": "investment_bank"},
            
            # 거래소
            "coinbase": {"impact_weight": 2.5, "type": "exchange"},
            "binance": {"impact_weight": 2.5, "type": "exchange"},
            "kraken": {"impact_weight": 1.5, "type": "exchange"}
        }
    
    def _build_regulatory_figures(self) -> Dict[str, Dict[str, Any]]:
        """규제 기관 인물"""
        return {
            "michael barr": {
                "role": "Fed Vice Chair for Supervision", "impact_weight": 2.0,
                "keywords": ["michael barr", "fed supervision", "barr"]
            },
            "neel kashkari": {
                "role": "Minneapolis Fed President", "impact_weight": 1.5,
                "keywords": ["neel kashkari", "kashkari", "minneapolis fed"]
            },
            "lisa cook": {
                "role": "Fed Governor", "impact_weight": 1.5,
                "keywords": ["lisa cook", "fed governor cook"]
            },
            "philip jefferson": {
                "role": "Fed Governor", "impact_weight": 1.5,
                "keywords": ["philip jefferson", "fed governor jefferson"]
            }
        }
    
    def _build_geopolitical_figures(self) -> Dict[str, Dict[str, Any]]:
        """지정학적 인물"""
        return {
            "xi jinping": {
                "role": "Chinese President", "impact_weight": 2.5,
                "keywords": ["xi jinping", "chinese president", "china policy"],
                "region": "China"
            },
            "vladimir putin": {
                "role": "Russian President", "impact_weight": 2.0,
                "keywords": ["vladimir putin", "putin", "russia sanctions"],
                "region": "Russia"
            },
            "christine lagarde": {
                "role": "ECB President", "impact_weight": 2.0,
                "keywords": ["christine lagarde", "ecb president", "lagarde"],
                "region": "Europe"
            }
        }
    
    def _build_predefined_queries(self) -> Dict[str, SearchQuery]:
        """사전 정의된 고영향 검색 쿼리"""
        queries = {}
        
        # Critical Impact Queries
        critical_patterns = [
            '("Jerome Powell" OR "Fed Chairman") AND ("emergency" OR "rate cut" OR "rate hike")',
            '("SEC" OR "Gary Gensler") AND ("Bitcoin ETF" OR "ETF approval" OR "ETF rejection")',
            '("Tesla" OR "Elon Musk") AND ("Bitcoin" OR "sell" OR "buy" OR "treasury")',
            '("MicroStrategy" OR "Michael Saylor") AND ("Bitcoin" OR "purchase" OR "sale")',
            '"Bitcoin ban" OR "crypto ban" OR "exchange shutdown"',
            '("hack" OR "breach") AND ("exchange" OR "wallet" OR "Bitcoin")',
            '"market crash" OR "flash crash" OR "liquidation cascade"'
        ]
        
        for i, pattern in enumerate(critical_patterns):
            queries[f"critical_{i}"] = SearchQuery(
                raw_query=pattern,
                tokens=self.parse_query(pattern),
                categories=[MarketImpactCategory.REGULATORY_POLICY],
                priority_score=1.0,
                estimated_impact="CRITICAL"
            )
        
        # High Impact Queries  
        high_impact_patterns = [
            '("BlackRock" OR "Fidelity") AND ("Bitcoin ETF" OR "crypto fund")',
            '("FOMC" OR "Fed meeting") AND ("interest rate" OR "monetary policy")',
            '("inflation" OR "CPI") AND ("Bitcoin" OR "hedge" OR "store of value")',
            '("institutional" OR "corporate") AND ("Bitcoin adoption" OR "treasury")',
            '"whale movement" OR "large transaction" OR "institutional flow"'
        ]
        
        for i, pattern in enumerate(high_impact_patterns):
            queries[f"high_{i}"] = SearchQuery(
                raw_query=pattern,
                tokens=self.parse_query(pattern),
                categories=[MarketImpactCategory.INSTITUTIONAL_FLOW],
                priority_score=0.8,
                estimated_impact="HIGH"
            )
        
        return queries
    
    def parse_query(self, query_string: str) -> List[QueryToken]:
        """검색 쿼리 파싱"""
        tokens = []
        
        # 정규식 패턴들
        phrase_pattern = r'"([^"]*)"'
        operator_pattern = r'\b(AND|OR)\b'
        negation_pattern = r'-(\w+)'
        wildcard_pattern = r'(\w+)\*'
        grouping_pattern = r'[()]'
        
        # 토큰화 과정
        remaining = query_string
        
        # 1. 구문(phrase) 추출
        phrases = re.findall(phrase_pattern, remaining)
        for phrase in phrases:
            tokens.append(QueryToken(
                text=phrase,
                operator=QueryOperator.PHRASE,
                is_phrase=True
            ))
            remaining = remaining.replace(f'"{phrase}"', '', 1)
        
        # 2. 부정 키워드 추출
        negations = re.findall(negation_pattern, remaining)
        for neg in negations:
            tokens.append(QueryToken(
                text=neg,
                operator=QueryOperator.NOT,
                is_negated=True
            ))
            remaining = remaining.replace(f'-{neg}', '', 1)
        
        # 3. 와일드카드 추출
        wildcards = re.findall(wildcard_pattern, remaining)
        for wild in wildcards:
            tokens.append(QueryToken(
                text=wild,
                operator=QueryOperator.WILDCARD,
                is_wildcard=True
            ))
            remaining = remaining.replace(f'{wild}*', '', 1)
        
        # 4. AND/OR 연산자 추출
        operators = re.findall(operator_pattern, remaining)
        for op in operators:
            tokens.append(QueryToken(
                text="",
                operator=QueryOperator.AND if op == "AND" else QueryOperator.OR
            ))
            remaining = remaining.replace(op, '', 1)
        
        # 5. 나머지 단어들
        words = [w.strip() for w in remaining.split() if w.strip() and w not in ['(', ')']]
        for word in words:
            if word:
                tokens.append(QueryToken(text=word))
        
        return tokens
    
    def build_smart_queries_for_category(self, category: MarketImpactCategory, 
                                       impact_level: str = "HIGH") -> List[SearchQuery]:
        """카테고리별 스마트 쿼리 생성"""
        queries = []
        
        if category == MarketImpactCategory.REGULATORY_POLICY:
            patterns = [
                '("SEC" OR "CFTC" OR "Treasury") AND ("Bitcoin" OR "crypto")',
                '("Gary Gensler" OR "Jerome Powell" OR "Janet Yellen") AND "Bitcoin"',
                '("Donald Trump" OR "President Trump") AND ("Bitcoin" OR "crypto policy")',
                '"Bitcoin ETF" OR "crypto ETF" OR "ETF approval"',
                '("regulation" OR "ban" OR "crackdown") AND "cryptocurrency"',
            ]
        
        elif category == MarketImpactCategory.INSTITUTIONAL_FLOW:
            patterns = [
                '("Tesla" OR "MicroStrategy" OR "BlackRock") AND "Bitcoin"',
                '("institutional" OR "corporate") AND ("adoption" OR "purchase")',
                '"whale movement" OR "large holders" OR "institutional flow"',
                '("ETF inflow" OR "ETF outflow") AND "Bitcoin"'
            ]
        
        elif category == MarketImpactCategory.MACRO_ECONOMIC:
            patterns = [
                '("Fed" OR "FOMC") AND ("rate" OR "monetary policy")',
                '("inflation" OR "CPI" OR "PPI") AND ("Bitcoin" OR "hedge")',
                '"recession" OR "economic uncertainty" OR "market volatility"',
                '("dollar" OR "DXY") AND ("weakness" OR "strength")'
            ]
        
        elif category == MarketImpactCategory.TECHNOLOGY_SECURITY:
            patterns = [
                '"network upgrade" OR "protocol update" OR "hard fork"',
                '("hack" OR "breach" OR "exploit") AND ("exchange" OR "DeFi")',
                '"hash rate" OR "mining difficulty" OR "network security"',
                '"scalability" OR "lightning network" OR "layer 2"'
            ]
        
        else:
            patterns = ['"Bitcoin" OR "cryptocurrency" OR "crypto market"']
        
        for pattern in patterns:
            queries.append(SearchQuery(
                raw_query=pattern,
                tokens=self.parse_query(pattern),
                categories=[category],
                priority_score=0.8 if impact_level == "HIGH" else 0.6,
                estimated_impact=impact_level
            ))
        
        return queries
    
    def evaluate_article_match(self, article: NewsArticle, query: SearchQuery) -> Tuple[bool, float]:
        """기사와 쿼리 매칭 평가"""
        
        # 전체 텍스트 준비
        full_text = f"{article.title} {article.summary} {article.content}".lower()
        
        match_score = 0.0
        total_tokens = len([t for t in query.tokens if t.text])
        
        if total_tokens == 0:
            return False, 0.0
        
        matched_tokens = 0
        
        for token in query.tokens:
            if not token.text:
                continue
                
            token_text = token.text.lower()
            
            if token.is_phrase:
                # 구문 매칭
                if token_text in full_text:
                    matched_tokens += 1
                    match_score += 2.0 * token.weight  # 구문 매칭은 높은 점수
            
            elif token.is_wildcard:
                # 와일드카드 매칭
                pattern = token_text.replace('*', '.*')
                if re.search(pattern, full_text):
                    matched_tokens += 1
                    match_score += 1.5 * token.weight
            
            elif token.is_negated:
                # 부정 키워드 - 존재하면 점수 감소
                if token_text in full_text:
                    match_score -= 3.0
                else:
                    matched_tokens += 1  # 부정 조건 만족
            
            else:
                # 일반 키워드 매칭
                if token_text in full_text:
                    matched_tokens += 1
                    match_score += 1.0 * token.weight
        
        # 매칭 비율 계산
        match_ratio = matched_tokens / total_tokens
        
        # 추가 점수 계산 (인물, 기관 언급)
        bonus_score = self._calculate_entity_bonus(full_text)
        
        final_score = (match_score + bonus_score) * match_ratio
        
        # 최소 매칭 비율 체크
        is_match = match_ratio >= 0.6 and final_score > 2.0
        
        return is_match, final_score
    
    def _calculate_entity_bonus(self, text: str) -> float:
        """엔티티 언급 보너스 점수"""
        bonus = 0.0
        
        # 주요 인물 언급
        for person, data in self.key_figures.items():
            for keyword in data["keywords"]:
                if keyword in text:
                    bonus += data["impact_weight"] * 0.5
                    break
        
        # 기관 언급
        for institution, data in self.institutions.items():
            if institution in text:
                bonus += data["impact_weight"] * 0.3
        
        return min(bonus, 5.0)  # 최대 5점으로 제한
    
    def search_articles_with_advanced_query(self, articles: List[NewsArticle], 
                                          query: Union[str, SearchQuery],
                                          min_score: float = 2.0) -> List[Tuple[NewsArticle, float]]:
        """고급 쿼리로 기사 검색"""
        
        if isinstance(query, str):
            query = SearchQuery(
                raw_query=query,
                tokens=self.parse_query(query),
                categories=[MarketImpactCategory.PRICE_MOVEMENT],
                priority_score=1.0,
                estimated_impact="MEDIUM"
            )
        
        matched_articles = []
        
        for article in articles:
            is_match, score = self.evaluate_article_match(article, query)
            
            if is_match and score >= min_score:
                matched_articles.append((article, score))
        
        # 점수별 정렬
        matched_articles.sort(key=lambda x: x[1], reverse=True)
        
        return matched_articles
    
    def get_trending_queries(self, timeframe_hours: int = 24) -> List[SearchQuery]:
        """트렌딩 쿼리 생성"""
        
        # 시간대별 중요도 조정
        current_hour = datetime.now().hour
        
        queries = []
        
        # 미국 시장 시간 (14:30-21:00 KST)
        if 6 <= current_hour <= 13:  # UTC 기준
            queries.extend([
                SearchQuery(
                    raw_query='("Jerome Powell" OR "Fed") AND ("rate" OR "policy")',
                    tokens=self.parse_query('("Jerome Powell" OR "Fed") AND ("rate" OR "policy")'),
                    categories=[MarketImpactCategory.MACRO_ECONOMIC],
                    priority_score=1.0,
                    estimated_impact="HIGH"
                ),
                SearchQuery(
                    raw_query='("SEC" OR "Gary Gensler") AND ("Bitcoin ETF" OR "crypto")',
                    tokens=self.parse_query('("SEC" OR "Gary Gensler") AND ("Bitcoin ETF" OR "crypto")'),
                    categories=[MarketImpactCategory.REGULATORY_POLICY],
                    priority_score=1.0,
                    estimated_impact="HIGH"
                )
            ])
        
        # 아시아 시장 시간
        elif 22 <= current_hour or current_hour <= 5:
            queries.extend([
                SearchQuery(
                    raw_query='("China" OR "Xi Jinping") AND ("Bitcoin" OR "crypto ban")',
                    tokens=self.parse_query('("China" OR "Xi Jinping") AND ("Bitcoin" OR "crypto ban")'),
                    categories=[MarketImpactCategory.GEOPOLITICAL],
                    priority_score=0.8,
                    estimated_impact="MEDIUM"
                )
            ])
        
        # 항상 모니터링할 쿼리
        always_queries = [
            '("Elon Musk" OR "Tesla") AND "Bitcoin"',
            '("Michael Saylor" OR "MicroStrategy") AND "Bitcoin"',
            '"flash crash" OR "market crash" OR "liquidation"',
            '"exchange hack" OR "wallet hack" OR "DeFi exploit"'
        ]
        
        for query_str in always_queries:
            queries.append(SearchQuery(
                raw_query=query_str,
                tokens=self.parse_query(query_str),
                categories=[MarketImpactCategory.CELEBRITY_INFLUENCE],
                priority_score=0.9,
                estimated_impact="HIGH"
            ))
        
        return queries


# 사용 예제
async def demo_advanced_query_engine():
    """고급 쿼리 엔진 데모"""
    
    engine = AdvancedNewsQueryEngine()
    
    print("🔍 Advanced News Query Engine Demo")
    print("=" * 50)
    
    # 샘플 쿼리들
    sample_queries = [
        '("Jerome Powell" OR "Fed Chairman") AND ("rate cut" OR "monetary policy")',
        '"Bitcoin ETF" AND ("approval" OR "rejection") -"fake news"',
        '("Elon Musk" OR "Tesla") AND "Bitcoin" AND ("buy" OR "sell")',
        '"crypto*" AND ("regulation" OR "ban") AND ("SEC" OR "CFTC")',
        '("institutional" OR "corporate") AND "Bitcoin adoption"'
    ]
    
    print("📝 Sample Query Parsing:")
    for query in sample_queries:
        print(f"\nQuery: {query}")
        tokens = engine.parse_query(query)
        print("Tokens:")
        for token in tokens:
            print(f"  - Text: '{token.text}', Operator: {token.operator}, "
                  f"Phrase: {token.is_phrase}, Negated: {token.is_negated}, "
                  f"Wildcard: {token.is_wildcard}")
    
    print(f"\n\n🎯 Key Figures Database: {len(engine.key_figures)} people")
    for name, data in list(engine.key_figures.items())[:5]:
        print(f"  - {name.title()}: {data['role']} (Impact: {data['impact_weight']})")
    
    print(f"\n🏢 Institutions Database: {len(engine.institutions)} entities")
    for name, data in list(engine.institutions.items())[:5]:
        print(f"  - {name.title()}: {data['type']} (Impact: {data['impact_weight']})")
    
    print(f"\n📊 Predefined Queries: {len(engine.predefined_queries)} patterns")
    for name, query in list(engine.predefined_queries.items())[:3]:
        print(f"  - {name}: {query.raw_query}")
        print(f"    Impact: {query.estimated_impact}, Priority: {query.priority_score}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_advanced_query_engine())