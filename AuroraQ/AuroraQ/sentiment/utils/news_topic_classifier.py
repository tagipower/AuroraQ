#!/usr/bin/env python3
"""
News Topic Classifier - 뉴스 토픽 자동 분류 시스템
거시경제, 규제, 기술, 시장 등 다양한 카테고리로 뉴스를 자동 분류
"""

import re
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class NewsTopicCategory(Enum):
    """뉴스 토픽 카테고리"""
    MACRO = "macro"                    # 거시경제 (금리, GDP, 인플레이션)
    REGULATION = "regulation"          # 규제/정책 (SEC, 법안, 정부 발표)
    TECHNOLOGY = "technology"          # 기술/개발 (블록체인, AI, 업그레이드)
    MARKET = "market"                  # 시장동향 (가격, 거래량, 투자)
    CORPORATE = "corporate"            # 기업/기관 (파트너십, 인수합병, 실적)
    SECURITY = "security"              # 보안/해킹 (취약점, 사고, 보안 업데이트)
    ADOPTION = "adoption"              # 채택/활용 (결제, 통합, 실사용)
    ANALYSIS = "analysis"              # 분석/전망 (리서치, 예측, 의견)
    EVENT = "event"                    # 이벤트/컨퍼런스 (행사, 발표, 미팅)
    OTHER = "other"                    # 기타

@dataclass
class TopicClassification:
    """토픽 분류 결과"""
    primary_topic: NewsTopicCategory
    secondary_topics: List[NewsTopicCategory] = field(default_factory=list)
    confidence_scores: Dict[NewsTopicCategory, float] = field(default_factory=dict)
    keywords_found: Dict[str, List[str]] = field(default_factory=dict)
    processing_time: float = 0.0
    
    @property
    def is_high_confidence(self) -> bool:
        """높은 신뢰도 여부"""
        return self.confidence_scores.get(self.primary_topic, 0) >= 0.7
    
    @property
    def is_multi_topic(self) -> bool:
        """다중 토픽 여부"""
        return len(self.secondary_topics) > 0

@dataclass
class TopicKeywordSet:
    """토픽별 키워드 세트"""
    strong_indicators: Set[str] = field(default_factory=set)
    moderate_indicators: Set[str] = field(default_factory=set)
    weak_indicators: Set[str] = field(default_factory=set)
    weight: float = 1.0

class NewsTopicClassifier:
    """뉴스 토픽 자동 분류기"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.2,
                 multi_topic_threshold: float = 0.15,
                 enable_context_analysis: bool = True):
        """
        초기화
        
        Args:
            confidence_threshold: 주요 토픽 판정 임계값
            multi_topic_threshold: 부가 토픽 판정 임계값
            enable_context_analysis: 문맥 기반 분석 활성화
        """
        self.confidence_threshold = confidence_threshold
        self.multi_topic_threshold = multi_topic_threshold
        self.enable_context_analysis = enable_context_analysis
        
        # 토픽별 키워드 정의
        self.topic_keywords = self._initialize_topic_keywords()
        
        # 통계 및 캐시
        self.classification_stats = {
            "total_classified": 0,
            "topic_distribution": {topic: 0 for topic in NewsTopicCategory},
            "avg_confidence": 0.0,
            "multi_topic_ratio": 0.0
        }
        
    def _initialize_topic_keywords(self) -> Dict[NewsTopicCategory, TopicKeywordSet]:
        """토픽별 키워드 초기화"""
        return {
            NewsTopicCategory.MACRO: TopicKeywordSet(
                strong_indicators={
                    'federal reserve', 'fed', 'interest rate', 'inflation', 'gdp', 'fomc',
                    'monetary policy', 'economic growth', 'recession', 'unemployment',
                    'cpi', 'pce', 'yield curve', 'quantitative easing', 'tapering',
                    'hawkish', 'dovish', 'basis points', 'economic indicator'
                },
                moderate_indicators={
                    'economy', 'growth', 'employment', 'jobs', 'wages', 'spending',
                    'consumer', 'business', 'manufacturing', 'services', 'trade',
                    'deficit', 'surplus', 'budget', 'fiscal', 'stimulus'
                },
                weak_indicators={
                    'market', 'finance', 'money', 'dollar', 'currency', 'global',
                    'international', 'domestic', 'outlook', 'forecast', 'data'
                },
                weight=1.2  # 거시경제는 더 높은 가중치
            ),
            
            NewsTopicCategory.REGULATION: TopicKeywordSet(
                strong_indicators={
                    'sec', 'regulation', 'regulatory', 'policy', 'law', 'legislation',
                    'compliance', 'enforcement', 'sanctions', 'ban', 'approval',
                    'license', 'framework', 'guidelines', 'rules', 'oversight',
                    'government', 'authority', 'commission', 'etf approval'
                },
                moderate_indicators={
                    'legal', 'court', 'lawsuit', 'fine', 'penalty', 'investigation',
                    'audit', 'review', 'proposal', 'draft', 'amendment', 'act',
                    'bill', 'statute', 'jurisdiction', 'regulator'
                },
                weak_indicators={
                    'official', 'statement', 'announce', 'decision', 'ruling',
                    'update', 'change', 'new', 'require', 'must', 'should'
                },
                weight=1.15
            ),
            
            NewsTopicCategory.TECHNOLOGY: TopicKeywordSet(
                strong_indicators={
                    'blockchain', 'smart contract', 'defi', 'nft', 'web3', 'layer 2',
                    'protocol', 'upgrade', 'mainnet', 'testnet', 'fork', 'consensus',
                    'algorithm', 'cryptography', 'hash', 'mining', 'staking',
                    'validator', 'node', 'scalability', 'interoperability'
                },
                moderate_indicators={
                    'technology', 'tech', 'innovation', 'development', 'platform',
                    'network', 'system', 'infrastructure', 'solution', 'integration',
                    'api', 'sdk', 'framework', 'library', 'tool'
                },
                weak_indicators={
                    'digital', 'online', 'software', 'hardware', 'compute', 'data',
                    'speed', 'efficiency', 'performance', 'feature', 'update'
                },
                weight=1.0
            ),
            
            NewsTopicCategory.MARKET: TopicKeywordSet(
                strong_indicators={
                    'price', 'trading', 'volume', 'market cap', 'bullish', 'bearish',
                    'rally', 'crash', 'volatility', 'liquidity', 'exchange', 'otc',
                    'spot', 'futures', 'options', 'derivatives', 'leverage',
                    'long', 'short', 'position', 'breakout', 'resistance', 'support'
                },
                moderate_indicators={
                    'buy', 'sell', 'trade', 'invest', 'portfolio', 'asset', 'return',
                    'profit', 'loss', 'gain', 'performance', 'momentum', 'trend',
                    'technical', 'fundamental', 'analysis', 'strategy'
                },
                weak_indicators={
                    'market', 'investor', 'trader', 'fund', 'capital', 'money',
                    'value', 'worth', 'move', 'change', 'up', 'down', 'high', 'low'
                },
                weight=1.1
            ),
            
            NewsTopicCategory.CORPORATE: TopicKeywordSet(
                strong_indicators={
                    'partnership', 'acquisition', 'merger', 'investment', 'funding',
                    'ipo', 'earnings', 'revenue', 'profit', 'quarterly', 'annual',
                    'ceo', 'founder', 'board', 'shareholder', 'dividend', 'buyback',
                    'subsidiary', 'joint venture', 'collaboration'
                },
                moderate_indicators={
                    'company', 'corporation', 'firm', 'enterprise', 'business',
                    'organization', 'institution', 'bank', 'fund', 'venture',
                    'startup', 'unicorn', 'announce', 'report', 'disclose'
                },
                weak_indicators={
                    'team', 'employee', 'staff', 'hire', 'expand', 'grow', 'plan',
                    'strategy', 'goal', 'target', 'achieve', 'success', 'lead'
                },
                weight=1.0
            ),
            
            NewsTopicCategory.SECURITY: TopicKeywordSet(
                strong_indicators={
                    'hack', 'breach', 'vulnerability', 'exploit', 'attack', 'scam',
                    'fraud', 'theft', 'stolen', 'compromised', 'security', 'audit',
                    'bug', 'patch', 'fix', 'malware', 'phishing', 'ransomware',
                    'cybersecurity', 'encryption', 'private key', 'wallet'
                },
                moderate_indicators={
                    'risk', 'threat', 'danger', 'warning', 'alert', 'incident',
                    'investigation', 'forensic', 'protection', 'defense', 'safe',
                    'secure', 'privacy', 'anonymous', 'identity'
                },
                weak_indicators={
                    'issue', 'problem', 'concern', 'report', 'discover', 'find',
                    'prevent', 'avoid', 'protect', 'check', 'verify', 'confirm'
                },
                weight=1.05
            ),
            
            NewsTopicCategory.ADOPTION: TopicKeywordSet(
                strong_indicators={
                    'adoption', 'accept', 'payment', 'integration', 'implement',
                    'launch', 'rollout', 'deploy', 'use case', 'real world',
                    'mainstream', 'mass adoption', 'institutional', 'retail',
                    'merchant', 'vendor', 'supplier', 'customer'
                },
                moderate_indicators={
                    'use', 'utilize', 'adopt', 'embrace', 'support', 'enable',
                    'facilitate', 'process', 'transaction', 'transfer', 'send',
                    'receive', 'store', 'hold', 'custody'
                },
                weak_indicators={
                    'new', 'first', 'begin', 'start', 'initial', 'pilot', 'test',
                    'trial', 'experiment', 'explore', 'consider', 'evaluate'
                },
                weight=1.0
            ),
            
            NewsTopicCategory.ANALYSIS: TopicKeywordSet(
                strong_indicators={
                    'analysis', 'research', 'report', 'study', 'survey', 'forecast',
                    'prediction', 'projection', 'estimate', 'model', 'simulation',
                    'analyst', 'expert', 'opinion', 'perspective', 'insight'
                },
                moderate_indicators={
                    'think', 'believe', 'expect', 'predict', 'anticipate', 'suggest',
                    'indicate', 'show', 'reveal', 'demonstrate', 'conclude',
                    'recommend', 'advise', 'view', 'outlook'
                },
                weak_indicators={
                    'may', 'might', 'could', 'should', 'would', 'likely', 'possible',
                    'probable', 'potential', 'opportunity', 'risk', 'factor'
                },
                weight=0.9  # 분석/의견은 낮은 가중치
            ),
            
            NewsTopicCategory.EVENT: TopicKeywordSet(
                strong_indicators={
                    'conference', 'summit', 'forum', 'symposium', 'workshop',
                    'webinar', 'meetup', 'event', 'announcement', 'presentation',
                    'keynote', 'panel', 'discussion', 'debate', 'expo'
                },
                moderate_indicators={
                    'attend', 'participate', 'speak', 'present', 'host', 'organize',
                    'sponsor', 'exhibit', 'showcase', 'demonstrate', 'network',
                    'gather', 'meet', 'join', 'register'
                },
                weak_indicators={
                    'date', 'time', 'location', 'venue', 'schedule', 'agenda',
                    'program', 'session', 'talk', 'topic', 'speaker', 'guest'
                },
                weight=0.8  # 이벤트는 낮은 가중치
            ),
            
            NewsTopicCategory.OTHER: TopicKeywordSet(
                strong_indicators=set(),
                moderate_indicators=set(),
                weak_indicators=set(),
                weight=0.5
            )
        }
    
    def classify(self, 
                 title: str,
                 content: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> TopicClassification:
        """
        뉴스 토픽 분류
        
        Args:
            title: 뉴스 제목
            content: 뉴스 내용 (선택)
            metadata: 추가 메타데이터 (소스, 날짜 등)
            
        Returns:
            TopicClassification: 분류 결과
        """
        start_time = time.time()
        
        # 텍스트 전처리
        combined_text = self._preprocess_text(title, content)
        
        # 각 토픽별 점수 계산
        topic_scores = self._calculate_topic_scores(combined_text, title)
        
        # 문맥 기반 조정 (선택적)
        if self.enable_context_analysis and content:
            topic_scores = self._adjust_scores_by_context(topic_scores, combined_text)
        
        # 메타데이터 기반 조정 (선택적)
        if metadata:
            topic_scores = self._adjust_scores_by_metadata(topic_scores, metadata)
        
        # 최종 분류 결정
        classification = self._determine_classification(topic_scores, combined_text)
        
        # 디버깅 정보 로깅
        if logger.isEnabledFor(logging.DEBUG):
            sorted_scores = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            logger.debug(f"Topic scoring for '{title[:50]}': {[(t.value, f'{s:.3f}') for t, s in sorted_scores[:3]]}")
        
        # 처리 시간 기록
        classification.processing_time = time.time() - start_time
        
        # 통계 업데이트
        self._update_statistics(classification)
        
        return classification
    
    def _preprocess_text(self, title: str, content: Optional[str]) -> str:
        """텍스트 전처리"""
        # 제목과 내용 결합
        text = title.lower()
        if content:
            text += " " + content.lower()
        
        # 특수문자 정규화
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_topic_scores(self, text: str, title: str) -> Dict[NewsTopicCategory, float]:
        """토픽별 점수 계산"""
        scores = {}
        keywords_found = {}
        
        for topic, keyword_set in self.topic_keywords.items():
            score = 0.0
            found_keywords = {
                'strong': [],
                'moderate': [],
                'weak': []
            }
            
            # 강한 지표 키워드 (3점)
            for keyword in keyword_set.strong_indicators:
                if keyword in text:
                    score += 3.0
                    found_keywords['strong'].append(keyword)
                    # 제목에 있으면 추가 점수
                    if keyword in title.lower():
                        score += 1.0
            
            # 중간 지표 키워드 (2점)
            for keyword in keyword_set.moderate_indicators:
                if keyword in text:
                    score += 2.0
                    found_keywords['moderate'].append(keyword)
                    if keyword in title.lower():
                        score += 0.5
            
            # 약한 지표 키워드 (1점)
            for keyword in keyword_set.weak_indicators:
                if keyword in text:
                    score += 1.0
                    found_keywords['weak'].append(keyword)
            
            # 가중치 적용
            score *= keyword_set.weight
            
            # 정규화 (0-1 범위)
            # 최대 예상 점수를 기준으로 정규화
            max_expected_score = 30.0  # 조정 가능
            normalized_score = min(score / max_expected_score, 1.0)
            
            scores[topic] = normalized_score
            keywords_found[topic] = found_keywords
        
        return scores
    
    def _adjust_scores_by_context(self, 
                                  scores: Dict[NewsTopicCategory, float],
                                  text: str) -> Dict[NewsTopicCategory, float]:
        """문맥 기반 점수 조정"""
        adjusted_scores = scores.copy()
        
        # 예시: 특정 패턴이 발견되면 관련 토픽 점수 증가
        patterns = {
            r'breaking\s+news': {NewsTopicCategory.EVENT: 0.1, NewsTopicCategory.MARKET: 0.1},
            r'according\s+to\s+\w+\s+research': {NewsTopicCategory.ANALYSIS: 0.2},
            r'announced\s+\w+\s+partnership': {NewsTopicCategory.CORPORATE: 0.2},
            r'vulnerability\s+discovered': {NewsTopicCategory.SECURITY: 0.3},
            r'federal\s+reserve\s+\w+\s+rates': {NewsTopicCategory.MACRO: 0.3}
        }
        
        for pattern, adjustments in patterns.items():
            if re.search(pattern, text):
                for topic, adjustment in adjustments.items():
                    adjusted_scores[topic] = min(adjusted_scores[topic] + adjustment, 1.0)
        
        return adjusted_scores
    
    def _adjust_scores_by_metadata(self,
                                   scores: Dict[NewsTopicCategory, float],
                                   metadata: Dict[str, Any]) -> Dict[NewsTopicCategory, float]:
        """메타데이터 기반 점수 조정"""
        adjusted_scores = scores.copy()
        
        # 소스 기반 조정
        source = metadata.get('source', '').lower()
        source_adjustments = {
            'reuters': {NewsTopicCategory.MACRO: 0.1, NewsTopicCategory.CORPORATE: 0.1},
            'bloomberg': {NewsTopicCategory.MARKET: 0.1, NewsTopicCategory.MACRO: 0.1},
            'coindesk': {NewsTopicCategory.TECHNOLOGY: 0.1, NewsTopicCategory.MARKET: 0.1},
            'sec.gov': {NewsTopicCategory.REGULATION: 0.3},
            'techcrunch': {NewsTopicCategory.TECHNOLOGY: 0.2, NewsTopicCategory.CORPORATE: 0.1}
        }
        
        for source_key, adjustments in source_adjustments.items():
            if source_key in source:
                for topic, adjustment in adjustments.items():
                    adjusted_scores[topic] = min(adjusted_scores[topic] + adjustment, 1.0)
        
        return adjusted_scores
    
    def _determine_classification(self,
                                  scores: Dict[NewsTopicCategory, float],
                                  text: str) -> TopicClassification:
        """최종 분류 결정"""
        # 점수 기준 정렬
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 주요 토픽 결정
        primary_topic = sorted_topics[0][0] if sorted_topics[0][1] >= self.confidence_threshold else NewsTopicCategory.OTHER
        
        # 부가 토픽 결정
        secondary_topics = []
        for topic, score in sorted_topics[1:]:
            if score >= self.multi_topic_threshold and score > 0:
                secondary_topics.append(topic)
                if len(secondary_topics) >= 3:  # 최대 3개 부가 토픽
                    break
        
        # 키워드 수집
        keywords_found = {}
        for topic in [primary_topic] + secondary_topics:
            if topic != NewsTopicCategory.OTHER:
                keywords_found[topic.value] = self._extract_found_keywords(text, topic)
        
        return TopicClassification(
            primary_topic=primary_topic,
            secondary_topics=secondary_topics,
            confidence_scores=scores,
            keywords_found=keywords_found
        )
    
    def _extract_found_keywords(self, text: str, topic: NewsTopicCategory) -> List[str]:
        """텍스트에서 발견된 토픽 키워드 추출"""
        found = []
        keyword_set = self.topic_keywords[topic]
        
        for keyword in keyword_set.strong_indicators.union(
            keyword_set.moderate_indicators, 
            keyword_set.weak_indicators
        ):
            if keyword in text:
                found.append(keyword)
        
        return found[:10]  # 최대 10개만 반환
    
    def _update_statistics(self, classification: TopicClassification):
        """통계 업데이트"""
        self.classification_stats["total_classified"] += 1
        self.classification_stats["topic_distribution"][classification.primary_topic] += 1
        
        # 평균 신뢰도 업데이트 (이동 평균)
        n = self.classification_stats["total_classified"]
        prev_avg = self.classification_stats["avg_confidence"]
        new_confidence = classification.confidence_scores.get(classification.primary_topic, 0)
        self.classification_stats["avg_confidence"] = (prev_avg * (n-1) + new_confidence) / n
        
        # 다중 토픽 비율 업데이트
        if classification.is_multi_topic:
            multi_count = sum(1 for t, c in self.classification_stats["topic_distribution"].items() 
                            if c > 0 and t in classification.secondary_topics)
            self.classification_stats["multi_topic_ratio"] = multi_count / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """분류 통계 반환"""
        return {
            **self.classification_stats,
            "topic_percentages": {
                topic.value: (count / self.classification_stats["total_classified"] * 100 
                            if self.classification_stats["total_classified"] > 0 else 0)
                for topic, count in self.classification_stats["topic_distribution"].items()
            }
        }
    
    def classify_batch(self, news_items: List[Dict[str, Any]]) -> List[TopicClassification]:
        """배치 분류"""
        results = []
        
        for item in news_items:
            classification = self.classify(
                title=item.get('title', ''),
                content=item.get('content'),
                metadata=item.get('metadata', {})
            )
            results.append(classification)
        
        return results

# 편의 함수
def classify_news_topic(title: str, 
                       content: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> TopicClassification:
    """뉴스 토픽 분류 편의 함수"""
    classifier = NewsTopicClassifier()
    return classifier.classify(title, content, metadata)

if __name__ == "__main__":
    # 테스트
    classifier = NewsTopicClassifier()
    
    test_cases = [
        {
            "title": "Federal Reserve Raises Interest Rates by 0.75% to Combat Inflation",
            "content": "The Federal Open Market Committee announced a 75 basis point rate hike..."
        },
        {
            "title": "SEC Approves First Bitcoin ETF After Years of Delays",
            "content": "The Securities and Exchange Commission has finally approved..."
        },
        {
            "title": "Ethereum Successfully Completes Shanghai Upgrade",
            "content": "The Ethereum network has successfully implemented the Shanghai upgrade..."
        },
        {
            "title": "Major Exchange Hacked, $100M in Crypto Stolen",
            "content": "A major cryptocurrency exchange reported a security breach..."
        }
    ]
    
    print("=== News Topic Classification Test ===\n")
    
    for i, test in enumerate(test_cases, 1):
        result = classifier.classify(test["title"], test["content"])
        
        print(f"{i}. {test['title']}")
        print(f"   Primary Topic: {result.primary_topic.value} (confidence: {result.confidence_scores[result.primary_topic]:.2f})")
        
        if result.secondary_topics:
            print(f"   Secondary Topics: {[t.value for t in result.secondary_topics]}")
        
        if result.keywords_found:
            for topic, keywords in result.keywords_found.items():
                print(f"   Keywords ({topic}): {keywords[:5]}")
        
        print()
    
    # 통계 출력
    stats = classifier.get_statistics()
    print("\n=== Classification Statistics ===")
    print(f"Total Classified: {stats['total_classified']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Multi-topic Ratio: {stats['multi_topic_ratio']:.1%}")
    print("\nTopic Distribution:")
    for topic, percentage in stats['topic_percentages'].items():
        if percentage > 0:
            print(f"  {topic}: {percentage:.1f}%")