# sentiment_event_manager.py - 개선된 버전

import datetime
import logging
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json
from pathlib import Path
import threading
import statistics

# 외부 의존성 (가정)
from sentiment.sentiment_score import get_sentiment_score

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """이벤트 우선순위"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SentimentTrend(Enum):
    """감정 트렌드"""
    STRONG_BEARISH = "strong_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    STRONG_BULLISH = "strong_bullish"

@dataclass
class Event:
    """이벤트 데이터 모델"""
    name: str
    time: datetime.datetime
    priority: EventPriority = EventPriority.MEDIUM
    expected_impact: Optional[float] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class SentimentSnapshot:
    """감정 스냅샷"""
    timestamp: datetime.datetime
    score: float
    sources: Dict[str, float]
    trend: SentimentTrend
    volatility: float = 0.0

class SentimentEventManager:
    """개선된 감정 이벤트 관리자"""
    
    def __init__(self, for_backtest: bool = False,
                 history_size: int = 100,
                 staleness_threshold: int = 1200):
        """
        Args:
            for_backtest: 백테스트 모드 여부
            history_size: 히스토리 보관 크기
            staleness_threshold: 데이터 오래됨 임계값 (초)
        """
        self.for_backtest = for_backtest
        self.history_size = history_size
        self.staleness_threshold = staleness_threshold
        
        # 상태 관리
        self.sentiment_sources: Dict[str, float] = {}
        self.last_update_time: Optional[datetime.datetime] = None
        self.cached_score: Optional[float] = None
        self.previous_score: Optional[float] = None
        
        # 히스토리 관리 (deque for O(1) operations)
        self.history: deque[SentimentSnapshot] = deque(maxlen=history_size)
        
        # 이벤트 관리
        self.event_schedule: List[Event] = []
        self.past_events: List[Event] = []
        
        # 통계 추적
        self.stats = {
            'updates': 0,
            'stale_warnings': 0,
            'trend_changes': 0
        }
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 설정 로드
        self._load_config()
        
        logger.info(f"SentimentEventManager initialized (backtest={for_backtest})")
    
    def _load_config(self):
        """설정 로드"""
        # 트렌드 임계값
        self.trend_thresholds = {
            'strong_bearish': -0.6,
            'bearish': -0.2,
            'neutral': 0.2,
            'bullish': 0.6,
            'strong_bullish': 1.0
        }
        
        # 가중치 설정
        self.source_weights = {
            'news': 0.4,
            'social': 0.3,
            'technical': 0.2,
            'historical': 0.1
        }
    
    def update_sentiment(self, source_scores: Dict[str, float]):
        """
        감정 점수 업데이트 (개선된 버전)
        
        Args:
            source_scores: 소스별 감정 점수
        """
        with self._lock:
            if not source_scores:
                logger.warning("Empty source scores provided")
                return
            
            # 소스 검증
            validated_scores = self._validate_scores(source_scores)
            if not validated_scores:
                return
            
            # 이전 점수 저장
            if self.cached_score is not None:
                self.previous_score = self.cached_score
            
            # 가중 평균 계산
            self.cached_score = self._calculate_weighted_average(validated_scores)
            
            # 상태 업데이트
            self.sentiment_sources = validated_scores
            self.last_update_time = datetime.datetime.utcnow()
            
            # 트렌드 계산
            trend = self._calculate_trend()
            
            # 변동성 계산
            volatility = self._calculate_volatility()
            
            # 스냅샷 저장
            snapshot = SentimentSnapshot(
                timestamp=self.last_update_time,
                score=self.cached_score,
                sources=validated_scores.copy(),
                trend=trend,
                volatility=volatility
            )
            self.history.append(snapshot)
            
            # 통계 업데이트
            self.stats['updates'] += 1
            
            # 트렌드 변화 감지
            if len(self.history) > 1 and self.history[-2].trend != trend:
                self.stats['trend_changes'] += 1
                logger.info(f"Trend changed: {self.history[-2].trend.value} → {trend.value}")
            
            logger.debug(f"Sentiment updated: {self.cached_score:.4f} ({trend.value})")
    
    def _validate_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """점수 유효성 검증"""
        validated = {}
        
        for source, score in scores.items():
            if not isinstance(score, (int, float)):
                logger.warning(f"Invalid score type for {source}: {type(score)}")
                continue
                
            # 범위 검증 (-1 ~ 1)
            if not -1 <= score <= 1:
                logger.warning(f"Score out of range for {source}: {score}")
                score = max(-1, min(1, score))
            
            validated[source] = float(score)
        
        return validated
    
    def _calculate_weighted_average(self, scores: Dict[str, float]) -> float:
        """가중 평균 계산"""
        total_weight = 0
        weighted_sum = 0
        
        for source, score in scores.items():
            weight = self.source_weights.get(source, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        return weighted_sum / total_weight
    
    def _calculate_trend(self) -> SentimentTrend:
        """현재 트렌드 계산"""
        if self.cached_score is None:
            return SentimentTrend.NEUTRAL
        
        score = self.cached_score
        
        if score <= self.trend_thresholds['strong_bearish']:
            return SentimentTrend.STRONG_BEARISH
        elif score <= self.trend_thresholds['bearish']:
            return SentimentTrend.BEARISH
        elif score <= self.trend_thresholds['neutral']:
            return SentimentTrend.NEUTRAL
        elif score <= self.trend_thresholds['bullish']:
            return SentimentTrend.BULLISH
        else:
            return SentimentTrend.STRONG_BULLISH
    
    def _calculate_volatility(self, window: int = 20) -> float:
        """변동성 계산"""
        if len(self.history) < 2:
            return 0.0
        
        # 최근 window 개의 점수
        recent_scores = [
            snapshot.score 
            for snapshot in list(self.history)[-window:]
        ]
        
        if len(recent_scores) < 2:
            return 0.0
        
        # 표준편차 계산
        try:
            return statistics.stdev(recent_scores)
        except:
            return 0.0
    
    def update_sentiment_from_news(self, news_text: str):
        """뉴스 텍스트로부터 감정 업데이트"""
        if not news_text or not news_text.strip():
            logger.warning("Empty news text provided")
            return
        
        try:
            score = get_sentiment_score(news_text)
            self.update_sentiment({"news": score})
            
        except Exception as e:
            logger.error(f"Failed to analyze news sentiment: {e}")
    
    def get_sentiment_score(self) -> Optional[float]:
        """현재 감정 점수 반환"""
        with self._lock:
            if self.last_update_time is None:
                logger.warning("No sentiment data available")
                return None
            
            # 오래된 데이터 경고
            age = (datetime.datetime.utcnow() - self.last_update_time).total_seconds()
            if age > self.staleness_threshold:
                self.stats['stale_warnings'] += 1
                logger.warning(f"Sentiment data is {age:.0f} seconds old (threshold: {self.staleness_threshold}s)")
            
            return self.cached_score
    
    def get_previous_sentiment(self) -> Optional[float]:
        """이전 감정 점수 반환"""
        with self._lock:
            return self.previous_score
    
    def get_sentiment_delta(self) -> float:
        """감정 점수 변화량 반환"""
        with self._lock:
            if self.previous_score is None or self.cached_score is None:
                return 0.0
            return self.cached_score - self.previous_score
    
    def get_sentiment_trend(self) -> SentimentTrend:
        """현재 감정 트렌드 반환"""
        with self._lock:
            if not self.history:
                return SentimentTrend.NEUTRAL
            return self.history[-1].trend
    
    def get_volatility(self, window: int = 20) -> float:
        """현재 변동성 반환"""
        with self._lock:
            if not self.history:
                return 0.0
            return self.history[-1].volatility
    
    def get_moving_average(self, window: int = 10) -> Optional[float]:
        """이동 평균 계산"""
        with self._lock:
            if len(self.history) < window:
                return None
            
            recent_scores = [
                snapshot.score 
                for snapshot in list(self.history)[-window:]
            ]
            
            return sum(recent_scores) / len(recent_scores)
    
    def register_event(self, event_name: str, event_time: Union[str, datetime.datetime],
                      priority: Union[str, EventPriority] = EventPriority.MEDIUM,
                      expected_impact: Optional[float] = None,
                      description: str = "",
                      tags: Optional[List[str]] = None):
        """
        이벤트 등록 (개선된 버전)
        
        Args:
            event_name: 이벤트 이름
            event_time: 이벤트 시간 (문자열 또는 datetime)
            priority: 우선순위
            expected_impact: 예상 영향도 (-1 ~ 1)
            description: 이벤트 설명
            tags: 태그 리스트
        """
        with self._lock:
            try:
                # 시간 파싱
                if isinstance(event_time, str):
                    dt = datetime.datetime.strptime(event_time, "%Y-%m-%d %H:%M")
                else:
                    dt = event_time
                
                # 우선순위 파싱
                if isinstance(priority, str):
                    priority = EventPriority[priority.upper()]
                
                # 이벤트 생성
                event = Event(
                    name=event_name,
                    time=dt,
                    priority=priority,
                    expected_impact=expected_impact,
                    description=description,
                    tags=tags or []
                )
                
                # 과거/미래 이벤트 분류
                if dt < datetime.datetime.utcnow():
                    self.past_events.append(event)
                else:
                    self.event_schedule.append(event)
                    # 시간순 정렬
                    self.event_schedule.sort(key=lambda e: e.time)
                
                logger.info(f"Event registered: {event_name} at {dt}")
                
            except ValueError as e:
                logger.error(f"Failed to register event: {e}")
    
    def get_upcoming_events(self, within_minutes: int = 60) -> List[Event]:
        """다가오는 이벤트 조회"""
        with self._lock:
            now = datetime.datetime.utcnow()
            horizon = now + datetime.timedelta(minutes=within_minutes)
            
            upcoming = []
            for event in self.event_schedule:
                if now <= event.time <= horizon:
                    upcoming.append(event)
                elif event.time > horizon:
                    break  # 정렬되어 있으므로 중단
            
            return upcoming
    
    def get_recent_events(self, past_minutes: int = 60) -> List[Event]:
        """최근 발생한 이벤트 조회"""
        with self._lock:
            now = datetime.datetime.utcnow()
            cutoff = now - datetime.timedelta(minutes=past_minutes)
            
            recent = []
            for event in reversed(self.past_events):
                if event.time >= cutoff:
                    recent.append(event)
                else:
                    break
            
            return list(reversed(recent))
    
    def get_event_impact_analysis(self, lookback_minutes: int = 30) -> Dict:
        """이벤트 영향 분석"""
        with self._lock:
            recent_events = self.get_recent_events(lookback_minutes)
            
            if not recent_events:
                return {"events": [], "impact": None}
            
            # 이벤트 전후 감정 변화 분석
            analysis = {
                "events": [],
                "total_impact": 0.0,
                "average_impact": 0.0
            }
            
            for event in recent_events:
                # 이벤트 전후 스냅샷 찾기
                before = None
                after = None
                
                for snapshot in self.history:
                    if snapshot.timestamp < event.time:
                        before = snapshot
                    elif snapshot.timestamp >= event.time and after is None:
                        after = snapshot
                        break
                
                if before and after:
                    impact = after.score - before.score
                    analysis["events"].append({
                        "name": event.name,
                        "time": event.time.isoformat(),
                        "priority": event.priority.value,
                        "impact": impact,
                        "expected_impact": event.expected_impact
                    })
                    analysis["total_impact"] += impact
            
            if analysis["events"]:
                analysis["average_impact"] = (
                    analysis["total_impact"] / len(analysis["events"])
                )
            
            return analysis
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        with self._lock:
            stats = self.stats.copy()
            
            # 추가 통계
            if self.history:
                scores = [s.score for s in self.history]
                stats.update({
                    'history_size': len(self.history),
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'avg_score': sum(scores) / len(scores),
                    'current_trend': self.get_sentiment_trend().value
                })
            
            return stats
    
    def save_state(self, filepath: Union[str, Path]):
        """상태 저장"""
        with self._lock:
            state = {
                'sentiment_sources': self.sentiment_sources,
                'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
                'cached_score': self.cached_score,
                'previous_score': self.previous_score,
                'history': [
                    {
                        'timestamp': s.timestamp.isoformat(),
                        'score': s.score,
                        'sources': s.sources,
                        'trend': s.trend.value,
                        'volatility': s.volatility
                    }
                    for s in self.history
                ],
                'event_schedule': [
                    {
                        'name': e.name,
                        'time': e.time.isoformat(),
                        'priority': e.priority.value,
                        'expected_impact': e.expected_impact,
                        'description': e.description,
                        'tags': e.tags
                    }
                    for e in self.event_schedule
                ],
                'stats': self.stats
            }
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: Union[str, Path]):
        """상태 로드"""
        with self._lock:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.warning(f"State file not found: {filepath}")
                return
            
            try:
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                # 상태 복원
                self.sentiment_sources = state.get('sentiment_sources', {})
                
                if state.get('last_update_time'):
                    self.last_update_time = datetime.datetime.fromisoformat(
                        state['last_update_time']
                    )
                
                self.cached_score = state.get('cached_score')
                self.previous_score = state.get('previous_score')
                self.stats = state.get('stats', {})
                
                # 히스토리 복원
                self.history.clear()
                for h in state.get('history', []):
                    snapshot = SentimentSnapshot(
                        timestamp=datetime.datetime.fromisoformat(h['timestamp']),
                        score=h['score'],
                        sources=h['sources'],
                        trend=SentimentTrend(h['trend']),
                        volatility=h.get('volatility', 0.0)
                    )
                    self.history.append(snapshot)
                
                # 이벤트 복원
                self.event_schedule.clear()
                for e in state.get('event_schedule', []):
                    event = Event(
                        name=e['name'],
                        time=datetime.datetime.fromisoformat(e['time']),
                        priority=EventPriority(e['priority']),
                        expected_impact=e.get('expected_impact'),
                        description=e.get('description', ''),
                        tags=e.get('tags', [])
                    )
                    self.event_schedule.append(event)
                
                logger.info(f"State loaded from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load state: {e}")


# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 매니저 생성
    manager = SentimentEventManager(for_backtest=False)
    
    # 테스트 데이터
    print("=== 감정 업데이트 테스트 ===")
    
    # 뉴스 기반 업데이트
    manager.update_sentiment_from_news(
        "Bitcoin plunges as regulatory uncertainty increases."
    )
    print(f"감정 점수: {manager.get_sentiment_score():.4f}")
    print(f"트렌드: {manager.get_sentiment_trend().value}")
    
    # 다중 소스 업데이트
    manager.update_sentiment({
        "news": -0.3,
        "social": -0.5,
        "technical": 0.1
    })
    print(f"업데이트 후 점수: {manager.get_sentiment_score():.4f}")
    print(f"점수 변화량: {manager.get_sentiment_delta():.4f}")
    print(f"변동성: {manager.get_volatility():.4f}")
    
    # 이벤트 등록
    print("\n=== 이벤트 관리 테스트 ===")
    
    # 미래 이벤트
    manager.register_event(
        "FOMC Meeting",
        (datetime.datetime.utcnow() + datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
        priority="HIGH",
        expected_impact=-0.2,
        description="Federal Reserve interest rate decision",
        tags=["macro", "fed", "rates"]
    )
    
    # 과거 이벤트
    manager.register_event(
        "CPI Release",
        (datetime.datetime.utcnow() - datetime.timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"),
        priority="MEDIUM",
        expected_impact=0.1,
        description="Consumer Price Index data release"
    )
    
    # 이벤트 조회
    upcoming = manager.get_upcoming_events(within_minutes=180)
    print(f"다가오는 이벤트: {len(upcoming)}개")
    for event in upcoming:
        print(f"  - {event.name} at {event.time} (Priority: {event.priority.name})")
    
    # 이벤트 영향 분석
    impact_analysis = manager.get_event_impact_analysis(lookback_minutes=60)
    print(f"\n이벤트 영향 분석: {impact_analysis}")
    
    # 통계
    print(f"\n통계: {manager.get_statistics()}")
    
    # 상태 저장/로드 테스트
    print("\n=== 상태 저장/로드 테스트 ===")
    manager.save_state("test_state.json")
    
    # 새 매니저로 로드
    new_manager = SentimentEventManager()
    new_manager.load_state("test_state.json")
    print(f"로드된 점수: {new_manager.get_sentiment_score():.4f}")