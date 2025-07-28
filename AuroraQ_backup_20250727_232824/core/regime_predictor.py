import datetime
import json
import os
import logging
from statistics import mean
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.path_config import get_log_path

logger = logging.getLogger(__name__)

class RegimePredictor:
    """시장 레짐 예측 클래스"""
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Args:
            log_path: 레짐 상태 저장 경로 (None이면 기본 경로 사용)
        """
        self.regime_data: List[str] = []
        self.long_term_trends: List[str] = []
        self.last_updated: Optional[datetime.datetime] = None
        
        if log_path:
            self.log_path = Path(log_path)
        else:
            self.log_path = get_log_path("regime_status")
        
        # 로그 디렉토리 생성
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"RegimePredictor initialized with log path: {self.log_path}")

    def update_regime_data(self, price_series: List[float]) -> str:
        """
        가격 데이터를 기반으로 레짐 업데이트
        
        Args:
            price_series: 가격 시계열 데이터
            
        Returns:
            str: 현재 레짐 ('volatile', 'bullish', 'bearish', 'neutral')
        """
        try:
            if not price_series or len(price_series) < 10:
                logger.warning("Insufficient price data for regime analysis")
                self.regime_data.append("neutral")
                return "neutral"

            # 최근 변동성 계산
            recent_prices = price_series[-10:]
            recent_volatility = max(recent_prices) - min(recent_prices)
            price_range = max(recent_prices) if max(recent_prices) > 0 else 1
            volatility_ratio = recent_volatility / price_range
            
            # 레짐 판단 (개선된 로직)
            if volatility_ratio > 0.05:
                regime = "volatile"
                logger.debug(f"High volatility detected: {volatility_ratio:.3f}")
            elif price_series[-1] > price_series[-5] * 1.01:  # 1% 이상 상승
                regime = "bullish"
            elif price_series[-1] < price_series[-5] * 0.99:  # 1% 이상 하락
                regime = "bearish"
            else:
                regime = "neutral"

            self.regime_data.append(regime)
            self.last_updated = datetime.datetime.utcnow()

            # 장기 추세 분석
            if len(price_series) >= 30:
                try:
                    short_ma = mean(price_series[-10:])
                    long_ma = mean(price_series[-30:])
                    
                    if short_ma > long_ma * 1.02:  # 2% 이상 위
                        trend = "uptrend"
                    elif short_ma < long_ma * 0.98:  # 2% 이상 아래
                        trend = "downtrend"
                    else:
                        trend = "sideways"
                    
                    self.long_term_trends.append(trend)
                    logger.debug(f"Long-term trend: {trend}")
                    
                except Exception as e:
                    logger.error(f"Error calculating long-term trend: {e}")

            return regime
            
        except Exception as e:
            logger.error(f"Error updating regime data: {e}", exc_info=True)
            return "neutral"

    def predict_regime(self) -> str:
        """현재 레짐 예측"""
        return self.regime_data[-1] if self.regime_data else "neutral"

    def get_long_term_trend(self) -> str:
        """장기 추세 조회"""
        return self.long_term_trends[-1] if self.long_term_trends else "sideways"
    
    def get_regime_history(self, limit: int = 10) -> List[str]:
        """최근 레짐 히스토리 조회"""
        return self.regime_data[-limit:] if self.regime_data else []

    def save_regime_status(self, regime: str, additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        레짐 상태를 파일에 저장
        
        Args:
            regime: 현재 레짐
            additional_data: 추가 저장할 데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            data = {
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "regime": regime,
                "long_term_trend": self.get_long_term_trend()
            }
            
            # 추가 데이터 병합
            if additional_data:
                data.update(additional_data)
            
            # 임시 파일에 먼저 쓰고 원자적으로 교체
            temp_file = self.log_path.with_suffix('.tmp')
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.log_path)
            logger.debug(f"Regime status saved: {regime}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save regime status: {e}", exc_info=True)
            return False

    def predict(self, price_data: Dict[str, List[float]], sentiment_score: Optional[float] = None) -> str:
        """
        가격과 감정 데이터를 기반으로 시장 레짐 예측
        
        Args:
            price_data: OHLCV 데이터
            sentiment_score: 감정 점수 (0~1)
            
        Returns:
            str: 예측된 레짐
        """
        try:
            prices = price_data.get("close", [])
            if not prices:
                logger.warning("No price data available for regime prediction")
                self.save_regime_status("neutral")
                return "neutral"

            # 기본 레짐 분석
            recent_regime = self.update_regime_data(prices)
            final_regime = recent_regime

            # 감정 점수를 이용한 레짐 세분화
            if sentiment_score is not None:
                logger.debug(f"Adjusting regime with sentiment score: {sentiment_score:.3f}")
                
                if sentiment_score > 0.7 and recent_regime == "bullish":
                    final_regime = "strong_bull"
                    logger.info("Strong bullish regime detected")
                elif sentiment_score > 0.6 and recent_regime == "neutral":
                    final_regime = "bullish"  # 중립에서 상승으로 전환
                elif sentiment_score < 0.3 and recent_regime == "bearish":
                    final_regime = "panic_bear"
                    logger.info("Panic bearish regime detected")
                elif sentiment_score < 0.4 and recent_regime == "neutral":
                    final_regime = "bearish"  # 중립에서 하락으로 전환

            # 상태 저장
            self.save_regime_status(final_regime, {
                "base_regime": recent_regime,
                "sentiment_score": sentiment_score,
                "volatility": self._calculate_volatility(prices) if len(prices) > 20 else None
            })
            
            return final_regime

        except Exception as e:
            logger.error(f"Regime prediction failed: {e}", exc_info=True)
            self.save_regime_status("neutral", {"error": str(e)})
            return "neutral"
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """변동성 계산 (표준편차 기반)"""
        try:
            if len(prices) < 2:
                return 0.0
            
            returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                      for i in range(1, len(prices)) 
                      if prices[i-1] != 0]
            
            if not returns:
                return 0.0
                
            avg_return = mean(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            return variance ** 0.5
            
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 0.0
