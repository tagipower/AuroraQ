#!/usr/bin/env python3
"""
Notification Integration Module
알림 시스템과 다른 컴포넌트들을 연결하는 통합 모듈
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..notifications.notification_manager import NotificationManager, NotificationLevel
from ..integrations.aurora_adapter import TradingSignal, TradingMode
from ..processors.big_event_detector import BigEvent

logger = logging.getLogger(__name__)

class NotificationIntegration:
    """알림 시스템 통합 클래스"""
    
    def __init__(self, notification_manager: Optional[NotificationManager] = None):
        """
        초기화
        
        Args:
            notification_manager: 알림 관리자 인스턴스
        """
        self.notification_manager = notification_manager
        self.enabled = notification_manager is not None
        
        # 통계
        self.stats = {
            "notifications_sent": 0,
            "trading_signals": 0,
            "big_events": 0,
            "system_alerts": 0,
            "errors": 0
        }

    async def send_trading_signal_notification(self,
                                             signal: TradingSignal,
                                             trading_mode: TradingMode = TradingMode.LIVE):
        """매매 신호 알림 발송"""
        if not self.enabled:
            return
        
        try:
            # TradingSignal을 dict로 변환
            signal_dict = {
                "symbol": signal.symbol,
                "direction": signal.signal_type,
                "strength": signal.strength.value,
                "confidence": signal.confidence,
                "sentiment_score": signal.sentiment_score,
                "timestamp": signal.timestamp.isoformat() if hasattr(signal, 'timestamp') else datetime.now().isoformat()
            }
            
            await self.notification_manager.send_trading_signal_notification(
                signal_dict, trading_mode.value.upper()
            )
            
            self.stats["notifications_sent"] += 1
            self.stats["trading_signals"] += 1
            
            logger.info("Trading signal notification sent",
                       symbol=signal.symbol,
                       direction=signal.signal_type,
                       mode=trading_mode.value)
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send trading signal notification", error=str(e))

    async def send_big_event_notification(self, event: BigEvent):
        """빅 이벤트 알림 발송"""
        if not self.enabled:
            return
        
        try:
            # BigEvent를 dict로 변환
            event_dict = {
                "event_type": event.event_type.value,
                "symbol": event.symbol,
                "impact_score": event.impact_score,
                "confidence": event.confidence,
                "description": event.description,
                "timestamp": event.timestamp.isoformat()
            }
            
            await self.notification_manager.send_big_event_notification(event_dict)
            
            self.stats["notifications_sent"] += 1
            self.stats["big_events"] += 1
            
            logger.info("Big event notification sent",
                       event_type=event.event_type.value,
                       symbol=event.symbol)
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send big event notification", error=str(e))

    async def send_system_alert(self,
                              message: str,
                              level: str = "info",
                              service: str = "sentiment-service"):
        """시스템 알림 발송"""
        if not self.enabled:
            return
        
        try:
            # 문자열을 NotificationLevel로 변환
            level_map = {
                "info": NotificationLevel.INFO,
                "warning": NotificationLevel.WARNING,
                "error": NotificationLevel.ERROR,
                "critical": NotificationLevel.CRITICAL
            }
            
            notification_level = level_map.get(level.lower(), NotificationLevel.INFO)
            
            await self.notification_manager.send_system_notification(
                message, notification_level, service
            )
            
            self.stats["notifications_sent"] += 1
            self.stats["system_alerts"] += 1
            
            logger.info("System alert notification sent",
                       level=level,
                       service=service)
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send system alert notification", error=str(e))

    async def send_performance_report(self, stats: Dict[str, Any]):
        """성능 보고서 알림 발송"""
        if not self.enabled:
            return
        
        try:
            await self.notification_manager.send_performance_notification(stats)
            
            self.stats["notifications_sent"] += 1
            
            logger.info("Performance report notification sent")
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send performance report notification", error=str(e))

    async def send_health_check_alert(self,
                                    component: str,
                                    status: str,
                                    details: Optional[Dict[str, Any]] = None):
        """헬스체크 알림 발송"""
        if not self.enabled:
            return
        
        if status == "healthy":
            return  # 정상 상태는 알림 안함
        
        try:
            await self.notification_manager.send_health_alert(component, status, details)
            
            self.stats["notifications_sent"] += 1
            self.stats["system_alerts"] += 1
            
            logger.info("Health check alert sent",
                       component=component,
                       status=status)
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send health check alert", error=str(e))

    async def send_startup_notification(self):
        """서비스 시작 알림 발송"""
        if not self.enabled:
            return
        
        try:
            await self.notification_manager.send_startup_notification()
            
            self.stats["notifications_sent"] += 1
            self.stats["system_alerts"] += 1
            
            logger.info("Startup notification sent")
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send startup notification", error=str(e))

    async def send_batch_completion_notification(self,
                                               task_name: str,
                                               success: bool,
                                               stats: Optional[Dict[str, Any]] = None):
        """배치 작업 완료 알림"""
        if not self.enabled:
            return
        
        try:
            if success:
                message = f"배치 작업 완료: {task_name}"
                if stats:
                    message += f"\n처리 결과: {json.dumps(stats, ensure_ascii=False)}"
                level = "info"
            else:
                message = f"배치 작업 실패: {task_name}"
                level = "error"
            
            await self.send_system_alert(message, level)
            
            logger.info("Batch completion notification sent",
                       task=task_name,
                       success=success)
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send batch completion notification", error=str(e))

    async def send_api_error_notification(self,
                                        endpoint: str,
                                        error_message: str,
                                        status_code: Optional[int] = None):
        """API 에러 알림"""
        if not self.enabled:
            return
        
        try:
            message = f"API 에러 발생\n엔드포인트: {endpoint}\n에러: {error_message}"
            if status_code:
                message += f"\n상태 코드: {status_code}"
            
            level = "critical" if status_code and status_code >= 500 else "error"
            
            await self.send_system_alert(message, level)
            
            logger.info("API error notification sent",
                       endpoint=endpoint,
                       status_code=status_code)
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Failed to send API error notification", error=str(e))

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            **self.stats,
            "enabled": self.enabled,
            "notification_manager_stats": (
                self.notification_manager.get_stats() 
                if self.notification_manager else None
            )
        }

    def is_enabled(self) -> bool:
        """알림 활성화 여부 확인"""
        return self.enabled


# 전역 인스턴스 (싱글톤 패턴)
_notification_integration: Optional[NotificationIntegration] = None

def get_notification_integration(notification_manager: Optional[NotificationManager] = None) -> NotificationIntegration:
    """알림 통합 인스턴스 가져오기"""
    global _notification_integration
    
    if _notification_integration is None:
        _notification_integration = NotificationIntegration(notification_manager)
    
    return _notification_integration

def set_notification_integration(notification_manager: Optional[NotificationManager]):
    """알림 통합 인스턴스 설정"""
    global _notification_integration
    _notification_integration = NotificationIntegration(notification_manager)