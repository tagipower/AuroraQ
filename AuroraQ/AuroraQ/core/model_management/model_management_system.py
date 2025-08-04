#!/usr/bin/env python3
"""
통합 모델 관리 시스템
P4: 모델 품질 모니터링 및 Fine-tuning 시스템 구축 - 통합 관리자
"""

import sys
import os
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import threading
import warnings

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

class ManagementAction(Enum):
    """관리 액션 타입"""
    MONITOR_ONLY = "monitor_only"
    AUTO_TUNE = "auto_tune"
    BACKUP_MODEL = "backup_model"
    ALERT_ADMIN = "alert_admin"
    EMERGENCY_ROLLBACK = "emergency_rollback"

class AutoTuningPolicy(Enum):
    """자동 튜닝 정책"""
    CONSERVATIVE = "conservative"  # 품질이 많이 떨어져야 튜닝
    MODERATE = "moderate"         # 적당한 품질 저하에 튜닝
    AGGRESSIVE = "aggressive"     # 작은 품질 저하에도 튜닝
    DISABLED = "disabled"         # 자동 튜닝 비활성화

@dataclass
class ManagementPolicy:
    """모델 관리 정책"""
    model_type: str
    auto_tuning: AutoTuningPolicy = AutoTuningPolicy.MODERATE
    quality_alert_threshold: float = 0.7
    emergency_rollback_threshold: float = 0.5
    max_auto_tuning_per_day: int = 2
    backup_interval_hours: int = 24
    monitoring_interval_minutes: int = 30
    enable_notifications: bool = True
    custom_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ManagementEvent:
    """관리 이벤트"""
    event_id: str
    timestamp: datetime
    model_type: str
    event_type: ManagementAction
    trigger_reason: str
    quality_score: float
    action_taken: str
    result: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'model_type': self.model_type,
            'event_type': self.event_type.value,
            'trigger_reason': self.trigger_reason,
            'quality_score': self.quality_score,
            'action_taken': self.action_taken,
            'result': self.result,
            'metadata': self.metadata
        }

class ModelManagementSystem:
    """통합 모델 관리 시스템"""
    
    def __init__(self, config_file: str = "model_management_config.json"):
        self.config_file = config_file
        
        # 하위 시스템들
        self.quality_monitor = None
        self.tuning_manager = None
        
        # 관리 정책
        self.policies: Dict[str, ManagementPolicy] = {}
        self.events: List[ManagementEvent] = []
        
        # 자동 관리 스레드
        self._management_active = False
        self._management_thread = None
        self._lock = threading.RLock()
        
        # 통계
        self.stats = {
            'total_events': 0,
            'auto_tuning_triggered': 0,
            'backups_created': 0,
            'rollbacks_performed': 0,
            'alerts_sent': 0,
            'last_management_run': None
        }
        
        logger.info("Model management system initializing...")
        self._load_configuration()
        self._setup_default_policies()
    
    def _load_configuration(self):
        """설정 로드"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 정책 로드
                for policy_data in config.get('policies', []):
                    policy = ManagementPolicy(
                        model_type=policy_data['model_type'],
                        auto_tuning=AutoTuningPolicy(policy_data.get('auto_tuning', 'moderate')),
                        quality_alert_threshold=policy_data.get('quality_alert_threshold', 0.7),
                        emergency_rollback_threshold=policy_data.get('emergency_rollback_threshold', 0.5),
                        max_auto_tuning_per_day=policy_data.get('max_auto_tuning_per_day', 2),
                        backup_interval_hours=policy_data.get('backup_interval_hours', 24),
                        monitoring_interval_minutes=policy_data.get('monitoring_interval_minutes', 30),
                        enable_notifications=policy_data.get('enable_notifications', True),
                        custom_rules=policy_data.get('custom_rules', {})
                    )
                    self.policies[policy.model_type] = policy
                
                # 통계 로드
                self.stats.update(config.get('stats', {}))
                
                logger.info(f"Configuration loaded from {config_path}")
            else:
                logger.info("No configuration file found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def _save_configuration(self):
        """설정 저장"""
        try:
            config = {
                'policies': [
                    {
                        'model_type': policy.model_type,
                        'auto_tuning': policy.auto_tuning.value,
                        'quality_alert_threshold': policy.quality_alert_threshold,
                        'emergency_rollback_threshold': policy.emergency_rollback_threshold,
                        'max_auto_tuning_per_day': policy.max_auto_tuning_per_day,
                        'backup_interval_hours': policy.backup_interval_hours,
                        'monitoring_interval_minutes': policy.monitoring_interval_minutes,
                        'enable_notifications': policy.enable_notifications,
                        'custom_rules': policy.custom_rules
                    }
                    for policy in self.policies.values()
                ],
                'stats': self.stats
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _setup_default_policies(self):
        """기본 정책 설정"""
        default_models = ['finbert', 'ppo']
        
        for model_type in default_models:
            if model_type not in self.policies:
                if model_type == 'finbert':
                    policy = ManagementPolicy(
                        model_type=model_type,
                        auto_tuning=AutoTuningPolicy.MODERATE,
                        quality_alert_threshold=0.75,
                        emergency_rollback_threshold=0.6,
                        max_auto_tuning_per_day=1,
                        backup_interval_hours=48
                    )
                elif model_type == 'ppo':
                    policy = ManagementPolicy(
                        model_type=model_type,
                        auto_tuning=AutoTuningPolicy.CONSERVATIVE,
                        quality_alert_threshold=0.65,
                        emergency_rollback_threshold=0.45,
                        max_auto_tuning_per_day=2,
                        backup_interval_hours=24
                    )
                else:
                    policy = ManagementPolicy(model_type=model_type)
                
                self.policies[model_type] = policy
                logger.info(f"Created default policy for {model_type}")
    
    def initialize_subsystems(self):
        """하위 시스템 초기화"""
        try:
            # 품질 모니터 초기화
            from utils.model_quality_monitor import get_quality_monitor
            self.quality_monitor = get_quality_monitor()
            logger.info("Quality monitor initialized")
            
            # Fine-tuning 관리자 초기화  
            from utils.fine_tuning_manager import get_tuning_manager
            self.tuning_manager = get_tuning_manager()
            
            # 품질 모니터와 튜닝 관리자 연결
            self.tuning_manager.set_quality_monitor(self.quality_monitor)
            logger.info("Fine-tuning manager initialized and connected")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize subsystems: {e}")
            return False
    
    def set_model_policy(self, model_type: str, policy: ManagementPolicy):
        """모델 정책 설정"""
        self.policies[model_type] = policy
        self._save_configuration()
        logger.info(f"Policy updated for {model_type}")
    
    def get_model_policy(self, model_type: str) -> Optional[ManagementPolicy]:
        """모델 정책 조회"""
        return self.policies.get(model_type)
    
    async def evaluate_model_quality(self, model_type: str) -> Tuple[float, Dict[str, Any]]:
        """모델 품질 평가"""
        try:
            if not self.quality_monitor:
                return 0.0, {'error': 'Quality monitor not available'}
            
            # 품질 보고서 생성
            from utils.model_quality_monitor import ModelType
            model_type_enum = ModelType(model_type.lower())
            report = self.quality_monitor.generate_quality_report(model_type_enum)
            
            # 전체 품질 점수 계산
            if report.metrics:
                # 주요 메트릭들의 가중 평균
                weights = {
                    'accuracy': 0.3,
                    'confidence': 0.25,
                    'stability': 0.2,
                    'latency': 0.15,  # 낮을수록 좋음 (역산)
                    'memory_usage': 0.1  # 낮을수록 좋음 (역산)
                }
                
                quality_score = 0.0
                total_weight = 0.0
                
                for metric_name, value in report.metrics.items():
                    metric_key = metric_name.value if hasattr(metric_name, 'value') else str(metric_name)
                    weight = weights.get(metric_key, 0.1)
                    
                    # latency와 memory_usage는 역산 (낮을수록 좋음)
                    if metric_key in ['latency', 'memory_usage']:
                        # 정규화 (임계값 기준)
                        if metric_key == 'latency':
                            normalized_value = max(0, min(1, 1.0 - (value / 1000)))  # 1초 기준
                        else:  # memory_usage
                            normalized_value = max(0, min(1, 1.0 - (value / 2000)))  # 2GB 기준
                        quality_score += normalized_value * weight
                    else:
                        quality_score += value * weight
                    
                    total_weight += weight
                
                if total_weight > 0:
                    quality_score = quality_score / total_weight
                else:
                    quality_score = 0.5  # 기본값
            else:
                quality_score = 0.5  # 메트릭이 없을 때 기본값
            
            return quality_score, {
                'status': report.status.value,
                'metrics_count': len(report.metrics),
                'recommendations': report.recommendations,
                'issues': report.issues,
                'summary': report.summary
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate model quality for {model_type}: {e}")
            return 0.0, {'error': str(e)}
    
    async def check_management_triggers(self, model_type: str) -> List[ManagementAction]:
        """관리 트리거 검사"""
        try:
            policy = self.policies.get(model_type)
            if not policy:
                return []
            
            quality_score, quality_details = await self.evaluate_model_quality(model_type)
            actions = []
            
            # 긴급 롤백 필요성 검사
            if quality_score < policy.emergency_rollback_threshold:
                actions.append(ManagementAction.EMERGENCY_ROLLBACK)
                logger.warning(f"{model_type} quality critically low: {quality_score:.3f}")
            
            # 자동 튜닝 필요성 검사
            elif (policy.auto_tuning != AutoTuningPolicy.DISABLED and 
                  self._should_trigger_auto_tuning(model_type, quality_score, policy)):
                actions.append(ManagementAction.AUTO_TUNE)
                logger.info(f"{model_type} quality degraded, auto-tuning triggered: {quality_score:.3f}")
            
            # 품질 경고 필요성 검사
            if quality_score < policy.quality_alert_threshold:
                actions.append(ManagementAction.ALERT_ADMIN)
            
            # 백업 필요성 검사
            if self._should_create_backup(model_type, policy):
                actions.append(ManagementAction.BACKUP_MODEL)
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to check management triggers for {model_type}: {e}")
            return []
    
    def _should_trigger_auto_tuning(self, model_type: str, quality_score: float, 
                                   policy: ManagementPolicy) -> bool:
        """자동 튜닝 트리거 여부 판단"""
        try:
            # 정책에 따른 임계값 설정
            thresholds = {
                AutoTuningPolicy.AGGRESSIVE: 0.85,
                AutoTuningPolicy.MODERATE: 0.75,
                AutoTuningPolicy.CONSERVATIVE: 0.65
            }
            
            threshold = thresholds.get(policy.auto_tuning, 0.75)
            
            # 품질 점수 체크
            if quality_score >= threshold:
                return False
            
            # 일일 튜닝 횟수 제한 체크
            today = datetime.now().date()
            today_events = [
                e for e in self.events 
                if (e.model_type == model_type and 
                    e.event_type == ManagementAction.AUTO_TUNE and
                    e.timestamp.date() == today)
            ]
            
            if len(today_events) >= policy.max_auto_tuning_per_day:
                logger.info(f"Auto-tuning limit reached for {model_type} today: {len(today_events)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking auto-tuning trigger: {e}")
            return False
    
    def _should_create_backup(self, model_type: str, policy: ManagementPolicy) -> bool:
        """백업 생성 필요 여부 판단"""
        try:
            if not self.tuning_manager:
                return False
            
            # 마지막 백업 시간 확인
            backups = self.tuning_manager.list_backups(model_type)
            if not backups:
                return True  # 백업이 없으면 생성 필요
            
            latest_backup = backups[0]  # 최신순으로 정렬되어 있음
            backup_time = datetime.fromisoformat(latest_backup['created_at'])
            
            hours_since_backup = (datetime.now() - backup_time).total_seconds() / 3600
            
            return hours_since_backup >= policy.backup_interval_hours
            
        except Exception as e:
            logger.error(f"Error checking backup necessity: {e}")
            return False
    
    async def execute_management_action(self, model_type: str, action: ManagementAction,
                                      trigger_reason: str, quality_score: float) -> ManagementEvent:
        """관리 액션 실행"""
        try:
            # 이벤트 생성
            event_id = f"{model_type}_{action.value}_{int(time.time())}"
            event = ManagementEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                model_type=model_type,
                event_type=action,
                trigger_reason=trigger_reason,
                quality_score=quality_score,
                action_taken=""
            )
            
            # 액션별 실행
            if action == ManagementAction.AUTO_TUNE:
                await self._execute_auto_tune(event)
            elif action == ManagementAction.BACKUP_MODEL:
                await self._execute_backup_model(event)
            elif action == ManagementAction.EMERGENCY_ROLLBACK:
                await self._execute_emergency_rollback(event)
            elif action == ManagementAction.ALERT_ADMIN:
                await self._execute_alert_admin(event)
            else:
                event.action_taken = f"Monitoring {model_type}"
                event.result = "completed"
            
            # 이벤트 저장
            with self._lock:
                self.events.append(event)
                self.stats['total_events'] += 1
                self.stats['last_management_run'] = datetime.now().isoformat()
            
            logger.info(f"Management action executed: {action.value} for {model_type}")
            return event
            
        except Exception as e:
            logger.error(f"Failed to execute management action {action.value}: {e}")
            event.result = "failed"
            event.metadata['error'] = str(e)
            return event
    
    async def _execute_auto_tune(self, event: ManagementEvent):
        """자동 튜닝 실행"""
        try:
            if not self.tuning_manager:
                raise Exception("Tuning manager not available")
            
            # 튜닝 설정 생성
            from utils.fine_tuning_manager import TuningConfig, TuningStrategy
            config = TuningConfig(
                model_type=event.model_type,
                strategy=TuningStrategy.INCREMENTAL,
                max_epochs=3,
                quality_threshold=0.8,
                auto_rollback=True
            )
            
            # 튜닝 작업 생성 및 시작
            job_id = self.tuning_manager.create_tuning_job(event.model_type, config)
            success = await self.tuning_manager.start_tuning_job(job_id)
            
            if success:
                event.action_taken = f"Started auto-tuning job: {job_id}"
                event.result = "started"
                event.metadata['job_id'] = job_id
                self.stats['auto_tuning_triggered'] += 1
            else:
                event.action_taken = "Failed to start auto-tuning"
                event.result = "failed"
            
        except Exception as e:
            event.action_taken = f"Auto-tuning failed: {str(e)}"
            event.result = "failed"
            event.metadata['error'] = str(e)
    
    async def _execute_backup_model(self, event: ManagementEvent):
        """모델 백업 실행"""
        try:
            if not self.tuning_manager:
                raise Exception("Tuning manager not available")
            
            # 모델 경로 추정
            model_path = f"models/{event.model_type}_model"
            
            # 백업 생성
            from utils.fine_tuning_manager import ModelBackupLevel
            backup = self.tuning_manager.create_model_backup(
                event.model_type,
                model_path,
                ModelBackupLevel.STANDARD,
                f"Scheduled backup (quality: {event.quality_score:.3f})"
            )
            
            if backup:
                event.action_taken = f"Model backup created: {backup.backup_id}"
                event.result = "completed"
                event.metadata['backup_id'] = backup.backup_id
                self.stats['backups_created'] += 1
            else:
                event.action_taken = "Backup creation failed"
                event.result = "failed"
            
        except Exception as e:
            event.action_taken = f"Backup failed: {str(e)}"
            event.result = "failed"
            event.metadata['error'] = str(e)
    
    async def _execute_emergency_rollback(self, event: ManagementEvent):
        """긴급 롤백 실행"""
        try:
            if not self.tuning_manager:
                raise Exception("Tuning manager not available")
            
            # 최신 백업 찾기
            backups = self.tuning_manager.list_backups(event.model_type)
            if not backups:
                raise Exception("No backups available for rollback")
            
            latest_backup = backups[0]
            backup_id = latest_backup['backup_id']
            
            # 롤백 실행
            model_path = f"models/{event.model_type}_model"
            success = self.tuning_manager.restore_model_backup(backup_id, model_path)
            
            if success:
                event.action_taken = f"Emergency rollback completed using backup: {backup_id}"
                event.result = "completed"
                event.metadata['backup_id'] = backup_id
                self.stats['rollbacks_performed'] += 1
            else:
                event.action_taken = "Emergency rollback failed"
                event.result = "failed"
            
        except Exception as e:
            event.action_taken = f"Emergency rollback failed: {str(e)}"
            event.result = "failed"
            event.metadata['error'] = str(e)
    
    async def _execute_alert_admin(self, event: ManagementEvent):
        """관리자 알림 실행"""
        try:
            # 실제 구현에서는 이메일, Slack 등으로 알림 발송
            alert_message = (
                f"Model Quality Alert: {event.model_type}\n"
                f"Quality Score: {event.quality_score:.3f}\n"
                f"Trigger: {event.trigger_reason}\n"
                f"Timestamp: {event.timestamp.isoformat()}"
            )
            
            # 시뮬레이션: 로그로 알림
            logger.warning(f"ADMIN ALERT: {alert_message}")
            
            event.action_taken = "Admin alert sent"
            event.result = "completed"
            event.metadata['alert_message'] = alert_message
            self.stats['alerts_sent'] += 1
            
        except Exception as e:
            event.action_taken = f"Alert failed: {str(e)}"
            event.result = "failed"
            event.metadata['error'] = str(e)
    
    async def run_management_cycle(self):
        """관리 사이클 실행"""
        try:
            logger.debug("Running management cycle...")
            
            for model_type in self.policies.keys():
                try:
                    # 관리 트리거 검사
                    actions = await self.check_management_triggers(model_type)
                    
                    if not actions:
                        continue
                    
                    # 우선순위에 따라 액션 실행
                    priority_order = [
                        ManagementAction.EMERGENCY_ROLLBACK,
                        ManagementAction.AUTO_TUNE, 
                        ManagementAction.BACKUP_MODEL,
                        ManagementAction.ALERT_ADMIN
                    ]
                    
                    for priority_action in priority_order:
                        if priority_action in actions:
                            quality_score, _ = await self.evaluate_model_quality(model_type)
                            
                            await self.execute_management_action(
                                model_type=model_type,
                                action=priority_action,
                                trigger_reason=f"Quality score: {quality_score:.3f}",
                                quality_score=quality_score
                            )
                            
                            # 긴급 롤백이나 자동 튜닝 후에는 다른 액션 중단
                            if priority_action in [ManagementAction.EMERGENCY_ROLLBACK, ManagementAction.AUTO_TUNE]:
                                break
                
                except Exception as e:
                    logger.error(f"Error in management cycle for {model_type}: {e}")
            
            self._save_configuration()
            
        except Exception as e:
            logger.error(f"Error in management cycle: {e}")
    
    def start_management(self, interval_minutes: int = None):
        """자동 관리 시작"""
        if self._management_active:
            logger.warning("Management already active")
            return
        
        if not self.quality_monitor or not self.tuning_manager:
            logger.error("Subsystems not initialized")
            return
        
        # 기본 간격은 정책에서 가져오기
        if interval_minutes is None:
            intervals = [p.monitoring_interval_minutes for p in self.policies.values()]
            interval_minutes = min(intervals) if intervals else 30
        
        self._management_active = True
        
        def management_loop():
            while self._management_active:
                try:
                    asyncio.run(self.run_management_cycle())
                    
                    # 다음 실행까지 대기
                    for _ in range(interval_minutes * 60):
                        if not self._management_active:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in management loop: {e}")
                    time.sleep(60)  # 에러 시 1분 후 재시도
        
        self._management_thread = threading.Thread(target=management_loop, daemon=True)
        self._management_thread.start()
        
        logger.info(f"Model management started (interval: {interval_minutes} minutes)")
    
    def stop_management(self):
        """자동 관리 중지"""
        if not self._management_active:
            return
        
        self._management_active = False
        if self._management_thread and self._management_thread.is_alive():
            self._management_thread.join(timeout=5)
        
        logger.info("Model management stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            status = {
                'management_active': self._management_active,
                'subsystems': {
                    'quality_monitor': self.quality_monitor is not None,
                    'tuning_manager': self.tuning_manager is not None
                },
                'policies_count': len(self.policies),
                'events_count': len(self.events),
                'stats': self.stats.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            # 최근 이벤트 요약
            if self.events:
                recent_events = sorted(self.events, key=lambda x: x.timestamp, reverse=True)[:5]
                status['recent_events'] = [e.to_dict() for e in recent_events]
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def get_model_summary(self, model_type: str) -> Dict[str, Any]:
        """모델 요약 정보"""
        try:
            summary = {
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }
            
            # 정책 정보
            policy = self.policies.get(model_type)
            if policy:
                summary['policy'] = {
                    'auto_tuning': policy.auto_tuning.value,
                    'quality_alert_threshold': policy.quality_alert_threshold,
                    'emergency_rollback_threshold': policy.emergency_rollback_threshold
                }
            
            # 최근 이벤트
            model_events = [e for e in self.events if e.model_type == model_type]
            if model_events:
                latest_event = max(model_events, key=lambda x: x.timestamp)
                summary['latest_event'] = latest_event.to_dict()
                summary['events_count'] = len(model_events)
            
            # 하위 시스템 상태
            if self.quality_monitor:
                quality_status = self.quality_monitor.get_model_status(
                    getattr(__import__('utils.model_quality_monitor'), 'ModelType')(model_type.lower())
                )
                summary['quality_status'] = quality_status
            
            if self.tuning_manager:
                active_jobs = [job for job in self.tuning_manager.list_active_jobs() 
                             if job['model_type'] == model_type]
                backups = self.tuning_manager.list_backups(model_type)
                
                summary['tuning_status'] = {
                    'active_jobs': len(active_jobs),
                    'total_backups': len(backups),
                    'latest_backup': backups[0] if backups else None
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get model summary for {model_type}: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.stop_management()
            
            if self.quality_monitor:
                self.quality_monitor.cleanup()
            
            if self.tuning_manager:
                self.tuning_manager.cleanup()
            
            self._save_configuration()
            
            logger.info("Model management system cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# 전역 관리 시스템
_global_management_system = None

def get_management_system(config_file: str = None) -> ModelManagementSystem:
    """전역 관리 시스템 반환"""
    global _global_management_system
    if _global_management_system is None:
        _global_management_system = ModelManagementSystem(config_file or "model_management_config.json")
    return _global_management_system

# 사용 예시 및 테스트
if __name__ == "__main__":
    import asyncio
    
    async def test_management_system():
        print("🧪 Model Management System 테스트")
        
        system = get_management_system("test_management_config.json")
        
        print("\n1️⃣ 하위 시스템 초기화")
        success = system.initialize_subsystems()
        print(f"  ✅ 초기화: {'성공' if success else '실패'}")
        
        print("\n2️⃣ 정책 설정")
        from utils.model_management_system import ManagementPolicy, AutoTuningPolicy
        
        finbert_policy = ManagementPolicy(
            model_type="finbert",
            auto_tuning=AutoTuningPolicy.MODERATE,
            quality_alert_threshold=0.75,
            emergency_rollback_threshold=0.5
        )
        system.set_model_policy("finbert", finbert_policy)
        print(f"  📋 FinBERT 정책 설정 완료")
        
        print("\n3️⃣ 품질 평가")
        quality_score, details = await system.evaluate_model_quality("finbert")
        print(f"  📊 FinBERT 품질 점수: {quality_score:.3f}")
        print(f"  📈 상태: {details.get('status', 'unknown')}")
        
        print("\n4️⃣ 관리 트리거 검사")
        actions = await system.check_management_triggers("finbert")
        print(f"  🎯 트리거된 액션: {[a.value for a in actions]}")
        
        if actions:
            print("\n5️⃣ 관리 액션 실행")
            for action in actions[:2]:  # 최대 2개만 실행
                event = await system.execute_management_action(
                    "finbert", action, "테스트 트리거", quality_score
                )
                print(f"  ⚡ {action.value}: {event.result}")
        
        print("\n6️⃣ 시스템 상태 확인")
        status = system.get_system_status()
        print(f"  📈 관리 활성화: {status['management_active']}")
        print(f"  📊 총 이벤트: {status['events_count']}")
        print(f"  🎯 자동 튜닝 횟수: {status['stats']['auto_tuning_triggered']}")
        
        print("\n7️⃣ 모델 요약")
        summary = system.get_model_summary("finbert")
        print(f"  📋 FinBERT 이벤트: {summary.get('events_count', 0)}개")
        if 'latest_event' in summary:
            print(f"  ⏰ 최근 이벤트: {summary['latest_event']['event_type']}")
        
        print("\n🎉 Model Management System 테스트 완료!")
        
        # 정리
        system.cleanup()
        
        # 테스트 파일 정리
        test_files = [
            Path("test_management_config.json"),
            Path("model_quality.db"),
            Path("test_quality.db")
        ]
        for file_path in test_files:
            if file_path.exists():
                file_path.unlink()
    
    # 테스트 실행
    asyncio.run(test_management_system())