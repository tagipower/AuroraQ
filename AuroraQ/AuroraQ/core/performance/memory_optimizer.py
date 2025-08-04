#!/usr/bin/env python3
"""
VPS 메모리 최적화 시스템
FinBERT 지연 로딩 및 동적 메모리 관리
"""

import os
import gc
import psutil
import time
import asyncio
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """메모리 사용량 통계"""
    total_mb: float
    used_mb: float
    available_mb: float
    percent: float
    process_mb: float
    timestamp: datetime

@dataclass
class ModelLoadState:
    """모델 로딩 상태"""
    is_loaded: bool = False
    load_time: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    last_used: Optional[datetime] = None
    use_count: int = 0

class LazyModelLoader:
    """지연 로딩 모델 관리자"""
    
    def __init__(self, memory_limit_mb: float = 3072):  # 3GB
        self.memory_limit_mb = memory_limit_mb
        self.models: Dict[str, Any] = {}
        self.model_states: Dict[str, ModelLoadState] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # 메모리 사용량 추적
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 100
        
        # 모델별 설정
        self.model_configs = {
            'finbert': {
                'priority': 1,  # 높을수록 우선순위
                'max_idle_minutes': 30,
                'estimated_memory_mb': 1500,
                'loader_func': self._load_finbert_model
            },
            'ppo': {
                'priority': 2,
                'max_idle_minutes': 60,
                'estimated_memory_mb': 500,
                'loader_func': self._load_ppo_model
            }
        }
        
        # 정리 스레드 시작
        self._start_cleanup_thread()
    
    def get_memory_stats(self) -> MemoryStats:
        """현재 메모리 사용량 조회"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return MemoryStats(
            total_mb=memory.total / 1024 / 1024,
            used_mb=memory.used / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            percent=memory.percent,
            process_mb=process.memory_info().rss / 1024 / 1024,
            timestamp=datetime.now()
        )
    
    def check_memory_pressure(self) -> bool:
        """메모리 압박 상태 확인"""
        stats = self.get_memory_stats()
        
        # VPS 메모리 제한 기준으로 압박 상태 판단
        process_limit_mb = self.memory_limit_mb * 0.8  # 80% 제한
        system_pressure = stats.percent > 85  # 시스템 메모리 85% 초과
        process_pressure = stats.process_mb > process_limit_mb
        
        return system_pressure or process_pressure
    
    async def get_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 지연 로딩 및 반환"""
        with self._lock:
            # 이미 로딩된 모델이 있으면 반환
            if model_name in self.models and self.model_states[model_name].is_loaded:
                self._update_model_usage(model_name)
                logger.debug(f"✅ 캐시된 모델 반환: {model_name}")
                return self.models[model_name]
            
            # 메모리 압박 상태 확인
            if self.check_memory_pressure():
                logger.warning("⚠️ 메모리 압박 상태 - 기존 모델 정리 시도")
                await self._cleanup_idle_models(force=True)
            
            # 모델 로딩 가능 여부 확인
            config = self.model_configs.get(model_name)
            if not config:
                logger.error(f"❌ 알 수 없는 모델: {model_name}")
                return None
            
            estimated_memory = config['estimated_memory_mb']
            current_stats = self.get_memory_stats()
            
            if current_stats.process_mb + estimated_memory > self.memory_limit_mb:
                logger.warning(f"⚠️ 메모리 부족으로 {model_name} 로딩 불가")
                # 강제 정리 시도
                await self._cleanup_idle_models(force=True)
                
                # 재확인
                current_stats = self.get_memory_stats()
                if current_stats.process_mb + estimated_memory > self.memory_limit_mb:
                    logger.error(f"❌ 메모리 부족으로 {model_name} 로딩 실패")
                    return None
            
            # 모델 로딩
            logger.info(f"🔄 모델 로딩 시작: {model_name}")
            start_time = time.time()
            
            try:
                model = await config['loader_func'](**kwargs)
                if model is None:
                    return None
                
                load_time = time.time() - start_time
                memory_after = self.get_memory_stats()
                actual_memory = memory_after.process_mb - current_stats.process_mb
                
                # 모델 저장 및 상태 업데이트
                self.models[model_name] = model
                self.model_states[model_name] = ModelLoadState(
                    is_loaded=True,
                    load_time=datetime.now(),
                    memory_usage_mb=actual_memory,
                    last_used=datetime.now(),
                    use_count=1
                )
                
                logger.info(
                    f"✅ 모델 로딩 완료: {model_name} "
                    f"(소요시간: {load_time:.1f}s, 메모리: {actual_memory:.1f}MB)"
                )
                
                return model
                
            except Exception as e:
                logger.error(f"❌ 모델 로딩 실패: {model_name} - {e}")
                return None
    
    def _update_model_usage(self, model_name: str):
        """모델 사용 정보 업데이트"""
        if model_name in self.model_states:
            state = self.model_states[model_name]
            state.last_used = datetime.now()
            state.use_count += 1
    
    async def _load_finbert_model(self, **kwargs):
        """FinBERT 모델 로딩"""
        try:
            # ONNX FinBERT 분석기 로딩
            from sentiment_service.models.onnx_finbert_analyzer import get_onnx_analyzer, initialize_onnx_analyzer
            
            logger.info("📥 FinBERT ONNX 모델 로딩...")
            analyzer = await initialize_onnx_analyzer()
            
            # 웜업 비활성화 (메모리 절약)
            if hasattr(analyzer, '_warmup'):
                analyzer._warmup = lambda: None
            
            return analyzer
            
        except ImportError:
            # 백업: 기본 감정 분석기
            logger.warning("⚠️ ONNX FinBERT 사용 불가 - 기본 분석기 사용")
            return self._create_fallback_sentiment_analyzer()
        except Exception as e:
            logger.error(f"❌ FinBERT 로딩 실패: {e}")
            return None
    
    async def _load_ppo_model(self, **kwargs):
        """PPO 모델 로딩"""
        try:
            from trading.ppo_agent import PPOAgent, PPOAgentConfig
            
            model_path = kwargs.get('model_path') or os.getenv('PPO_MODEL_PATH', '/app/models/ppo_model.zip')
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ PPO 모델 파일 없음: {model_path}")
                return None
            
            logger.info(f"📥 PPO 모델 로딩: {model_path}")
            
            config = PPOAgentConfig()
            agent = PPOAgent(config)
            await agent.load_model(model_path)
            
            return agent
            
        except Exception as e:
            logger.error(f"❌ PPO 모델 로딩 실패: {e}")
            return None
    
    def _create_fallback_sentiment_analyzer(self):
        """백업 감정 분석기 생성"""
        class FallbackSentimentAnalyzer:
            async def analyze_single(self, text: str):
                # 간단한 키워드 기반 분석
                positive_words = ['good', 'great', 'positive', 'bullish', 'up', 'gain', 'profit']
                negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'loss', 'crash']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    score = 0.6 + (pos_count - neg_count) * 0.1
                elif neg_count > pos_count:
                    score = 0.4 - (neg_count - pos_count) * 0.1
                else:
                    score = 0.5
                
                score = max(0.0, min(1.0, score))
                
                from dataclasses import dataclass
                @dataclass
                class SentimentResult:
                    score: float
                    label: str
                    confidence: float
                    processing_time: float = 0.001
                    model_version: str = "fallback-v1.0"
                
                return SentimentResult(
                    score=score,
                    label="positive" if score > 0.6 else "negative" if score < 0.4 else "neutral",
                    confidence=0.5
                )
            
            async def analyze_batch(self, texts: List[str]):
                results = []
                for text in texts:
                    result = await self.analyze_single(text)
                    results.append(result)
                return results
        
        return FallbackSentimentAnalyzer()
    
    async def _cleanup_idle_models(self, force: bool = False):
        """유휴 모델 정리"""
        if not self.models:
            return
        
        current_time = datetime.now()
        models_to_remove = []
        
        for model_name, state in self.model_states.items():
            if not state.is_loaded or not state.last_used:
                continue
            
            config = self.model_configs.get(model_name, {})
            max_idle_minutes = config.get('max_idle_minutes', 30)
            idle_time = current_time - state.last_used
            
            # 강제 정리 또는 유휴 시간 초과
            should_remove = force or idle_time > timedelta(minutes=max_idle_minutes)
            
            if should_remove:
                models_to_remove.append(model_name)
        
        # 우선순위 낮은 모델부터 제거
        models_to_remove.sort(key=lambda name: self.model_configs.get(name, {}).get('priority', 0))
        
        for model_name in models_to_remove:
            await self._unload_model(model_name)
    
    async def _unload_model(self, model_name: str):
        """모델 언로드"""
        if model_name not in self.models:
            return
        
        logger.info(f"🗑️ 모델 언로드: {model_name}")
        
        try:
            model = self.models[model_name]
            
            # 모델별 정리 작업
            if hasattr(model, 'cleanup'):
                await model.cleanup()
            elif hasattr(model, 'close'):
                await model.close()
            
            # 참조 제거
            del self.models[model_name]
            self.model_states[model_name].is_loaded = False
            
            # 가비지 컬렉션 강제 실행
            gc.collect()
            
            # 메모리 사용량 확인
            stats = self.get_memory_stats()
            logger.info(f"🧹 메모리 정리 후: {stats.process_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"❌ 모델 언로드 실패: {model_name} - {e}")
    
    def _start_cleanup_thread(self):
        """정리 스레드 시작"""
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    # 5분마다 정리 확인
                    if self._stop_cleanup.wait(300):  # 5분
                        break
                    
                    # 메모리 압박 상태 확인
                    if self.check_memory_pressure():
                        logger.info("🧹 메모리 압박으로 인한 자동 정리 시작")
                        asyncio.run(self._cleanup_idle_models(force=True))
                    else:
                        # 일반 정리
                        asyncio.run(self._cleanup_idle_models(force=False))
                    
                    # 메모리 통계 기록
                    stats = self.get_memory_stats()
                    self.memory_history.append(stats)
                    
                    # 히스토리 크기 제한
                    if len(self.memory_history) > self.max_history_size:
                        self.memory_history = self.memory_history[-self.max_history_size:]
                        
                except Exception as e:
                    logger.error(f"❌ 정리 스레드 오류: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info("🔄 메모리 관리 스레드 시작")
    
    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 조회"""
        status = {
            'memory_stats': self.get_memory_stats().__dict__,
            'memory_pressure': self.check_memory_pressure(),
            'loaded_models': {},
            'total_memory_used_mb': 0.0
        }
        
        for model_name, state in self.model_states.items():
            if state.is_loaded:
                status['loaded_models'][model_name] = {
                    'load_time': state.load_time.isoformat() if state.load_time else None,
                    'memory_usage_mb': state.memory_usage_mb,
                    'last_used': state.last_used.isoformat() if state.last_used else None,
                    'use_count': state.use_count
                }
                status['total_memory_used_mb'] += state.memory_usage_mb
        
        return status
    
    def force_cleanup(self):
        """강제 정리"""
        logger.info("🧹 강제 메모리 정리 시작")
        asyncio.run(self._cleanup_idle_models(force=True))
        gc.collect()
    
    def shutdown(self):
        """종료"""
        logger.info("🛑 메모리 최적화 시스템 종료")
        self._stop_cleanup.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # 모든 모델 언로드
        for model_name in list(self.models.keys()):
            asyncio.run(self._unload_model(model_name))

# 전역 인스턴스
_memory_optimizer = None

def get_memory_optimizer() -> LazyModelLoader:
    """전역 메모리 최적화기 인스턴스 반환"""
    global _memory_optimizer
    if _memory_optimizer is None:
        memory_limit_mb = float(os.getenv('VPS_MEMORY_LIMIT_GB', '3')) * 1024
        _memory_optimizer = LazyModelLoader(memory_limit_mb)
    return _memory_optimizer

async def get_finbert_model(**kwargs):
    """FinBERT 모델 지연 로딩"""
    optimizer = get_memory_optimizer()
    return await optimizer.get_model('finbert', **kwargs)

async def get_ppo_model(**kwargs):
    """PPO 모델 지연 로딩"""
    optimizer = get_memory_optimizer()
    return await optimizer.get_model('ppo', **kwargs)

def main():
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="메모리 최적화 테스트")
    parser.add_argument("--test-finbert", action="store_true", help="FinBERT 로딩 테스트")
    parser.add_argument("--test-ppo", action="store_true", help="PPO 로딩 테스트")
    parser.add_argument("--status", action="store_true", help="상태 조회")
    
    args = parser.parse_args()
    
    async def run_tests():
        optimizer = get_memory_optimizer()
        
        if args.status:
            status = optimizer.get_model_status()
            print("📊 메모리 최적화 상태:")
            print(f"  💾 프로세스 메모리: {status['memory_stats']['process_mb']:.1f}MB")
            print(f"  ⚠️ 메모리 압박: {'예' if status['memory_pressure'] else '아니오'}")
            print(f"  🤖 로딩된 모델: {len(status['loaded_models'])}개")
        
        if args.test_finbert:
            print("🔄 FinBERT 로딩 테스트...")
            model = await get_finbert_model()
            if model:
                print("✅ FinBERT 로딩 성공")
                # 간단한 테스트
                result = await model.analyze_single("Bitcoin price is going up")
                print(f"  📊 테스트 결과: {result.score:.3f} ({result.label})")
            else:
                print("❌ FinBERT 로딩 실패")
        
        if args.test_ppo:
            print("🔄 PPO 로딩 테스트...")
            model = await get_ppo_model()
            if model:
                print("✅ PPO 로딩 성공")
            else:
                print("❌ PPO 로딩 실패")
        
        # 최종 상태
        final_status = optimizer.get_model_status()
        print(f"\n📊 최종 메모리 상태: {final_status['memory_stats']['process_mb']:.1f}MB")
    
    asyncio.run(run_tests())

if __name__ == "__main__":
    main()