#!/usr/bin/env python3
"""
PPO 통합 검증 스크립트
VPS deployment에서 PPO 전략 통합 상태 확인
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import asyncio
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
script_dir = Path(__file__).parent
vps_root = script_dir.parent
sys.path.insert(0, str(vps_root))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ppo_strategy_import():
    """PPO 전략 임포트 테스트"""
    logger.info("🧪 PPO 전략 임포트 테스트 시작")
    
    try:
        from trading.ppo_strategy import PPOStrategy, PPOConfig
        logger.info("✅ PPOStrategy 임포트 성공")
        
        # 설정 객체 생성 테스트
        config = PPOConfig(
            model_path="/tmp/test_model.zip",
            confidence_threshold=0.6
        )
        logger.info(f"✅ PPOConfig 생성 성공: {config}")
        
        # 전략 객체 생성 테스트 (모델 없이)
        strategy = PPOStrategy(config)
        logger.info(f"✅ PPOStrategy 인스턴스 생성 성공: {strategy.name}")
        
        return True, strategy
        
    except ImportError as e:
        logger.error(f"❌ PPO 임포트 실패: {e}")
        return False, None
    except Exception as e:
        logger.error(f"❌ PPO 초기화 실패: {e}")
        return False, None

def test_vps_adapter_integration():
    """VPS 어댑터 통합 테스트"""
    logger.info("🔗 VPS 어댑터 통합 테스트 시작")
    
    try:
        from trading.vps_strategy_adapter import create_enhanced_vps_strategy_adapter
        
        # 테스트 설정
        config = {
            'enabled_strategies': ['PPOStrategy'],
            'ppo_model_path': '/tmp/test_model.zip',
            'ppo_confidence_threshold': 0.6,
            'max_concurrent_strategies': 6
        }
        
        adapter = create_enhanced_vps_strategy_adapter(config)
        logger.info("✅ VPS 어댑터 생성 성공")
        
        # 전략 목록 확인
        available_strategies = adapter.get_available_strategies()
        logger.info(f"📋 사용 가능한 전략: {list(available_strategies.keys())}")
        
        if 'PPOStrategy' in available_strategies:
            logger.info("✅ PPOStrategy가 어댑터에 등록됨")
        else:
            logger.warning("⚠️ PPOStrategy가 어댑터에 등록되지 않음")
        
        return True, adapter
        
    except Exception as e:
        logger.error(f"❌ VPS 어댑터 통합 실패: {e}")
        return False, None

async def test_signal_generation():
    """신호 생성 테스트 (감정 분석 통합 포함)"""
    logger.info("📊 신호 생성 테스트 시작")
    
    try:
        from trading.ppo_strategy import PPOStrategy, PPOConfig
        
        # 테스트 데이터 생성
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        test_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50200,
            'low': np.random.randn(100).cumsum() + 49800,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(100, 1000, 100)
        })
        
        # PPO 전략으로 신호 생성
        config = PPOConfig(confidence_threshold=0.5)
        strategy = PPOStrategy(config)
        
        # 감정 분석 통합 신호 생성 (async)
        signal = await strategy.generate_signal(test_data, "BTCUSDT")
        logger.info(f"📈 생성된 신호: {signal['action']} (강도: {signal['strength']:.3f})")
        
        # 감정 분석 통합 확인
        metadata = signal.get('metadata', {})
        sentiment_integrated = metadata.get('sentiment_integrated', False)
        sentiment_boost = metadata.get('sentiment_boost', False)
        
        logger.info(f"🧠 감정 분석 통합: {'✅' if sentiment_integrated else '❌'}")
        logger.info(f"⚡ 감정 부스트: {'✅' if sentiment_boost else '❌'}")
        
        if sentiment_integrated:
            sentiment_features = metadata.get('sentiment_features', [])
            if sentiment_features:
                logger.info(f"📊 감정 특성: {[f'{f:.3f}' for f in sentiment_features[:3]]}")
        
        # 점수 계산 테스트
        score, details = strategy.score(test_data)
        logger.info(f"🎯 전략 점수: {score:.3f}")
        logger.info(f"📋 상세 점수: {list(details.keys())}")
        
        return True, signal
        
    except Exception as e:
        logger.error(f"❌ 신호 생성 실패: {e}")
        return False, None

def test_config_loading():
    """설정 파일 로딩 테스트"""
    logger.info("⚙️ 설정 파일 로딩 테스트 시작")
    
    config_path = vps_root / "trading" / "config" / "vps_trading_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info("✅ 설정 파일 로딩 성공")
        
        # PPO 관련 설정 확인
        strategies = config.get('strategies', {})
        enabled_strategies = strategies.get('enabled_strategies', [])
        
        if 'PPOStrategy' in enabled_strategies:
            logger.info("✅ PPOStrategy가 활성화됨")
        else:
            logger.warning("⚠️ PPOStrategy가 비활성화 상태")
        
        # PPO 설정 섹션 확인
        rule_strategies = strategies.get('rule_strategies', {})
        if 'PPOStrategy' in rule_strategies:
            ppo_config = rule_strategies['PPOStrategy']
            logger.info(f"📋 PPO 설정: {ppo_config}")
        else:
            logger.warning("⚠️ PPO 상세 설정이 없음")
        
        # 어댑터 설정 확인
        adapter_config = strategies.get('enhanced_adapter', {})
        max_strategies = adapter_config.get('max_concurrent_strategies', 0)
        ppo_model_path = adapter_config.get('ppo_model_path')
        
        logger.info(f"🔧 최대 동시 전략: {max_strategies}")
        logger.info(f"🗂️ PPO 모델 경로: {ppo_model_path}")
        
        return True, config
        
    except Exception as e:
        logger.error(f"❌ 설정 파일 로딩 실패: {e}")
        return False, None

def test_sentiment_integration():
    """감정 분석 통합 테스트"""
    logger.info("🧠 감정 분석 통합 테스트 시작")
    
    try:
        from trading.sentiment_integration import get_sentiment_client, SentimentScore, MarketSentiment
        
        # 감정 분석 클라이언트 생성
        client = get_sentiment_client()
        logger.info("✅ 감정 분석 클라이언트 생성 성공")
        
        # 기본 감정 점수 객체 테스트
        score = SentimentScore(value=0.5, confidence=0.8)
        logger.info(f"📊 감정 점수 객체: value={score.value}, confidence={score.confidence}")
        
        # 시장 감정 객체 테스트
        market_sentiment = MarketSentiment(overall_score=0.3, fear_greed_index=0.6)
        feature_vector = market_sentiment.to_feature_vector()
        logger.info(f"📈 시장 감정 특성 벡터: {len(feature_vector)}개 특성")
        
        # PPO 전략에서 감정 통합 확인
        from trading.ppo_strategy import PPOStrategy, SENTIMENT_AVAILABLE
        
        if SENTIMENT_AVAILABLE:
            logger.info("✅ PPO 전략에 감정 분석 통합됨")
        else:
            logger.warning("⚠️ PPO 전략에 감정 분석 통합되지 않음")
        
        return True, {
            'client': client,
            'sentiment_available': SENTIMENT_AVAILABLE,
            'feature_vector_size': len(feature_vector)
        }
        
    except Exception as e:
        logger.error(f"❌ 감정 분석 통합 실패: {e}")
        return False, None

def test_logging_integration():
    """로깅 통합 테스트"""
    logger.info("📝 로깅 통합 테스트 시작")
    
    try:
        from vps_logging.vps_integration import get_vps_log_integrator
        
        integrator = get_vps_log_integrator("/tmp/test_logs")
        logger.info("✅ VPS 로깅 통합기 생성 성공")
        
        # 통계 확인
        stats = integrator.get_stats() 
        logger.info(f"📊 로깅 통계: {stats}")
        
        return True, integrator
        
    except Exception as e:
        logger.error(f"❌ 로깅 통합 실패: {e}")
        return False, None

def test_dependencies():
    """의존성 테스트"""
    logger.info("📦 의존성 테스트 시작")
    
    dependencies = {
        'torch': 'PyTorch',
        'stable_baselines3': 'Stable Baselines3',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    results = {}
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"✅ {name} 사용 가능")
            results[module] = True
        except ImportError:
            logger.warning(f"⚠️ {name} 사용 불가 (선택적 의존성)")
            results[module] = False
    
    return results

async def main():
    """메인 검증 함수"""
    logger.info("🚀 PPO 통합 검증 시작")
    logger.info(f"📍 VPS 루트: {vps_root}")
    
    results = {}
    
    # 1. 의존성 테스트
    logger.info("\n" + "="*50)
    results['dependencies'] = test_dependencies()
    
    # 2. PPO 전략 임포트 테스트
    logger.info("\n" + "="*50)
    results['ppo_import'] = test_ppo_strategy_import()
    
    # 3. VPS 어댑터 통합 테스트
    logger.info("\n" + "="*50)
    results['adapter_integration'] = test_vps_adapter_integration()
    
    # 4. 신호 생성 테스트 (async)
    logger.info("\n" + "="*50)
    results['signal_generation'] = await test_signal_generation()
    
    # 5. 감정 분석 통합 테스트
    logger.info("\n" + "="*50)
    results['sentiment_integration'] = test_sentiment_integration()
    
    # 6. 설정 파일 로딩 테스트
    logger.info("\n" + "="*50)
    results['config_loading'] = test_config_loading()
    
    # 7. 로깅 통합 테스트
    logger.info("\n" + "="*50)
    results['logging_integration'] = test_logging_integration()
    
    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info("📋 검증 결과 요약")
    logger.info("="*50)
    
    success_count = 0
    total_count = 0
    
    for test_name, result in results.items():
        if test_name == 'dependencies':
            continue
            
        status = "✅ 성공" if result[0] else "❌ 실패"
        logger.info(f"{test_name}: {status}")
        
        if result[0]:
            success_count += 1
        total_count += 1
    
    # 의존성 요약
    deps = results.get('dependencies', {})
    optional_deps = sum(1 for available in deps.values() if available)
    logger.info(f"선택적 의존성: {optional_deps}/{len(deps)} 사용가능")
    
    logger.info(f"\n전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        logger.info("🎉 모든 테스트 통과! PPO 통합이 성공적으로 완료되었습니다.")
        return 0
    else:
        logger.warning(f"⚠️ {total_count - success_count}개 테스트 실패. 문제를 확인해주세요.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)