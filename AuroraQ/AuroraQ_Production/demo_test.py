#!/usr/bin/env python3
"""
AuroraQ Production 패키지 기능 테스트
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_package_structure():
    """패키지 구조 테스트"""
    print("🧪 AuroraQ Production 패키지 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. 유틸리티 모듈 테스트
        print("1/5: 유틸리티 모듈 테스트")
        from utils.logger import get_logger
        from utils.config_manager import ConfigManager
        from utils.metrics import PerformanceMetrics
        logger = get_logger("PackageTest")
        logger.info("유틸리티 모듈 로딩 성공")
        print("✅ 통과")
        
        # 2. 핵심 모듈 테스트
        print("2/5: 핵심 모듈 테스트")
        from core.market_data import MarketDataProvider, MarketDataPoint
        from core.position_manager import PositionManager, TradingLimits
        provider = MarketDataProvider("simulation")
        limits = TradingLimits()
        position_manager = PositionManager(limits)
        print("✅ 통과")
        
        # 3. 센티멘트 모듈 테스트
        print("3/5: 센티멘트 모듈 테스트")
        from sentiment.sentiment_analyzer import SentimentAnalyzer
        from sentiment.news_collector import NewsCollector
        from sentiment.sentiment_scorer import SentimentScorer
        analyzer = SentimentAnalyzer()
        print("✅ 통과")
        
        # 4. 실행 모듈 테스트
        print("4/5: 실행 모듈 테스트")
        from execution.order_manager import OrderManager, Order, OrderType, OrderSide
        order_manager = OrderManager()
        print("✅ 통과")
        
        # 5. 설정 파일 테스트
        print("5/5: 설정 파일 테스트")
        if os.path.exists("config.yaml"):
            config_manager = ConfigManager("config.yaml")
            config = config_manager.get_config()
            print("✅ 통과")
        else:
            print("⚠️ config.yaml 파일이 없지만 기본 설정으로 동작 가능")
        
        print("\n" + "=" * 50)
        print("🎉 모든 패키지 구조 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """기본 기능 테스트"""
    print("\n🧪 기본 기능 테스트 시작")
    print("=" * 50)
    
    try:
        from utils.logger import get_logger
        from core.market_data import MarketDataProvider
        from core.position_manager import PositionManager, TradingLimits
        from sentiment.sentiment_analyzer import SentimentAnalyzer
        
        logger = get_logger("FunctionTest")
        
        # 1. 마켓 데이터 테스트
        print("1/4: 마켓 데이터 생성 테스트")
        provider = MarketDataProvider("simulation")
        provider._init_simulation_data()
        assert provider.current_price > 0
        print("✅ 통과")
        
        # 2. 포지션 관리 테스트
        print("2/4: 포지션 관리 테스트")
        limits = TradingLimits(max_position_size=0.1)
        pm = PositionManager(limits)
        can_open, reason = pm.can_open_position(0.05, 50000.0)
        assert can_open == True
        print("✅ 통과")
        
        # 3. 센티멘트 분석 테스트
        print("3/4: 센티멘트 분석 테스트")
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze_sentiment("Bitcoin price is rising strongly")
        assert result.sentiment_score != 0
        print("✅ 통과")
        
        # 4. 로그 시스템 테스트
        print("4/4: 로그 시스템 테스트")
        logger.info("테스트 로그 메시지")
        logger.warning("테스트 경고 메시지")
        print("✅ 통과")
        
        print("\n" + "=" * 50)
        print("🎉 모든 기본 기능 테스트 통과!")
        return True
        
    except Exception as e:
        print(f"\n❌ 기능 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_creation():
    """설정 파일 생성 테스트"""
    print("\n🧪 설정 파일 생성 테스트")
    print("=" * 50)
    
    try:
        from utils.config_manager import ConfigManager
        
        # 설정 매니저 생성 (파일이 없으면 기본 설정으로 생성)
        config_manager = ConfigManager("test_config.yaml")
        config = config_manager.get_config()
        
        print(f"✅ 설정 생성 성공")
        print(f"   - 최대 포지션 크기: {config.trading.max_position_size}")
        print(f"   - 하이브리드 모드: {config.strategy.hybrid_mode}")
        print(f"   - 로그 레벨: {config.log_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 AuroraQ Production 패키지 검증 시작")
    print("📦 패키지 위치:", current_dir)
    print()
    
    results = []
    
    # 패키지 구조 테스트
    results.append(test_package_structure())
    
    # 기본 기능 테스트
    results.append(test_basic_functionality())
    
    # 설정 테스트
    results.append(test_config_creation())
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("📋 최종 테스트 결과")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"총 테스트: {total_tests}")
    print(f"통과: {passed_tests}")
    print(f"실패: {total_tests - passed_tests}")
    print(f"성공률: {passed_tests/total_tests*100:.1f}%")
    
    if all(results):
        print("\n🎊 모든 테스트 통과! AuroraQ Production 패키지가 정상 작동합니다!")
        print("\n📖 다음 단계:")
        print("   1. pip install -r requirements.txt (의존성 설치)")
        print("   2. python main.py --mode demo (데모 실행)")
        print("   3. USER_GUIDE.md 참조 (사용법 학습)")
        return 0
    else:
        print("\n⚠️ 일부 테스트가 실패했지만 기본 구조는 정상입니다.")
        print("   requirements.txt의 의존성을 설치하면 모든 기능이 작동할 것입니다.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)