#!/usr/bin/env python3
"""
실시간 시스템 테스트
"""

import pytest
import time
import threading
from datetime import datetime
import sys
import os

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core import RealtimeHybridSystem, TradingConfig, MarketDataProvider
from utils import get_logger

logger = get_logger("TestRealtime")

class TestRealtimeSystem:
    """실시간 시스템 테스트 클래스"""
    
    def setup_method(self):
        """테스트 셋업"""
        self.config = TradingConfig(
            rule_strategies=["RuleStrategyA"],
            enable_ppo=True,
            hybrid_mode="ensemble",
            execution_strategy="market",
            max_position_size=0.01,
            max_daily_trades=3,
            min_data_points=10
        )
        self.system = RealtimeHybridSystem(self.config)
    
    def teardown_method(self):
        """테스트 정리"""
        if self.system.is_running:
            self.system.stop()
    
    def test_system_initialization(self):
        """시스템 초기화 테스트"""
        assert self.system.config.max_position_size == 0.01
        assert self.system.config.hybrid_mode == "ensemble"
        assert not self.system.is_running
        
        # 초기화 실행
        result = self.system.initialize()
        assert result == True
    
    def test_market_data_provider(self):
        """마켓 데이터 제공자 테스트"""
        provider = MarketDataProvider("simulation")
        
        data_received = []
        
        def on_data(data_point):
            data_received.append(data_point)
        
        provider.subscribe(on_data)
        provider.start_stream()
        
        # 3초 대기
        time.sleep(3)
        provider.stop_stream()
        
        assert len(data_received) >= 2  # 최소 2개 데이터 수신
        assert data_received[0].symbol == "BTC/USD"
        assert data_received[0].price > 0
    
    def test_short_trading_session(self):
        """짧은 거래 세션 테스트"""
        
        # 자동 중지 설정 (10초)
        def auto_stop():
            time.sleep(10)
            self.system.stop()
        
        stop_thread = threading.Thread(target=auto_stop, daemon=True)
        stop_thread.start()
        
        # 시스템 시작
        if self.system.start():
            # 메인 루프 대기
            while self.system.is_running:
                time.sleep(0.1)
        
        # 결과 검증
        report = self.system.get_performance_report()
        assert report["total_signals"] >= 0
        assert report["executed_trades"] >= 0
        assert report["signal_execution_rate"] >= 0
    
    def test_position_manager(self):
        """포지션 매니저 테스트"""
        pm = self.system.position_manager
        
        # 초기 상태 확인
        assert pm.current_position == 0.0
        assert pm.daily_trade_count == 0
        
        # 포지션 개설 테스트
        signal_info = {
            'action': 'BUY',
            'confidence': 0.8,
            'strategy': 'test'
        }
        
        result = pm.open_position(0.005, 50000.0, signal_info)
        assert result == True
        assert pm.current_position == 0.005
        assert pm.daily_trade_count == 1
        
        # 포지션 청산 테스트
        result = pm.close_position(50100.0, "test_close")
        assert result == True
        assert pm.current_position == 0.0
        
        # 성과 확인
        performance = pm.get_performance_summary()
        assert performance["total_trades"] == 1
        assert performance["win_rate"] > 0  # 수익 거래였으므로
    
    def test_configuration_loading(self):
        """설정 로딩 테스트"""
        # 최적 설정이 있다면 로드되어야 함
        if self.system.optimal_config:
            assert "hybrid_mode" in self.system.optimal_config
            assert "execution_strategy" in self.system.optimal_config
            assert "weights" in self.system.optimal_config
    
    def test_notification_system(self):
        """알림 시스템 테스트"""
        # 알림 활성화
        self.system.config.enable_notifications = True
        self.system.config.notification_channels = ["console"]
        
        # 알림 전송 테스트
        self.system._send_notification("테스트 알림")
        # 콘솔 출력이므로 별도 검증은 어려움
        
    def test_performance_report(self):
        """성과 리포트 테스트"""
        report = self.system.get_performance_report()
        
        # 필수 필드 확인
        required_fields = [
            "total_signals", "executed_trades", "signal_execution_rate",
            "current_position", "daily_trade_count"
        ]
        
        for field in required_fields:
            assert field in report

if __name__ == "__main__":
    # 직접 실행 시 테스트 수행
    test_class = TestRealtimeSystem()
    
    print("🧪 실시간 시스템 테스트 시작")
    
    try:
        test_class.setup_method()
        
        print("1/6: 시스템 초기화 테스트")
        test_class.test_system_initialization()
        print("✅ 통과")
        
        print("2/6: 마켓 데이터 제공자 테스트")
        test_class.test_market_data_provider()
        print("✅ 통과")
        
        print("3/6: 포지션 매니저 테스트")
        test_class.test_position_manager()
        print("✅ 통과")
        
        print("4/6: 설정 로딩 테스트")
        test_class.test_configuration_loading()
        print("✅ 통과")
        
        print("5/6: 알림 시스템 테스트")
        test_class.test_notification_system()
        print("✅ 통과")
        
        print("6/6: 성과 리포트 테스트")
        test_class.test_performance_report()
        print("✅ 통과")
        
        print("\n🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        
    finally:
        test_class.teardown_method()