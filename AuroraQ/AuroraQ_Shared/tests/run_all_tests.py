#!/usr/bin/env python3
"""
전체 테스트 스위트 실행 스크립트
리스크 관리, 보정 시스템, 통합 시스템 테스트
"""

import unittest
import sys
import os
from datetime import datetime
import json

# 테스트 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 테스트 모듈 임포트
from test_risk_management import *
from test_calibration_system import *
from test_integration_system import *


class TestReport:
    """테스트 결과 보고서 생성"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def start_testing(self):
        """테스트 시작"""
        self.start_time = datetime.now()
        print("=" * 70)
        print("🚀 AuroraQ 통합 시스템 테스트 시작")
        print(f"⏰ 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
    def end_testing(self):
        """테스트 종료"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("✅ AuroraQ 통합 시스템 테스트 완료")
        print(f"⏰ 종료 시간: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⌛ 실행 시간: {duration:.2f}초")
        print("=" * 70)
        
    def add_test_result(self, test_name, result):
        """테스트 결과 추가"""
        self.results[test_name] = {
            'success': result.wasSuccessful(),
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0
        }
        
    def print_summary(self):
        """결과 요약 출력"""
        print("\n📊 테스트 결과 요약:")
        print("-" * 50)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        successful_modules = 0
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{test_name:30} {status:8} "
                  f"({result['tests_run']} tests, "
                  f"{result['failures']} failures, "
                  f"{result['errors']} errors)")
            
            total_tests += result['tests_run']
            total_failures += result['failures']
            total_errors += result['errors']
            total_skipped += result['skipped']
            
            if result['success']:
                successful_modules += 1
        
        print("-" * 50)
        print(f"📈 전체 통계:")
        print(f"   모듈: {successful_modules}/{len(self.results)} 성공")
        print(f"   테스트: {total_tests} 실행")
        print(f"   실패: {total_failures}")
        print(f"   에러: {total_errors}")
        print(f"   건너뜀: {total_skipped}")
        
        success_rate = (total_tests - total_failures - total_errors) / total_tests * 100 if total_tests > 0 else 0
        print(f"   성공률: {success_rate:.1f}%")
        
        return total_failures == 0 and total_errors == 0
        
    def export_results(self, filename="test_results.json"):
        """결과를 JSON 파일로 내보내기"""
        report_data = {
            'test_run_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds()
            },
            'test_results': self.results,
            'summary': {
                'total_modules': len(self.results),
                'successful_modules': sum(1 for r in self.results.values() if r['success']),
                'total_tests': sum(r['tests_run'] for r in self.results.values()),
                'total_failures': sum(r['failures'] for r in self.results.values()),
                'total_errors': sum(r['errors'] for r in self.results.values())
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 테스트 결과가 {filename}에 저장되었습니다.")


def run_test_module(test_module_name, test_classes):
    """개별 테스트 모듈 실행"""
    print(f"\n🧪 {test_module_name} 테스트 실행 중...")
    
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


def main():
    """메인 테스트 실행"""
    report = TestReport()
    report.start_testing()
    
    # 테스트 모듈 정의
    test_modules = [
        ("리스크 관리 시스템", [
            TestVaRCalculator,
            TestAdvancedRiskManager,
            TestPortfolioRiskAnalyzer,
            TestRiskIntegration,
            TestRiskMetrics
        ]),
        ("보정 시스템", [
            TestExecutionAnalyzer,
            TestMarketConditionDetector,
            TestCalibrationManager,
            TestRealTradeMonitor,
            TestCalibrationIntegration
        ]),
        ("통합 시스템", [
            TestBacktestRiskIntegration,
            TestConvenienceFunctions,
            TestPerformanceAndScalability,
            TestErrorHandlingAndRecovery
        ])
    ]
    
    # 각 모듈별 테스트 실행
    all_success = True
    
    for module_name, test_classes in test_modules:
        try:
            result = run_test_module(module_name, test_classes)
            report.add_test_result(module_name, result)
            
            if not result.wasSuccessful():
                all_success = False
                
        except Exception as e:
            print(f"❌ {module_name} 테스트 실행 중 오류 발생: {e}")
            all_success = False
    
    # 테스트 종료 및 결과 요약
    report.end_testing()
    final_success = report.print_summary()
    report.export_results()
    
    # 최종 결과 출력
    if final_success and all_success:
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        return 0
    else:
        print("\n💥 일부 테스트가 실패했습니다. 로그를 확인하세요.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)