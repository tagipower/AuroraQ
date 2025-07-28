#!/usr/bin/env python3
"""
AuroraQ Unified Test Runner
===========================

Centralized test runner for all AuroraQ components:
- AuroraQ_Shared modules (utilities, risk management, position management, calibration)
- AuroraQ_Production modules (strategies, optimization, real-time)
- AuroraQ_Backtest modules (backtest engine, performance analysis)
- Integration tests between components
"""

import sys
import os
import importlib
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
shared_path = project_root / "AuroraQ_Shared"
production_path = project_root / "AuroraQ_Production"
backtest_path = project_root / "AuroraQ_Backtest"

for path in [shared_path, production_path, backtest_path, project_root]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import shared utilities
try:
    from AuroraQ_Shared.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("UnifiedTestRunner")

class TestResult:
    """Test result container"""
    
    def __init__(self, name: str, success: bool, error: str = "", duration: float = 0.0):
        self.name = name
        self.success = success
        self.error = error
        self.duration = duration
        self.timestamp = datetime.now()

class TestSuite:
    """Test suite container"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results: List[TestResult] = []
        self.setup_error: str = ""
        self.start_time = None
        self.end_time = None
    
    def add_result(self, result: TestResult):
        """Add test result"""
        self.results.append(result)
    
    @property
    def total_tests(self) -> int:
        return len(self.results)
    
    @property
    def passed_tests(self) -> int:
        return sum(1 for r in self.results if r.success)
    
    @property
    def failed_tests(self) -> int:
        return self.total_tests - self.passed_tests
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / max(1, self.total_tests)
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class UnifiedTestRunner:
    """Unified test runner for all AuroraQ components"""
    
    def __init__(self):
        self.test_suites: List[TestSuite] = []
        self.start_time = None
        self.end_time = None
    
    def discover_and_run_tests(self, 
                              test_modules: List[str] = None,
                              include_integration: bool = True,
                              verbose: bool = True) -> Dict[str, Any]:
        """Discover and run all tests"""
        
        self.start_time = datetime.now()
        
        if test_modules is None:
            test_modules = self._discover_test_modules()
        
        logger.info(f"🧪 Starting AuroraQ Unified Test Suite ({len(test_modules)} modules)")
        
        # Run individual test modules
        for module_name in test_modules:
            suite = self._run_test_module(module_name, verbose)
            self.test_suites.append(suite)
        
        # Run integration tests if requested
        if include_integration:
            integration_suite = self._run_integration_tests(verbose)
            self.test_suites.append(integration_suite)
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        return self._generate_report(verbose)
    
    def _discover_test_modules(self) -> List[str]:
        """Discover all test modules"""
        test_modules = []
        
        # Shared tests
        shared_tests = [
            "AuroraQ_Shared.tests.test_basic_functionality",
            "AuroraQ_Shared.tests.test_risk_management", 
            "AuroraQ_Shared.tests.test_calibration_system",
            "AuroraQ_Shared.tests.test_realtime_system",
            "AuroraQ_Shared.tests.test_integration_system",
            "AuroraQ_Shared.tests.test_production_modules"
        ]
        
        # Check which modules actually exist
        for module_name in shared_tests:
            if self._module_exists(module_name):
                test_modules.append(module_name)
        
        # Additional test discovery from other components
        additional_paths = [
            (project_root / "tests", "tests"),
            (production_path / "tests", "AuroraQ_Production.tests"), 
            (backtest_path / "tests", "AuroraQ_Backtest.tests")
        ]
        
        for test_dir, module_prefix in additional_paths:
            if test_dir.exists():
                for test_file in test_dir.glob("test_*.py"):
                    module_name = f"{module_prefix}.{test_file.stem}"
                    if self._module_exists(module_name):
                        test_modules.append(module_name)
        
        return test_modules
    
    def _module_exists(self, module_name: str) -> bool:
        """Check if module exists and is importable"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def _run_test_module(self, module_name: str, verbose: bool = True) -> TestSuite:
        """Run tests from a specific module"""
        
        suite = TestSuite(module_name, f"Tests from {module_name}")
        suite.start_time = datetime.now()
        
        if verbose:
            print(f"\n📋 Running {module_name}")
        
        try:
            # Import the test module
            test_module = importlib.import_module(module_name)
            
            # Find test classes
            test_classes = []
            for attr_name in dir(test_module):
                attr = getattr(test_module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name.startswith('Test') and 
                    attr != type):
                    test_classes.append(attr)
            
            # Run tests from each class
            for test_class in test_classes:
                self._run_test_class(test_class, suite, verbose)
            
            # Check for standalone test functions
            test_functions = [getattr(test_module, name) 
                            for name in dir(test_module) 
                            if name.startswith('test_') and callable(getattr(test_module, name))]
            
            for test_func in test_functions:
                self._run_test_function(test_func, suite, verbose)
                
        except Exception as e:
            suite.setup_error = f"Failed to load module {module_name}: {str(e)}"
            logger.error(f"Module loading error: {e}")
            if verbose:
                print(f"  ❌ Module loading failed: {e}")
        
        suite.end_time = datetime.now()
        return suite
    
    def _run_test_class(self, test_class, suite: TestSuite, verbose: bool = True):
        """Run all test methods from a test class"""
        
        try:
            test_instance = test_class()
            
            # Get all test methods
            test_methods = [name for name in dir(test_instance) 
                          if name.startswith('test_') and callable(getattr(test_instance, name))]
            
            for method_name in test_methods:
                result = self._run_single_test(test_instance, method_name, verbose)
                suite.add_result(result)
                
        except Exception as e:
            error_msg = f"Failed to instantiate {test_class.__name__}: {str(e)}"
            result = TestResult(f"{test_class.__name__}.__init__", False, error_msg)
            suite.add_result(result)
            if verbose:
                print(f"  ❌ {test_class.__name__}: {error_msg}")
    
    def _run_test_function(self, test_func, suite: TestSuite, verbose: bool = True):
        """Run a standalone test function"""
        
        start_time = datetime.now()
        
        try:
            test_func()
            duration = (datetime.now() - start_time).total_seconds()
            result = TestResult(test_func.__name__, True, "", duration)
            
            if verbose:
                print(f"  ✅ {test_func.__name__}")
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            result = TestResult(test_func.__name__, False, error_msg, duration)
            
            if verbose:
                print(f"  ❌ {test_func.__name__}: {error_msg}")
        
        suite.add_result(result)
    
    def _run_single_test(self, test_instance, method_name: str, verbose: bool = True) -> TestResult:
        """Run a single test method"""
        
        start_time = datetime.now()
        
        try:
            # Run setup if available
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()
            
            # Run the test method
            test_method = getattr(test_instance, method_name)
            test_method()
            
            duration = (datetime.now() - start_time).total_seconds()
            result = TestResult(f"{test_instance.__class__.__name__}.{method_name}", True, "", duration)
            
            if verbose:
                print(f"  ✅ {method_name}")
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            result = TestResult(f"{test_instance.__class__.__name__}.{method_name}", False, error_msg, duration)
            
            if verbose:
                print(f"  ❌ {method_name}: {error_msg}")
        
        return result
    
    def _run_integration_tests(self, verbose: bool = True) -> TestSuite:
        """Run integration tests between components"""
        
        suite = TestSuite("Integration Tests", "Cross-component integration tests")
        suite.start_time = datetime.now()
        
        if verbose:
            print(f"\n🔗 Running Integration Tests")
        
        # Test 1: Shared utilities integration
        result = self._test_shared_utilities_integration(verbose)
        suite.add_result(result)
        
        # Test 2: Risk-Position management integration
        result = self._test_risk_position_integration(verbose)
        suite.add_result(result)
        
        # Test 3: Configuration consistency
        result = self._test_configuration_consistency(verbose)
        suite.add_result(result)
        
        # Test 4: Metrics calculation consistency
        result = self._test_metrics_consistency(verbose)
        suite.add_result(result)
        
        suite.end_time = datetime.now()
        return suite
    
    def _test_shared_utilities_integration(self, verbose: bool = True) -> TestResult:
        """Test shared utilities integration"""
        
        start_time = datetime.now()
        
        try:
            # Test logger integration
            from AuroraQ_Shared.utils.logger import get_logger
            test_logger = get_logger("IntegrationTest")
            assert test_logger is not None
            
            # Test config manager integration
            from AuroraQ_Shared.utils.config_manager import load_config
            config = load_config()
            assert config is not None
            
            # Test metrics integration
            from AuroraQ_Shared.utils.metrics import calculate_performance_metrics
            
            mock_trades = [
                {'action': 'close', 'pnl_pct': 0.02, 'timestamp': datetime.now()},
                {'action': 'close', 'pnl_pct': -0.01, 'timestamp': datetime.now()}
            ]
            metrics = calculate_performance_metrics(mock_trades)
            assert metrics is not None
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if verbose:
                print("  ✅ Shared utilities integration")
            
            return TestResult("shared_utilities_integration", True, "", duration)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Shared utilities integration failed: {str(e)}"
            
            if verbose:
                print(f"  ❌ Shared utilities integration: {error_msg}")
            
            return TestResult("shared_utilities_integration", False, error_msg, duration)
    
    def _test_risk_position_integration(self, verbose: bool = True) -> TestResult:
        """Test risk and position management integration"""
        
        start_time = datetime.now()
        
        try:
            # Test that risk and position management modules can be imported together
            from AuroraQ_Shared.risk_management.integrated_risk_manager import IntegratedRiskManager
            from AuroraQ_Shared.position_management.enhanced_position_manager import EnhancedPositionManager
            
            # Test basic initialization
            risk_manager = IntegratedRiskManager()
            position_manager = EnhancedPositionManager()
            
            assert risk_manager is not None
            assert position_manager is not None
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if verbose:
                print("  ✅ Risk-Position integration")
            
            return TestResult("risk_position_integration", True, "", duration)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Risk-Position integration failed: {str(e)}"
            
            if verbose:
                print(f"  ❌ Risk-Position integration: {error_msg}")
            
            return TestResult("risk_position_integration", False, error_msg, duration)
    
    def _test_configuration_consistency(self, verbose: bool = True) -> TestResult:
        """Test configuration consistency across components"""
        
        start_time = datetime.now()
        
        try:
            from AuroraQ_Shared.utils.config_manager import load_config
            
            # Test different component configurations
            shared_config = load_config(component_type="shared")
            production_config = load_config(component_type="production")
            backtest_config = load_config(component_type="backtest")
            
            # Verify basic structure
            for config in [shared_config, production_config, backtest_config]:
                assert hasattr(config, 'trading')
                assert hasattr(config, 'risk')
                assert hasattr(config, 'log_level')
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if verbose:
                print("  ✅ Configuration consistency")
            
            return TestResult("configuration_consistency", True, "", duration)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Configuration consistency failed: {str(e)}"
            
            if verbose:
                print(f"  ❌ Configuration consistency: {error_msg}")
            
            return TestResult("configuration_consistency", False, error_msg, duration)
    
    def _test_metrics_consistency(self, verbose: bool = True) -> TestResult:
        """Test metrics calculation consistency"""
        
        start_time = datetime.now()
        
        try:
            from AuroraQ_Shared.utils.metrics import calculate_performance_metrics, calculate_risk_metrics
            
            # Create consistent test data
            trades = [
                {'action': 'close', 'pnl_pct': 0.05, 'pnl': 2500, 'timestamp': datetime.now()},
                {'action': 'close', 'pnl_pct': -0.02, 'pnl': -1000, 'timestamp': datetime.now()},
                {'action': 'close', 'pnl_pct': 0.03, 'pnl': 1500, 'timestamp': datetime.now()}
            ]
            
            # Test performance metrics
            perf_metrics = calculate_performance_metrics(trades)
            assert perf_metrics.total_trades == 3
            assert perf_metrics.win_rate > 0
            
            # Test risk metrics
            risk_metrics = calculate_risk_metrics(trades)
            assert isinstance(risk_metrics, dict)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            if verbose:
                print("  ✅ Metrics consistency")
            
            return TestResult("metrics_consistency", True, "", duration)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Metrics consistency failed: {str(e)}"
            
            if verbose:
                print(f"  ❌ Metrics consistency: {error_msg}")
            
            return TestResult("metrics_consistency", False, error_msg, duration)
    
    def _generate_report(self, verbose: bool = True) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Aggregate statistics
        total_tests = sum(suite.total_tests for suite in self.test_suites)
        total_passed = sum(suite.passed_tests for suite in self.test_suites)
        total_failed = total_tests - total_passed
        overall_success_rate = total_passed / max(1, total_tests)
        
        # Suite summaries
        suite_summaries = []
        for suite in self.test_suites:
            summary = {
                'name': suite.name,
                'description': suite.description,
                'total_tests': suite.total_tests,
                'passed_tests': suite.passed_tests,
                'failed_tests': suite.failed_tests,
                'success_rate': suite.success_rate,
                'duration': suite.duration,
                'setup_error': suite.setup_error
            }
            suite_summaries.append(summary)
        
        # Failed test details
        failed_tests = []
        for suite in self.test_suites:
            for result in suite.results:
                if not result.success:
                    failed_tests.append({
                        'suite': suite.name,
                        'test': result.name,
                        'error': result.error,
                        'duration': result.duration
                    })
        
        report = {
            'timestamp': self.end_time.isoformat(),
            'duration': total_duration,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'success_rate': overall_success_rate,
                'suites_run': len(self.test_suites)
            },
            'suites': suite_summaries,
            'failed_tests': failed_tests
        }
        
        if verbose:
            self._print_summary_report(report)
        
        return report
    
    def _print_summary_report(self, report: Dict[str, Any]):
        """Print summary report to console"""
        
        print(f"\n{'='*60}")
        print(f"🧪 AuroraQ Unified Test Suite Results")
        print(f"{'='*60}")
        
        summary = report['summary']
        print(f"📊 Overall: {summary['passed_tests']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1%})")
        print(f"⏱️  Duration: {report['duration']:.2f} seconds")
        print(f"📦 Suites: {summary['suites_run']}")
        
        print(f"\n📋 Suite Results:")
        for suite in report['suites']:
            status = "✅" if suite['failed_tests'] == 0 else "❌"
            print(f"  {status} {suite['name']}: {suite['passed_tests']}/{suite['total_tests']} ({suite['success_rate']:.1%})")
            
            if suite['setup_error']:
                print(f"    ⚠️  Setup Error: {suite['setup_error']}")
        
        if report['failed_tests']:
            print(f"\n❌ Failed Tests ({len(report['failed_tests'])}):")
            for failed in report['failed_tests'][:10]:  # Show first 10 failures
                print(f"  • {failed['suite']}.{failed['test']}")
                print(f"    {failed['error']}")
            
            if len(report['failed_tests']) > 10:
                print(f"  ... and {len(report['failed_tests']) - 10} more")
        
        if summary['success_rate'] == 1.0:
            print(f"\n🎉 All tests passed!")
        elif summary['success_rate'] >= 0.8:
            print(f"\n✅ Most tests passed ({summary['success_rate']:.1%})")
        else:
            print(f"\n⚠️  Many tests failed ({summary['failed_tests']} failures)")

def main():
    """Main test runner entry point"""
    
    print("🚀 AuroraQ Unified Test Runner")
    print("Testing all components: Shared, Production, Backtest, Integration")
    print("-" * 60)
    
    runner = UnifiedTestRunner()
    
    try:
        # Run all tests
        report = runner.discover_and_run_tests(
            test_modules=None,  # Auto-discover
            include_integration=True,
            verbose=True
        )
        
        # Save detailed report
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Exit with appropriate code
        success_rate = report['summary']['success_rate']
        exit_code = 0 if success_rate == 1.0 else 1
        
        return exit_code
        
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)