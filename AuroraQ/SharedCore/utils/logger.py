# 📁 AuroraQ_Shared/utils/logger.py

import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

_loggers = {}

def get_logger(name: str, log_to_file: bool = True, log_level: str = "INFO") -> logging.Logger:
    """
    AuroraQ 시스템 통합 로거 - Production, Backtest, Shared 컴포넌트 지원
    
    Args:
        name (str): 로거 이름 (예: "RunLoop", "TrainPPO", "BacktestEngine")
        log_to_file (bool): 로그를 파일에도 저장할지 여부
        log_level (str): 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: 구성된 로거 객체
    """
    global _loggers
    
    # 로거 캐싱으로 중복 생성 방지
    logger_key = f"{name}_{log_level}_{log_to_file}"
    if logger_key in _loggers:
        return _loggers[logger_key]

    logger = logging.getLogger(name)
    
    # 로그 레벨 설정
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logger.setLevel(log_levels.get(log_level.upper(), logging.INFO))
    logger.propagate = False  # 루트로 로그가 중복 전달되는 것 방지

    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()

    # 포맷터 설정 - 통합된 형식
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택)
    if log_to_file:
        log_dir = Path("logs") / name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[logger_key] = logger
    return logger


class BacktestLogger:
    """백테스트 전용 로거 - 기존 AuroraQ_Backtest와 호환"""
    
    def __init__(self, 
                 name: str = "BacktestLogger",
                 log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 타임스탬프
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 로그 파일들
        self.main_log = self.log_dir / f"backtest_{timestamp}.log"
        self.trade_log = self.log_dir / f"trades_{timestamp}.log"
        self.error_log = self.log_dir / f"errors_{timestamp}.log"
        
        # 로거들 설정
        self.main_logger = get_logger(f"{name}.main", True, "INFO")
        self.trade_logger = get_logger(f"{name}.trade", True, "INFO")
        self.error_logger = get_logger(f"{name}.error", True, "ERROR")
    
    def info(self, message: str):
        """일반 정보 로그"""
        self.main_logger.info(message)
    
    def warning(self, message: str):
        """경고 로그"""
        self.main_logger.warning(message)
    
    def error(self, message: str):
        """에러 로그"""
        self.main_logger.error(message)
        self.error_logger.error(message)
    
    def trade(self, trade_info: dict):
        """거래 로그"""
        trade_msg = (
            f"Trade: {trade_info.get('side', 'N/A')} "
            f"{trade_info.get('size', 0):.6f} @ "
            f"{trade_info.get('price', 0):.2f} "
            f"(ID: {trade_info.get('trade_id', 'N/A')})"
        )
        self.trade_logger.info(trade_msg)
    
    def backtest_start(self, config: dict):
        """백테스트 시작 로그"""
        self.info("=" * 50)
        self.info("백테스트 시작")
        self.info(f"초기 자본: ${config.get('initial_capital', 0):,.0f}")
        self.info(f"수수료: {config.get('commission', 0):.3%}")
        self.info(f"슬리피지: {config.get('slippage', 0):.3%}")
        self.info("=" * 50)
    
    def backtest_end(self, results: dict):
        """백테스트 종료 로그"""
        self.info("=" * 50)
        self.info("백테스트 완료")
        self.info(f"최종 자본: ${results.get('final_capital', 0):,.0f}")
        self.info(f"총 수익률: {results.get('total_return', 0):.2%}")
        self.info(f"총 거래: {results.get('total_trades', 0)}")
        self.info(f"승률: {results.get('win_rate', 0):.1%}")
        self.info("=" * 50)