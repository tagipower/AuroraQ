# 📁 utils/logger.py

import logging
import os
from datetime import datetime

_loggers = {}

def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    AuroraQ 시스템 전용 로거.
    모듈 이름별로 로거를 구분하고, 콘솔 + 파일 로깅 지원.

    Args:
        name (str): 로거 이름 (예: "RunLoop", "TrainPPO")
        log_to_file (bool): 로그를 파일에도 저장할지 여부

    Returns:
        logging.Logger: 구성된 로거 객체
    """
    global _loggers
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 루트로 로그가 중복 전달되는 것 방지

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택)
    if log_to_file:
        log_dir = os.path.join("logs", name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
