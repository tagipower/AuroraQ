# utils/logging_config.py
"""Structured logging configuration for sentiment service"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

import structlog
from structlog.stdlib import LoggerFactory
from structlog.typing import Processor

from config.settings import settings


def setup_logging() -> None:
    """Configure structured logging for the application"""
    
    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structlog processors
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.log_format == "json":
        # JSON formatting for production
        processors.append(
            structlog.processors.JSONRenderer(serializer=structlog.processors.json_fallback_handler)
        )
    else:
        # Pretty formatting for development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Configure file handler if specified
    if settings.log_file_path:
        file_handler = logging.FileHandler(settings.log_file_path)
        file_handler.setLevel(getattr(logging, settings.log_level.upper()))
        
        if settings.log_format == "json":
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Set log levels for external libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # Suppress noisy logs in production
    if not settings.debug:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("redis").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class LoggingMixin:
    """Mixin class to add structured logging to other classes"""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)


def log_function_call(func_name: str, **kwargs) -> Dict[str, Any]:
    """Create standardized log context for function calls"""
    return {
        "function": func_name,
        "args": {k: str(v)[:100] for k, v in kwargs.items()}  # Truncate long values
    }


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    **extra_context
) -> Dict[str, Any]:
    """Create standardized log context for API requests"""
    return {
        "request_method": method,
        "request_path": path,
        "response_status": status_code,
        "response_time": round(duration, 4),
        **extra_context
    }


def log_model_performance(
    model_name: str,
    operation: str,
    duration: float,
    input_size: int = None,
    **metrics
) -> Dict[str, Any]:
    """Create standardized log context for model performance"""
    context = {
        "model": model_name,
        "operation": operation,
        "duration": round(duration, 4),
    }
    
    if input_size is not None:
        context["input_size"] = input_size
    
    # Add performance metrics
    context.update({
        k: round(v, 4) if isinstance(v, float) else v
        for k, v in metrics.items()
    })
    
    return context