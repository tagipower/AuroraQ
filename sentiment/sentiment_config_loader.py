# sentiment/sentiment_config_loader.py - watchdog 의존성 해결 최종 버전

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from watchdog.observers import Observer
from dataclasses import dataclass, field
from functools import lru_cache
import copy
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class SentimentConfig:
    """감정 분석 설정 데이터 모델"""
    # 기본 설정
    mode: str = "live"
    
    # 데이터 수집 설정
    collector: Dict[str, Any] = field(default_factory=dict)
    
    # 분석 설정
    analyzer: Dict[str, Any] = field(default_factory=dict)
    
    # 저장 설정
    storage: Dict[str, Any] = field(default_factory=dict)
    
    # 소스 가중치
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "news": 0.4,
        "social": 0.3,
        "technical": 0.2,
        "historical": 0.1
    })
    
    # 임계값 설정
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "sentiment_filter": 0.2,
        "confidence_min": 0.6,
        "volatility_max": 0.5
    })
    
    # API 설정
    api_settings: Dict[str, Any] = field(default_factory=dict)
    
    # 기타 설정
    cache_ttl: int = 300
    enable_async: bool = True
    debug_mode: bool = False
    
    def validate(self) -> List[str]:
        """설정 유효성 검증"""
        errors = []
        
        # 모드 검증
        if self.mode not in ["live", "backtest", "hybrid"]:
            errors.append(f"Invalid mode: {self.mode}")
        
        # 가중치 합 검증
        weight_sum = sum(self.source_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(f"Source weights sum to {weight_sum}, should be 1.0")
        
        # 임계값 범위 검증
        for key, value in self.thresholds.items():
            if not 0 <= value <= 1:
                errors.append(f"Threshold {key} out of range: {value}")
        
        return errors


class ConfigLoader:
    """설정 파일 로더 (watchdog 선택적 의존성)"""
    
    # 지원하는 파일 형식
    SUPPORTED_FORMATS = {'.yaml', '.yml', '.json'}
    
    # 기본 설정
    DEFAULT_CONFIG = {
        "sentiment": {
            "mode": "live",
            "collector": {
                "output_dir": "data/sentiment",
                "batch_size": 10,
                "max_articles_per_stream": 50
            },
            "analyzer": {
                "model": "ProsusAI/finbert",
                "max_length": 512,
                "device": "auto"
            },
            "storage": {
                "format": "csv",
                "compression": None,
                "retention_days": 30
            },
            "source_weights": {
                "news": 0.4,
                "social": 0.3,
                "technical": 0.2,
                "historical": 0.1
            },
            "thresholds": {
                "sentiment_filter": 0.2,
                "confidence_min": 0.6,
                "volatility_max": 0.5
            }
        }
    }
    
    def __init__(self, 
                 config_dir: Union[str, Path] = "config",
                 env_prefix: str = "SENTIMENT_"):
        """
        Args:
            config_dir: 설정 파일 디렉토리
            env_prefix: 환경변수 prefix
        """
        self.config_dir = Path(config_dir)
        self.env_prefix = env_prefix
        self._config_cache = {}
        self._file_watchers = {}
        self._watchdog_available = self._check_watchdog_availability()
        
    def _check_watchdog_availability(self) -> bool:
        """watchdog 라이브러리 사용 가능 여부 확인"""
        try:
            import watchdog
            return True
        except ImportError:
            logger.info(
                "watchdog 라이브러리가 설치되지 않았습니다. "
                "파일 감시는 폴링 방식으로 작동합니다. "
                "더 효율적인 감시를 원하시면 'pip install watchdog'를 실행하세요."
            )
            return False
        
    def load_sentiment_config(self, 
                            config_path: Optional[Union[str, Path]] = None,
                            use_env_override: bool = True,
                            validate: bool = True) -> SentimentConfig:
        """
        감정 분석 설정 로드
        
        Args:
            config_path: 설정 파일 경로
            use_env_override: 환경변수 오버라이드 사용 여부
            validate: 유효성 검증 여부
            
        Returns:
            SentimentConfig 객체
        """
        # 기본 경로 설정
        if config_path is None:
            config_path = self._find_config_file()
        else:
            config_path = Path(config_path)
        
        # 캐시 확인
        cache_key = str(config_path)
        if cache_key in self._config_cache:
            logger.debug(f"Using cached config from {config_path}")
            return copy.deepcopy(self._config_cache[cache_key])
        
        try:
            # 설정 로드
            config_dict = self._load_config_file(config_path)
            
            # 기본값과 병합
            merged_config = self._merge_with_defaults(config_dict)
            
            # 환경변수 오버라이드
            if use_env_override:
                merged_config = self._apply_env_overrides(merged_config)
            
            # Include 처리
            merged_config = self._process_includes(merged_config)
            
            # SentimentConfig 객체 생성
            sentiment_config = self._dict_to_config(merged_config.get("sentiment", {}))
            
            # 유효성 검증
            if validate:
                errors = sentiment_config.validate()
                if errors:
                    raise ValueError(f"Config validation failed: {errors}")
            
            # 캐시 저장
            self._config_cache[cache_key] = sentiment_config
            
            logger.info(f"Loaded config from {config_path}")
            return sentiment_config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _find_config_file(self) -> Path:
        """설정 파일 자동 탐색"""
        search_paths = [
            self.config_dir / "sentiment_config.yaml",
            self.config_dir / "sentiment_config.yml",
            self.config_dir / "sentiment_config.json",
            self.config_dir / "config.yaml",
            self.config_dir / "config.yml",
            self.config_dir / "config.json"
        ]
        
        for path in search_paths:
            if path.exists():
                logger.debug(f"Found config file: {path}")
                return path
        
        # 설정 파일이 없으면 기본 설정 파일 생성
        default_path = self.config_dir / "sentiment_config.yaml"
        logger.warning(
            f"No config file found. Creating default config at {default_path}"
        )
        self._create_default_config(default_path)
        return default_path
    
    def _create_default_config(self, path: Path):
        """기본 설정 파일 생성"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.DEFAULT_CONFIG, f, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """설정 파일 로드"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        suffix = config_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported config format: {suffix}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in {'.yaml', '.yml'}:
                return yaml.safe_load(f) or {}
            elif suffix == '.json':
                return json.load(f)
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """기본 설정과 병합"""
        return self._deep_merge(copy.deepcopy(self.DEFAULT_CONFIG), config)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """딕셔너리 깊은 병합"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """환경변수 오버라이드 적용"""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # 환경변수 키를 설정 경로로 변환
                # SENTIMENT_MODE -> sentiment.mode
                # SENTIMENT_COLLECTOR_BATCH_SIZE -> sentiment.collector.batch_size
                config_path = key[len(self.env_prefix):].lower().split('_')
                
                # 값 타입 변환
                typed_value = self._parse_env_value(value)
                
                # 설정에 적용
                self._set_nested_dict(config, config_path, typed_value)
                logger.debug(f"Applied env override: {key} = {typed_value}")
        
        return config
    
    def _parse_env_value(self, value: str) -> Any:
        """환경변수 값 타입 변환"""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except:
                pass
        
        # String
        return value
    
    def _set_nested_dict(self, d: Dict, path: List[str], value: Any):
        """중첩 딕셔너리에 값 설정"""
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value
    
    def _process_includes(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """include 파일 처리"""
        if '_include' in config:
            includes = config.pop('_include')
            if isinstance(includes, str):
                includes = [includes]
            
            for include_path in includes:
                include_full_path = self.config_dir / include_path
                if include_full_path.exists():
                    include_config = self._load_config_file(include_full_path)
                    config = self._deep_merge(include_config, config)
                else:
                    logger.warning(f"Include file not found: {include_full_path}")
        
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SentimentConfig:
        """딕셔너리를 SentimentConfig 객체로 변환"""
        # 기본값과 병합
        default_sentiment = self.DEFAULT_CONFIG["sentiment"]
        merged = self._deep_merge(default_sentiment, config_dict)
        
        return SentimentConfig(
            mode=merged.get("mode", "live"),
            collector=merged.get("collector", {}),
            analyzer=merged.get("analyzer", {}),
            storage=merged.get("storage", {}),
            source_weights=merged.get("source_weights", {}),
            thresholds=merged.get("thresholds", {}),
            api_settings=merged.get("api_settings", {}),
            cache_ttl=merged.get("cache_ttl", 300),
            enable_async=merged.get("enable_async", True),
            debug_mode=merged.get("debug_mode", False)
        )
    
    @lru_cache(maxsize=32)
    def load_stream_ids(self, stream_file: str = "stream_ids.yaml") -> List[str]:
        """Stream ID 목록 로드 (캐싱)"""
        stream_path = self.config_dir / stream_file
        
        if not stream_path.exists():
            logger.warning(f"Stream ID file not found: {stream_path}")
            return []
        
        try:
            with open(stream_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                streams = data.get("streams", data.get("stream_ids", []))
                
            logger.info(f"Loaded {len(streams)} stream IDs")
            return streams
            
        except Exception as e:
            logger.error(f"Failed to load stream IDs: {e}")
            return []
    
    def save_config(self, config: SentimentConfig, 
                   output_path: Union[str, Path],
                   format: str = "yaml"):
        """설정 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Config를 딕셔너리로 변환
        config_dict = {
            "sentiment": {
                "mode": config.mode,
                "collector": config.collector,
                "analyzer": config.analyzer,
                "storage": config.storage,
                "source_weights": config.source_weights,
                "thresholds": config.thresholds,
                "api_settings": config.api_settings,
                "cache_ttl": config.cache_ttl,
                "enable_async": config.enable_async,
                "debug_mode": config.debug_mode
            },
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            elif format == "json":
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Config saved to {output_path}")
    
    def watch_config(self, config_path: Union[str, Path], 
                    callback: callable,
                    interval: int = 5) -> Optional[Union[threading.Thread, Any]]:
        """
        설정 파일 변경 감시
        
        Args:
            config_path: 감시할 설정 파일 경로
            callback: 변경 시 호출할 콜백 함수
            interval: 폴링 간격 (초, watchdog 없을 때만 사용)
            
        Returns:
            감시 객체 (Thread 또는 Observer) 또는 None
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.error(f"Cannot watch non-existent file: {config_path}")
            return None
        
        # watchdog 사용 가능한 경우
        if self._watchdog_available:
            try:
                return self._watch_with_watchdog(config_path, callback)
            except Exception as e:
                logger.warning(f"Watchdog failed, falling back to polling: {e}")
                self._watchdog_available = False
        
        # 폴링 방식 사용
        return self._watch_with_polling(config_path, callback, interval)
    
    def _watch_with_watchdog(self, config_path: Path, callback: callable):
        """watchdog을 사용한 파일 감시"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class ConfigChangeHandler(FileSystemEventHandler):
            def __init__(self, loader, path, cb):
                self.loader = loader
                self.path = Path(path)
                self.callback = cb
                self.last_modified = 0
                
            def on_modified(self, event):
                if event.is_directory:
                    return
                    
                event_path = Path(event.src_path)
                if event_path.name == self.path.name:
                    # 중복 이벤트 방지
                    current_time = time.time()
                    if current_time - self.last_modified > 1:
                        self.last_modified = current_time
                        logger.info(f"Config file changed: {self.path}")
                        self._reload_config()
            
            def _reload_config(self):
                try:
                    # 캐시 무효화
                    cache_key = str(self.path)
                    if cache_key in self.loader._config_cache:
                        del self.loader._config_cache[cache_key]
                    
                    # 설정 다시 로드
                    new_config = self.loader.load_sentiment_config(self.path)
                    self.callback(new_config)
                except Exception as e:
                    logger.error(f"Failed to reload config: {e}")
        
        handler = ConfigChangeHandler(self, config_path, callback)
        observer = Observer()
        observer.schedule(handler, str(config_path.parent), recursive=False)
        observer.start()
        
        logger.info(f"Watching config file with watchdog: {config_path}")
        return observer
    
    def _watch_with_polling(self, config_path: Path, callback: callable, interval: int):
        """폴링을 사용한 파일 감시"""
        def poll_file():
            last_mtime = config_path.stat().st_mtime
            
            while config_path in self._file_watchers:
                try:
                    current_mtime = config_path.stat().st_mtime
                    if current_mtime > last_mtime:
                        last_mtime = current_mtime
                        logger.info(f"Config file changed: {config_path}")
                        
                        # 파일이 완전히 쓰여질 때까지 잠시 대기
                        time.sleep(0.1)
                        
                        try:
                            # 캐시 무효화
                            cache_key = str(config_path)
                            if cache_key in self._config_cache:
                                del self._config_cache[cache_key]
                            
                            # 설정 다시 로드
                            new_config = self.load_sentiment_config(config_path)
                            callback(new_config)
                        except Exception as e:
                            logger.error(f"Failed to reload config: {e}")
                            
                except FileNotFoundError:
                    logger.warning(f"Config file deleted: {config_path}")
                    break
                except Exception as e:
                    logger.error(f"Error checking file: {e}")
                
                time.sleep(interval)
            
            logger.info(f"Stopped watching: {config_path}")
        
        # 이미 감시 중인지 확인
        if config_path in self._file_watchers:
            logger.warning(f"Already watching {config_path}")
            return self._file_watchers[config_path]
        
        # 새 스레드에서 감시 시작
        watcher_thread = threading.Thread(
            target=poll_file, 
            name=f"ConfigWatcher-{config_path.name}",
            daemon=True
        )
        watcher_thread.start()
        self._file_watchers[config_path] = watcher_thread
        
        logger.info(f"Watching config file with polling (interval={interval}s): {config_path}")
        return watcher_thread
    
    def stop_watching(self, config_path: Union[str, Path]):
        """파일 감시 중지"""
        config_path = Path(config_path)
        if config_path in self._file_watchers:
            del self._file_watchers[config_path]
            logger.info(f"Stopped watching: {config_path}")
    
    def stop_all_watchers(self):
        """모든 파일 감시 중지"""
        paths = list(self._file_watchers.keys())
        for path in paths:
            self.stop_watching(path)
        logger.info("Stopped all file watchers")


# 싱글톤 인스턴스 제공
_default_loader = None

def get_config_loader() -> ConfigLoader:
    """기본 ConfigLoader 인스턴스 반환"""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader

def load_sentiment_config(config_path: Optional[str] = None, **kwargs) -> SentimentConfig:
    """간편 로드 함수 (기존 인터페이스 호환)"""
    loader = get_config_loader()
    return loader.load_sentiment_config(config_path, **kwargs)


# 테스트 코드
if __name__ == "__main__":
    import tempfile
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트용 설정 파일 생성
    test_config = """
sentiment:
  mode: backtest
  collector:
    output_dir: data/test_sentiment
    batch_size: 20
  analyzer:
    model: ProsusAI/finbert
    device: cpu
  source_weights:
    news: 0.5
    social: 0.3
    technical: 0.2
  thresholds:
    sentiment_filter: 0.3
    confidence_min: 0.7
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(test_config)
        test_file = f.name
    
    try:
        # 1. 설정 로드 테스트
        print("=== 설정 로드 테스트 ===")
        loader = ConfigLoader()
        config = loader.load_sentiment_config(test_file)
        
        print(f"Mode: {config.mode}")
        print(f"Batch size: {config.collector.get('batch_size')}")
        print(f"Source weights: {config.source_weights}")
        print(f"Validation errors: {config.validate()}")
        
        # 2. 환경변수 오버라이드 테스트
        print("\n=== 환경변수 오버라이드 테스트 ===")
        os.environ['SENTIMENT_MODE'] = 'live'
        os.environ['SENTIMENT_COLLECTOR_BATCH_SIZE'] = '50'
        
        config2 = loader.load_sentiment_config(test_file)
        print(f"Mode after override: {config2.mode}")
        print(f"Batch size after override: {config2.collector.get('batch_size')}")
        
        # 3. 설정 저장 테스트
        print("\n=== 설정 저장 테스트 ===")
        output_path = Path(tempfile.gettempdir()) / "saved_config.yaml"
        loader.save_config(config2, output_path)
        print(f"Config saved to: {output_path}")
        
        # 저장된 설정 다시 로드
        config3 = loader.load_sentiment_config(output_path)
        print(f"Reloaded mode: {config3.mode}")
        
        # 4. 파일 감시 테스트
        print("\n=== 파일 감시 테스트 ===")
        print(f"Watchdog available: {loader._watchdog_available}")
        
        changes_detected = []
        def on_config_change(new_config):
            changes_detected.append(new_config.mode)
            print(f"✓ Config changed! New mode: {new_config.mode}")
        
        watcher = loader.watch_config(test_file, on_config_change, interval=1)
        print("Watching for 3 seconds...")
        
        # 파일 수정 시뮬레이션
        time.sleep(1)
        with open(test_file, 'w') as f:
            f.write(test_config.replace('backtest', 'live'))
        
        time.sleep(2)
        
        # 감시 중지
        loader.stop_watching(test_file)
        
        print(f"Changes detected: {changes_detected}")
        
    finally:
        # 정리
        os.unlink(test_file)
        if 'output_path' in locals() and output_path.exists():
            os.unlink(output_path)
        
        # 환경변수 정리
        for key in ['SENTIMENT_MODE', 'SENTIMENT_COLLECTOR_BATCH_SIZE']:
            if key in os.environ:
                del os.environ[key]
                
        print("\n✅ 테스트 완료 및 정리됨")