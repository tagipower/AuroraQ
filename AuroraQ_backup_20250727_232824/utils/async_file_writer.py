import asyncio
import json
import yaml
import csv
import os
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import logging


class AsyncBatchFileWriter:
    """
    비동기 배치 파일 쓰기 시스템
    - 배치 처리로 I/O 횟수 감소
    - 백그라운드 스레드에서 파일 쓰기
    - 다양한 파일 형식 지원 (JSON, YAML, CSV)
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_workers: int = 3):
        """
        Args:
            batch_size: 배치 크기
            flush_interval: 자동 플러시 간격 (초)
            max_workers: 워커 스레드 수
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # 파일별 쓰기 큐
        self.write_queues = {}
        self.file_locks = {}
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 플러시 타이머
        self.flush_timer = None
        self.running = True
        
        # 통계
        self.stats = {
            "writes_queued": 0,
            "writes_completed": 0,
            "batches_processed": 0,
            "errors": 0
        }
        
        # 로거
        self.logger = logging.getLogger("AsyncFileWriter")
        
        # 자동 플러시 시작
        self._start_auto_flush()
    
    def write_json(self, file_path: str, data: Union[Dict, List], mode: str = 'w'):
        """JSON 파일 쓰기 요청"""
        self._queue_write(file_path, data, 'json', mode)
    
    def write_yaml(self, file_path: str, data: Dict[str, Any], mode: str = 'w'):
        """YAML 파일 쓰기 요청"""
        self._queue_write(file_path, data, 'yaml', mode)
    
    def append_csv(self, file_path: str, rows: List[List], headers: Optional[List[str]] = None):
        """CSV 파일에 행 추가"""
        self._queue_write(file_path, {'rows': rows, 'headers': headers}, 'csv', 'a')
    
    def write_log(self, file_path: str, message: str):
        """로그 메시지 쓰기"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self._queue_write(file_path, log_entry, 'text', 'a')
    
    def _queue_write(self, file_path: str, data: Any, file_type: str, mode: str):
        """쓰기 작업을 큐에 추가"""
        # 파일별 큐 초기화
        if file_path not in self.write_queues:
            self.write_queues[file_path] = deque()
            self.file_locks[file_path] = threading.Lock()
        
        # 큐에 추가
        with self.file_locks[file_path]:
            self.write_queues[file_path].append({
                'data': data,
                'type': file_type,
                'mode': mode,
                'timestamp': datetime.now()
            })
            self.stats["writes_queued"] += 1
        
        # 배치 크기 도달시 즉시 플러시
        if len(self.write_queues[file_path]) >= self.batch_size:
            self.executor.submit(self._flush_file, file_path)
    
    def _flush_file(self, file_path: str):
        """특정 파일의 큐를 플러시"""
        if file_path not in self.write_queues:
            return
        
        with self.file_locks[file_path]:
            if not self.write_queues[file_path]:
                return
            
            # 큐 복사 후 초기화
            batch = list(self.write_queues[file_path])
            self.write_queues[file_path].clear()
        
        # 배치 처리
        try:
            self._process_batch(file_path, batch)
            self.stats["batches_processed"] += 1
        except Exception as e:
            self.logger.error(f"배치 처리 실패 ({file_path}): {e}")
            self.stats["errors"] += 1
    
    def _process_batch(self, file_path: str, batch: List[Dict[str, Any]]):
        """배치 처리 및 파일 쓰기"""
        # 디렉토리 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 파일 타입별 처리
        file_type = batch[0]['type']
        
        if file_type == 'json':
            self._write_json_batch(file_path, batch)
        elif file_type == 'yaml':
            self._write_yaml_batch(file_path, batch)
        elif file_type == 'csv':
            self._write_csv_batch(file_path, batch)
        elif file_type == 'text':
            self._write_text_batch(file_path, batch)
        
        self.stats["writes_completed"] += len(batch)
    
    def _write_json_batch(self, file_path: str, batch: List[Dict[str, Any]]):
        """JSON 배치 쓰기"""
        # 마지막 쓰기 모드 사용
        mode = batch[-1]['mode']
        
        if mode == 'a':
            # 기존 데이터 로드
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                except:
                    existing_data = []
            
            # 새 데이터 추가
            for item in batch:
                data = item['data']
                if isinstance(data, list):
                    existing_data.extend(data)
                else:
                    existing_data.append(data)
            
            # 전체 쓰기
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        else:
            # 마지막 데이터만 쓰기
            with open(file_path, 'w') as f:
                json.dump(batch[-1]['data'], f, indent=2)
    
    def _write_yaml_batch(self, file_path: str, batch: List[Dict[str, Any]]):
        """YAML 배치 쓰기"""
        # 마지막 데이터만 쓰기 (YAML은 덮어쓰기가 일반적)
        with open(file_path, 'w') as f:
            yaml.dump(batch[-1]['data'], f, default_flow_style=False)
    
    def _write_csv_batch(self, file_path: str, batch: List[Dict[str, Any]]):
        """CSV 배치 쓰기"""
        file_exists = os.path.exists(file_path)
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for item in batch:
                data = item['data']
                
                # 헤더 쓰기 (파일이 없고 헤더가 제공된 경우)
                if not file_exists and data.get('headers'):
                    writer.writerow(data['headers'])
                    file_exists = True
                
                # 데이터 행 쓰기
                if 'rows' in data:
                    writer.writerows(data['rows'])
    
    def _write_text_batch(self, file_path: str, batch: List[Dict[str, Any]]):
        """텍스트 배치 쓰기"""
        with open(file_path, 'a') as f:
            for item in batch:
                f.write(item['data'])
    
    def _start_auto_flush(self):
        """자동 플러시 타이머 시작"""
        if self.running:
            self.flush_timer = threading.Timer(
                self.flush_interval, 
                self._auto_flush
            )
            self.flush_timer.daemon = True
            self.flush_timer.start()
    
    def _auto_flush(self):
        """모든 큐 자동 플러시"""
        for file_path in list(self.write_queues.keys()):
            if self.write_queues[file_path]:
                self.executor.submit(self._flush_file, file_path)
        
        # 다음 플러시 예약
        self._start_auto_flush()
    
    def flush_all(self):
        """모든 큐 즉시 플러시"""
        futures = []
        for file_path in list(self.write_queues.keys()):
            if self.write_queues[file_path]:
                future = self.executor.submit(self._flush_file, file_path)
                futures.append(future)
        
        # 모든 플러시 완료 대기
        for future in futures:
            future.result()
    
    def get_stats(self) -> Dict[str, int]:
        """통계 반환"""
        pending = sum(len(queue) for queue in self.write_queues.values())
        return {
            **self.stats,
            "pending_writes": pending,
            "active_files": len(self.write_queues)
        }
    
    def shutdown(self):
        """시스템 종료"""
        self.running = False
        
        # 타이머 취소
        if self.flush_timer:
            self.flush_timer.cancel()
        
        # 모든 큐 플러시
        self.flush_all()
        
        # 스레드 풀 종료
        self.executor.shutdown(wait=True)


# 전역 인스턴스
_async_writer = AsyncBatchFileWriter()


def get_async_writer() -> AsyncBatchFileWriter:
    """전역 비동기 파일 쓰기 인스턴스 반환"""
    return _async_writer


# 편의 함수들
def write_json_async(file_path: str, data: Union[Dict, List], mode: str = 'w'):
    """비동기 JSON 쓰기"""
    _async_writer.write_json(file_path, data, mode)


def write_yaml_async(file_path: str, data: Dict[str, Any], mode: str = 'w'):
    """비동기 YAML 쓰기"""
    _async_writer.write_yaml(file_path, data, mode)


def append_csv_async(file_path: str, rows: List[List], headers: Optional[List[str]] = None):
    """비동기 CSV 추가"""
    _async_writer.append_csv(file_path, rows, headers)


def write_log_async(file_path: str, message: str):
    """비동기 로그 쓰기"""
    _async_writer.write_log(file_path, message)