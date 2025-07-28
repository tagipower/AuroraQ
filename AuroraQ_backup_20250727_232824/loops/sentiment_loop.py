# sentiment/sentiment_loop.py

import os
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
from pathlib import Path
import json
import psutil
import traceback

from utils.logger import get_logger
from report.sentiment_report_generator import run_sentiment_report
from sentiment.sentiment_fusion_manager import SentimentFusionManager

# 로거 설정
logger = get_logger("SentimentLoop")

# 설정값들
DEFAULT_INTERVAL = 15 * 60  # 기본 15분
MIN_INTERVAL = 60  # 최소 1분
MAX_INTERVAL = 3600  # 최대 1시간
DEFAULT_CSV_PATH = "data/sentiment/sentiment_scores.csv"
BACKUP_CSV_PATH = "data/sentiment/sentiment_scores_backup.csv"
STATE_FILE = "data/sentiment/.loop_state.json"


class SentimentLoopRunner:
    """
    감정 점수 수집 및 통합을 주기적으로 실행하는 루프 매니저
    """
    
    def __init__(self, 
                 interval: int = DEFAULT_INTERVAL,
                 csv_path: str = DEFAULT_CSV_PATH,
                 enable_report: bool = True,
                 max_retries: int = 3):
        """
        :param interval: 실행 간격 (초)
        :param csv_path: 저장할 CSV 경로
        :param enable_report: 리포트 자동 생성 여부
        :param max_retries: 실패 시 최대 재시도 횟수
        """
        # 간격 검증
        self.interval = max(MIN_INTERVAL, min(interval, MAX_INTERVAL))
        if self.interval != interval:
            logger.warning(f"간격을 {self.interval}초로 조정했습니다 (요청: {interval}초)")
        
        self.csv_path = csv_path
        self.enable_report = enable_report
        self.max_retries = max_retries
        
        # 상태 관리
        self.running = False
        self.paused = False
        self.fusion_manager = None
        self.last_run_time = None
        self.run_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        
        # 스레드 안전성을 위한 락
        self.lock = threading.Lock()
        
        # 디렉토리 생성
        self._ensure_directories()
        
        # 상태 복원
        self._load_state()
        
        # 시그널 핸들러 설정
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"[SentimentLoop] 초기화 완료 - 간격: {self.interval}초, "
                   f"리포트: {'활성' if enable_report else '비활성'}")

    def _ensure_directories(self):
        """필요한 디렉토리 생성"""
        for path in [self.csv_path, BACKUP_CSV_PATH, STATE_FILE]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _signal_handler(self, signum, frame):
        """시그널 처리"""
        signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
        logger.warning(f"[SentimentLoop] 🛑 {signal_name} 시그널 수신. 안전하게 종료합니다...")
        self.stop()

    def _load_state(self):
        """이전 실행 상태 복원"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.last_run_time = datetime.fromisoformat(state.get('last_run_time', ''))
                    self.run_count = state.get('run_count', 0)
                    self.error_count = state.get('error_count', 0)
                    logger.info(f"[SentimentLoop] 이전 상태 복원 - 실행 횟수: {self.run_count}")
        except Exception as e:
            logger.debug(f"상태 파일 로드 실패 (무시): {e}")

    def _save_state(self):
        """현재 상태 저장"""
        try:
            state = {
                'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
                'run_count': self.run_count,
                'error_count': self.error_count,
                'interval': self.interval
            }
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")

    def _check_system_resources(self) -> bool:
        """시스템 리소스 확인"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90:
                logger.warning(f"CPU 사용률이 높습니다: {cpu_percent}%")
                return False
            
            if memory.percent > 90:
                logger.warning(f"메모리 사용률이 높습니다: {memory.percent}%")
                return False
            
            return True
        except:
            return True  # psutil 실패 시 계속 진행

    def _backup_csv(self):
        """CSV 파일 백업"""
        try:
            if os.path.exists(self.csv_path):
                import shutil
                shutil.copy2(self.csv_path, BACKUP_CSV_PATH)
                logger.debug("CSV 백업 완료")
        except Exception as e:
            logger.error(f"CSV 백업 실패: {e}")

    def _run_once(self) -> bool:
        """한 번의 수집 사이클 실행"""
        start_time = time.time()
        
        try:
            # 시스템 리소스 체크
            if not self._check_system_resources():
                logger.warning("시스템 리소스 부족으로 이번 사이클 건너뜀")
                return False
            
            # Fusion Manager 초기화 (필요시)
            if self.fusion_manager is None:
                self.fusion_manager = SentimentFusionManager()
                logger.info("[SentimentLoop] Fusion Manager 초기화 완료")
            
            # 1. 감정 점수 수집
            logger.info("[SentimentLoop] 📊 감정 점수 수집 시작...")
            fused_scores = self.fusion_manager.get_fused_scores()
            
            if not fused_scores:
                logger.warning("[SentimentLoop] 수집된 감정 점수가 없습니다")
                return False
            
            logger.info(f"[SentimentLoop] ✅ {len(fused_scores)}개 기사의 통합 점수 수집 완료")
            
            # 2. 데이터 검증
            valid_scores = self._validate_scores(fused_scores)
            if len(valid_scores) < len(fused_scores):
                logger.warning(f"{len(fused_scores) - len(valid_scores)}개의 무효한 점수 제거됨")
            
            # 3. CSV 저장
            self._save_to_csv(valid_scores)
            
            # 4. 리포트 생성 (옵션)
            if self.enable_report:
                try:
                    logger.info("[SentimentLoop] 📄 리포트 생성 시작...")
                    run_sentiment_report(filepath=self.csv_path)
                    logger.info("[SentimentLoop] 📄 리포트 생성 완료")
                except Exception as e:
                    logger.error(f"리포트 생성 실패 (계속 진행): {e}")
            
            # 통계 업데이트
            elapsed_time = time.time() - start_time
            self.run_count += 1
            self.last_run_time = datetime.now()
            self.consecutive_errors = 0  # 성공 시 연속 에러 카운트 리셋
            
            logger.info(f"[SentimentLoop] ✨ 사이클 완료 - "
                       f"소요시간: {elapsed_time:.1f}초, "
                       f"총 실행: {self.run_count}회")
            
            # 상태 저장
            self._save_state()
            
            return True
            
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            logger.error(f"[SentimentLoop] ❌ 실행 중 오류 발생: {e}")
            logger.debug(traceback.format_exc())
            
            # 연속 에러가 많으면 간격 늘리기
            if self.consecutive_errors >= self.max_retries:
                self.interval = min(self.interval * 1.5, MAX_INTERVAL)
                logger.warning(f"연속 {self.consecutive_errors}회 실패. "
                             f"간격을 {self.interval}초로 조정")
            
            return False

    def _validate_scores(self, scores: List[Dict]) -> List[Dict]:
        """점수 데이터 검증"""
        valid_scores = []
        
        for score in scores:
            # 필수 필드 확인
            if not all(key in score for key in ['date', 'sentiment_score']):
                continue
            
            # 점수 범위 확인
            if not (0 <= score['sentiment_score'] <= 1):
                logger.warning(f"비정상 점수 발견: {score['sentiment_score']}")
                continue
            
            # 날짜 검증
            try:
                pd.to_datetime(score['date'])
            except:
                logger.warning(f"잘못된 날짜 형식: {score['date']}")
                continue
            
            valid_scores.append(score)
        
        return valid_scores

    def _save_to_csv(self, scores: List[Dict]):
        """CSV 파일로 저장"""
        try:
            # 백업 생성
            self._backup_csv()
            
            # DataFrame 생성
            df = pd.DataFrame(scores)
            
            # 기존 데이터와 병합 (중복 제거)
            if os.path.exists(self.csv_path):
                existing_df = pd.read_csv(self.csv_path)
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                df['date'] = pd.to_datetime(df['date'])
                
                # 날짜 기준으로 중복 제거 (최신 데이터 우선)
                combined_df = pd.concat([df, existing_df])
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
                combined_df = combined_df.sort_values('date', ascending=False)
                
                # 최대 보관 기간 적용 (예: 30일)
                cutoff_date = datetime.now() - timedelta(days=30)
                combined_df = combined_df[combined_df['date'] >= cutoff_date]
                
                df = combined_df
            
            # 저장
            df.to_csv(self.csv_path, index=False)
            logger.info(f"[SentimentLoop] 💾 CSV 저장 완료 - {len(df)}개 레코드")
            
        except Exception as e:
            logger.error(f"CSV 저장 실패: {e}")
            raise

    def run(self):
        """메인 루프 실행"""
        self.running = True
        logger.info(f"[SentimentLoop] 🚀 감정 점수 수집 루프 시작 (간격: {self.interval}초)")
        
        while self.running:
            try:
                # 일시정지 상태 확인
                if self.paused:
                    logger.debug("일시정지 상태...")
                    time.sleep(5)
                    continue
                
                # 다음 실행까지 대기할 시간 계산
                if self.last_run_time:
                    next_run = self.last_run_time + timedelta(seconds=self.interval)
                    wait_seconds = (next_run - datetime.now()).total_seconds()
                    
                    if wait_seconds > 0:
                        logger.info(f"[SentimentLoop] ⏱️ 다음 실행까지 {wait_seconds:.0f}초 대기")
                        time.sleep(min(wait_seconds, 60))  # 최대 60초씩 대기
                        continue
                
                # 실행
                with self.lock:
                    success = self._run_once()
                
                # 다음 실행 시간 로깅
                next_run_time = datetime.now() + timedelta(seconds=self.interval)
                logger.info(f"[SentimentLoop] 📅 다음 실행 예정: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                logger.error(f"[SentimentLoop] 루프 오류: {e}")
                time.sleep(10)  # 오류 시 짧은 대기
        
        logger.info("[SentimentLoop] 🛑 루프 종료됨")

    def stop(self):
        """루프 중지"""
        logger.info("[SentimentLoop] 종료 요청됨...")
        self.running = False
        self._save_state()

    def pause(self):
        """루프 일시정지"""
        self.paused = True
        logger.info("[SentimentLoop] ⏸️ 일시정지됨")

    def resume(self):
        """루프 재개"""
        self.paused = False
        logger.info("[SentimentLoop] ▶️ 재개됨")

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'running': self.running,
            'paused': self.paused,
            'interval': self.interval,
            'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None,
            'run_count': self.run_count,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors
        }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentiment Score Collection Loop')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                       help=f'실행 간격 (초), 기본값: {DEFAULT_INTERVAL}')
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH,
                       help=f'CSV 저장 경로, 기본값: {DEFAULT_CSV_PATH}')
    parser.add_argument('--no-report', action='store_true',
                       help='리포트 생성 비활성화')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 모드')
    
    args = parser.parse_args()
    
    # 디버그 모드 설정
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 루프 실행
    try:
        loop = SentimentLoopRunner(
            interval=args.interval,
            csv_path=args.csv_path,
            enable_report=not args.no_report
        )
        loop.run()
    except KeyboardInterrupt:
        logger.info("키보드 인터럽트 감지")
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        raise


if __name__ == "__main__":
    main()