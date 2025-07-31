#!/usr/bin/env python3
"""
Enhanced Terminal Dashboard Formatter
개선된 터미널 대시보드 포매터 - 에러 핸들링, 색상 테마, 캐싱, 실제 데이터 연동
"""

from wcwidth import wcswidth
import re
from typing import Optional, List, Tuple, Dict, Any, Callable
from functools import lru_cache
from enum import Enum
from datetime import datetime
import json
import sys
import os

class ColorTheme(Enum):
    """색상 테마 정의"""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    CYBERPUNK = "cyberpunk"
    MINIMAL = "minimal"

class Colors:
    """ANSI 색상 코드 관리"""
    
    THEMES = {
        ColorTheme.DEFAULT: {
            'success': '\033[32m',    # Green
            'warning': '\033[33m',    # Yellow
            'error': '\033[31m',      # Red
            'info': '\033[36m',       # Cyan
            'header': '\033[1;34m',   # Bold Blue
            'border': '\033[37m',     # White
            'value': '\033[37m',      # White
            'label': '\033[90m',      # Dark Gray
            'reset': '\033[0m'
        },
        ColorTheme.DARK: {
            'success': '\033[1;32m',  # Bright Green
            'warning': '\033[1;33m',  # Bright Yellow 
            'error': '\033[1;31m',    # Bright Red
            'info': '\033[1;36m',     # Bright Cyan
            'header': '\033[1;35m',   # Bright Magenta
            'border': '\033[1;37m',   # Bright White
            'value': '\033[37m',      # White
            'label': '\033[1;90m',    # Bright Black
            'reset': '\033[0m'
        },
        ColorTheme.LIGHT: {
            'success': '\033[92m',    # Light Green
            'warning': '\033[93m',    # Light Yellow
            'error': '\033[91m',      # Light Red
            'info': '\033[96m',       # Light Cyan
            'header': '\033[34m',     # Blue
            'border': '\033[30m',     # Black
            'value': '\033[30m',      # Black
            'label': '\033[37m',      # Light Gray
            'reset': '\033[0m'
        },
        ColorTheme.CYBERPUNK: {
            'success': '\033[38;5;46m',   # Neon Green
            'warning': '\033[38;5;226m',  # Neon Yellow
            'error': '\033[38;5;196m',    # Neon Red
            'info': '\033[38;5;51m',      # Neon Cyan
            'header': '\033[38;5;129m',   # Neon Purple
            'border': '\033[38;5;87m',    # Electric Blue
            'value': '\033[38;5;255m',    # Pure White
            'label': '\033[38;5;240m',    # Dark Gray
            'reset': '\033[0m'
        },
        ColorTheme.MINIMAL: {
            'success': '\033[37m',    # White
            'warning': '\033[37m',    # White
            'error': '\033[37m',      # White
            'info': '\033[37m',       # White
            'header': '\033[1m',      # Bold
            'border': '\033[37m',     # White
            'value': '\033[37m',      # White
            'label': '\033[90m',      # Dark Gray
            'reset': '\033[0m'
        }
    }
    
    def __init__(self, theme: ColorTheme = ColorTheme.DEFAULT):
        self.theme = theme
        self.colors = self.THEMES[theme]
    
    def get(self, color_type: str) -> str:
        """색상 코드 반환"""
        return self.colors.get(color_type, self.colors['reset'])
    
    def colorize(self, text: str, color_type: str) -> str:
        """텍스트에 색상 적용"""
        if not text:
            return text
        return f"{self.get(color_type)}{text}{self.get('reset')}"

class PerformanceTracker:
    """성능 추적 및 최적화"""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.render_times = []
        self.memory_usage = []
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def get_cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    def record_render_time(self, duration: float):
        self.render_times.append(duration)
        if len(self.render_times) > 100:  # 최근 100개만 유지
            self.render_times.pop(0)
    
    def get_avg_render_time(self) -> float:
        return sum(self.render_times) / len(self.render_times) if self.render_times else 0.0

class ValidationError(Exception):
    """검증 오류 클래스"""
    pass

class EnhancedTerminalFormatter:
    """개선된 터미널 포매터 클래스"""
    
    def __init__(self, 
                 width: int = 120, 
                 theme: ColorTheme = ColorTheme.DEFAULT,
                 enable_caching: bool = True,
                 validation_mode: bool = True):
        
        # 기본 설정
        self.width = self._validate_width(width)
        self.colors = Colors(theme)
        self.enable_caching = enable_caching
        self.validation_mode = validation_mode
        
        # 성능 추적
        self.performance = PerformanceTracker()
        
        # 캐시 설정
        if enable_caching:
            self.get_display_width = lru_cache(maxsize=1000)(self._get_display_width_impl)
        else:
            self.get_display_width = self._get_display_width_impl
        
        # 경계선 문자
        self.border_char = "│"
        self.separator_char = "─"
        
        # ANSI 이스케이프 시퀀스 패턴 (캐시됨)
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    def _validate_width(self, width: int) -> int:
        """너비 유효성 검사"""
        if not isinstance(width, int):
            raise ValidationError(f"Width must be integer, got {type(width)}")
        
        if width < 40:
            raise ValidationError(f"Width too small: {width}, minimum is 40")
        
        if width > 300:
            raise ValidationError(f"Width too large: {width}, maximum is 300")
        
        return width
    
    def _get_display_width_impl(self, text: str) -> int:
        """
        문자열의 실제 표시 폭 계산 (캐시되는 구현체)
        ANSI 이스케이프 시퀀스는 제외하고 계산
        """
        if not text:
            return 0
        
        try:
            # ANSI 이스케이프 시퀀스 제거
            clean_text = self.ansi_escape.sub('', text)
            
            # wcwidth로 실제 표시 폭 계산
            width = wcswidth(clean_text)
            
            # wcwidth가 None을 반환하는 경우 처리
            if width is None:
                # 문자별로 개별 계산
                total_width = 0
                for char in clean_text:
                    char_width = wcswidth(char)
                    if char_width is None:
                        # 제어 문자나 특수 문자의 경우 0 또는 1로 처리
                        if ord(char) < 32:  # 제어 문자
                            char_width = 0
                        else:
                            char_width = 1
                    total_width += char_width
                width = total_width
            
            # 캐시 통계 업데이트
            if self.enable_caching:
                self.performance.record_cache_miss()
            
            return max(0, width)  # 음수 방지
            
        except Exception as e:
            if self.validation_mode:
                # 검증 모드에서는 예외 발생
                raise ValidationError(f"Failed to calculate width for text: {str(e)}")
            else:
                # 기본 모드에서는 안전한 기본값 반환
                return len(text)
    
    def set_theme(self, theme: ColorTheme):
        """테마 변경"""
        self.colors = Colors(theme)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        return {
            'cache_hit_rate': f"{self.performance.get_cache_hit_rate():.1f}%",
            'avg_render_time': f"{self.performance.get_avg_render_time():.3f}ms",
            'cache_hits': self.performance.cache_hits,
            'cache_misses': self.performance.cache_misses
        }
    
    def pad_line(self, text: str, target_width: Optional[int] = None, 
                 alignment: str = 'left') -> str:
        """
        문자열을 지정된 너비로 패딩
        한글/이모지 등의 실제 표시 폭을 고려
        """
        if target_width is None:
            target_width = self.width - 4  # 양쪽 border와 공백 제외
            
        current_width = self.get_display_width(text)
        
        if current_width < target_width:
            padding = target_width - current_width
            
            if alignment == 'center':
                left_pad = padding // 2
                right_pad = padding - left_pad
                return ' ' * left_pad + text + ' ' * right_pad
            elif alignment == 'right':
                return ' ' * padding + text
            else:  # left
                return text + ' ' * padding
                
        elif current_width > target_width:
            # 텍스트 자르기
            return self.truncate_to_width(text, target_width - 3) + "..."
        else:
            return text
    
    def truncate_to_width(self, text: str, max_width: int) -> str:
        """
        지정된 표시 폭에 맞게 문자열 자르기
        Wide 문자(한글 등)가 잘리지 않도록 처리
        """
        if max_width <= 0:
            return ""
        
        # ANSI 코드 보존을 위한 처리
        ansi_codes = []
        clean_text = ""
        last_pos = 0
        
        for match in self.ansi_escape.finditer(text):
            ansi_codes.append((len(clean_text) + (match.start() - last_pos), match.group()))
            clean_text += text[last_pos:match.start()]
            last_pos = match.end()
        
        clean_text += text[last_pos:]
        
        # 폭 계산하여 자르기
        current_width = 0
        result_chars = []
        
        for i, char in enumerate(clean_text):
            char_width = wcswidth(char)
            if char_width is None:
                char_width = 0 if ord(char) < 32 else 1
                
            if current_width + char_width > max_width:
                break
                
            result_chars.append(char)
            current_width += char_width
        
        # ANSI 코드 재삽입
        result = ''.join(result_chars)
        for pos, code in reversed(ansi_codes):
            if pos <= len(result):
                result = result[:pos] + code + result[pos:]
        
        return result
    
    def format_line(self, content: str, alignment: str = 'left') -> str:
        """
        대시보드 라인 포맷팅
        양쪽에 border 추가하고 패딩 적용
        """
        padded_content = self.pad_line(content, alignment=alignment)
        border_color = self.colors.get('border')
        reset_color = self.colors.get('reset')
        
        return f"{border_color}{self.border_char}{reset_color} {padded_content} {border_color}{self.border_char}{reset_color}"
    
    def format_header(self, title: str, style: str = 'center') -> str:
        """헤더 라인 포맷팅"""
        colored_title = self.colors.colorize(title, 'header')
        
        if style == 'center':
            return self.format_line(colored_title, 'center')
        else:
            return self.format_line(colored_title, 'left')
    
    def format_separator(self, char: Optional[str] = None) -> str:
        """구분선 생성"""
        if char is None:
            char = self.separator_char
        
        border_color = self.colors.get('border')
        reset_color = self.colors.get('reset')
        
        return f"{border_color}{char * self.width}{reset_color}"
    
    def format_data_line(self, label: str, value: str, 
                        label_width: int = 30, 
                        value_color: str = 'value',
                        label_color: str = 'label') -> str:
        """
        라벨-값 쌍을 포맷팅
        """
        colored_label = self.colors.colorize(label, label_color)
        colored_value = self.colors.colorize(value, value_color)
        
        padded_label = self.pad_line(colored_label, label_width)
        content = f"{padded_label}: {colored_value}"
        
        return self.format_line(content)
    
    def format_progress_bar(self, label: str, percentage: float, 
                           bar_width: int = 20, label_width: int = 20,
                           show_percentage: bool = True) -> str:
        """
        프로그레스 바 포맷팅
        """
        # 라벨 처리
        colored_label = self.colors.colorize(label, 'label')
        padded_label = self.pad_line(colored_label, label_width)
        
        # 프로그레스 바 생성
        filled = int(bar_width * percentage / 100)
        empty = bar_width - filled
        
        # 색상 결정
        if percentage >= 80:
            bar_color = 'error'
        elif percentage >= 60:
            bar_color = 'warning'
        else:
            bar_color = 'success'
        
        # 바 문자 선택 (테마에 따라)
        if self.colors.theme == ColorTheme.CYBERPUNK:
            filled_char = '▰'
            empty_char = '▱'
        else:
            filled_char = '█'
            empty_char = '░'
        
        bar = self.colors.colorize(filled_char * filled + empty_char * empty, bar_color)
        
        # 퍼센트 표시
        if show_percentage:
            percentage_str = f" {percentage:5.1f}%"
        else:
            percentage_str = ""
        
        content = f"{padded_label}: {bar}{percentage_str}"
        return self.format_line(content)
    
    def format_status_line(self, service: str, status: str, details: str = "",
                          status_width: int = 15) -> str:
        """
        서비스 상태 라인 포맷팅
        """
        # 상태에 따른 색상 결정
        status_lower = status.lower()
        if 'connected' in status_lower or 'success' in status_lower or status_lower == 'ok':
            status_color = 'success'
            icon = "✅"
        elif 'warning' in status_lower or 'partial' in status_lower:
            status_color = 'warning'
            icon = "⚠️"
        elif 'error' in status_lower or 'failed' in status_lower or 'disconnect' in status_lower:
            status_color = 'error'
            icon = "❌"
        else:
            status_color = 'info'
            icon = "ℹ️"
        
        # 서비스명 포맷팅
        service_formatted = f"{icon} {service}"
        
        # 상태 포맷팅
        status_formatted = self.colors.colorize(status, status_color)
        padded_status = self.pad_line(status_formatted, status_width)
        
        # 전체 라인 구성
        if details:
            content = f"{service_formatted}: {padded_status} | {details}"
        else:
            content = f"{service_formatted}: {padded_status}"
        
        return self.format_line(content)
    
    def format_table_row(self, columns: List[str], column_widths: List[int],
                        alignments: Optional[List[str]] = None) -> str:
        """
        테이블 행 포맷팅
        """
        if alignments is None:
            alignments = ['left'] * len(columns)
        
        if len(columns) != len(column_widths) or len(columns) != len(alignments):
            raise ValidationError("Columns, widths, and alignments must have same length")
        
        formatted_columns = []
        for col, width, align in zip(columns, column_widths, alignments):
            formatted_columns.append(self.pad_line(col, width, align))
        
        content = " │ ".join(formatted_columns)
        return self.format_line(content)
    
    def create_dashboard_frame(self, title: str, content_lines: List[str],
                             footer: Optional[str] = None) -> str:
        """
        완전한 대시보드 프레임 생성
        """
        lines = []
        
        # 상단 경계
        lines.append(self.format_separator())
        
        # 헤더
        lines.append(self.format_header(title))
        lines.append(self.format_separator())
        
        # 컨텐츠
        for line in content_lines:
            if line == "---":  # 구분선 특수 처리
                lines.append(self.format_separator("─"))
            else:
                lines.append(line)
        
        # 푸터
        if footer:
            lines.append(self.format_separator())
            lines.append(self.format_line(footer, 'center'))
        
        # 하단 경계
        lines.append(self.format_separator())
        
        return '\n'.join(lines)

# 실제 데이터 연동을 위한 데이터 어댑터
class DataAdapter:
    """실제 데이터와의 연동을 위한 어댑터"""
    
    def __init__(self, verification_results_path: str = "verification_results.json"):
        self.verification_results_path = verification_results_path
        
    def load_verification_results(self) -> Optional[Dict[str, Any]]:
        """검증 결과 파일 로드"""
        try:
            if os.path.exists(self.verification_results_path):
                with open(self.verification_results_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load verification results: {e}")
        return None
    
    def get_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """연결 상태 정보 반환"""
        results = self.load_verification_results()
        
        if results and 'connections' in results:
            return results['connections']
        
        # 기본 데이터 (검증 결과가 없는 경우)
        return {
            'binance_testnet': {'success': False, 'details': 'Not verified'},
            'binance_mainnet': {'success': False, 'details': 'Not verified'},
            'newsapi': {'success': False, 'details': 'Not verified'},
            'finnhub': {'success': False, 'details': 'Not verified'},
            'telegram': {'success': False, 'details': 'Not verified'},
            'redis': {'success': False, 'details': 'Not verified'}
        }
    
    def get_system_metrics(self) -> Dict[str, float]:
        """시스템 메트릭 반환 (실제 구현에서는 psutil 등 사용)"""
        import random
        return {
            'cpu_usage': random.uniform(20, 80),
            'memory_usage': random.uniform(30, 70),
            'network_usage': random.uniform(5, 30),
            'disk_usage': random.uniform(40, 90)
        }

# 사용 예제 및 실제 대시보드 생성
def create_enhanced_dashboard(theme: ColorTheme = ColorTheme.DEFAULT) -> str:
    """개선된 대시보드 생성"""
    
    # 포매터 초기화
    formatter = EnhancedTerminalFormatter(width=120, theme=theme, enable_caching=True)
    data_adapter = DataAdapter()
    
    # 실제 데이터 로드
    connections = data_adapter.get_connection_status()
    metrics = data_adapter.get_system_metrics()
    
    content_lines = []
    
    # 빈 줄
    content_lines.append(formatter.format_line(""))
    
    # 메인 스코어 섹션
    content_lines.append(formatter.format_data_line(
        "Fusion Sentiment Score", 
        "5.0% (+0.05)", 
        30, 
        'error'
    ))
    
    # 카테고리별 점수
    categories = [
        ("News", "0.0%", 'value'),
        ("Reddit", "0.0%", 'error'),
        ("Tech", "0.0%", 'error'),
        ("Historical", "50.0%", 'warning')
    ]
    
    cat_content = ""
    for cat, val, color in categories:
        colored_val = formatter.colors.colorize(val, color)
        cat_content += f"  {cat}: {colored_val}"
    
    content_lines.append(formatter.format_line(cat_content))
    content_lines.append(formatter.format_line(""))
    
    # 이벤트 타임라인
    content_lines.append(formatter.format_line(formatter.colors.colorize("[ Big Event Timeline ]", 'header')))
    
    events = [
        ("2025. 7. 29. 오후  │ FOMC 발표", "Impact: 0.85 │ Sentiment: -22.0% │ Volatility: 0.60"),
        ("2025. 7. 29. 오후  │ ETF 승인", "Impact: 0.72 │ Sentiment: 40.0% │ Volatility: 0.45")
    ]
    
    for event, details in events:
        content_lines.append(formatter.format_data_line(event, details, 40))
    
    content_lines.append(formatter.format_line(""))
    
    # 전략 성과
    content_lines.append(formatter.format_line(formatter.colors.colorize("[ Strategy Performance ]", 'header')))
    
    strategies = [
        ("AuroraQ: RuleA ROI 3.2% Sharpe 1.25", "Score 5.0%", 'success'),
        ("MacroQ: BTC Portfolio ROI 2.5% Sharpe 0.90", "Score -1.0%", 'error')
    ]
    
    for strat, score, color in strategies:
        content_lines.append(formatter.format_data_line(strat, score, 50, color))
    
    content_lines.append(formatter.format_line(""))
    
    # API & 시스템 상태
    content_lines.append(formatter.format_line(formatter.colors.colorize("[ API & System Health ]", 'header')))
    
    # 연결 상태
    for service, info in connections.items():
        status = "Connected" if info['success'] else "Failed"
        details = str(info.get('details', ''))[:50]  # 길이 제한
        content_lines.append(formatter.format_status_line(service.replace('_', ' ').title(), status, details))
    
    content_lines.append(formatter.format_line(""))
    
    # 시스템 메트릭
    content_lines.append(formatter.format_data_line(
        "Redis Hit Rate", "92%", 30, 'success'
    ))
    
    # 프로그레스 바들
    content_lines.append(formatter.format_progress_bar("CPU Usage", metrics['cpu_usage'], 20, 30))
    content_lines.append(formatter.format_progress_bar("Memory Usage", metrics['memory_usage'], 20, 30))
    content_lines.append(formatter.format_progress_bar("Network Usage", metrics['network_usage'], 20, 30))
    content_lines.append(formatter.format_progress_bar("Disk Usage", metrics['disk_usage'], 20, 30))
    
    content_lines.append(formatter.format_line(""))
    
    # 성능 통계
    perf_stats = formatter.get_performance_stats()
    content_lines.append(formatter.format_line(formatter.colors.colorize("[ Performance Stats ]", 'header')))
    content_lines.append(formatter.format_data_line("Cache Hit Rate", perf_stats['cache_hit_rate'], 30, 'info'))
    content_lines.append(formatter.format_data_line("Avg Render Time", perf_stats['avg_render_time'], 30, 'info'))
    
    # 현재 시간
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = f"Last updated: {timestamp} | Theme: {theme.value} | AuroraQ Dashboard v2.0"
    
    # 완전한 대시보드 생성
    dashboard = formatter.create_dashboard_frame(
        "Enhanced Sentiment-Service Dashboard (AuroraQ + MacroQ)",
        content_lines,
        footer
    )
    
    return dashboard

def demo_all_themes():
    """모든 테마 데모"""
    for theme in ColorTheme:
        print(f"\n{'='*60}")
        print(f"THEME: {theme.value.upper()}")
        print('='*60)
        print(create_enhanced_dashboard(theme))
        
        if theme != list(ColorTheme)[-1]:  # 마지막이 아니면
            input("\nPress Enter to see next theme...")

if __name__ == "__main__":
    # 기본 대시보드 출력
    print(create_enhanced_dashboard())
    
    # 모든 테마 데모를 원하면 아래 주석 해제
    # demo_all_themes()