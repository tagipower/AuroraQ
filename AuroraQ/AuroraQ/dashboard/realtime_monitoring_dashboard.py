#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for AuroraQ
AuroraQ 실시간 모니터링 대시보드 - 폴백 이벤트 및 품질 메트릭 통합 모니터링
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import statistics
from collections import defaultdict, deque
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import jinja2

# AuroraQ 컴포넌트 임포트
try:
    from utils.enhanced_fallback_manager import get_fallback_manager, EnhancedFallbackManager
    from utils.predictive_quality_optimizer import get_quality_optimizer, PredictiveQualityOptimizer
except ImportError:
    # 개발 환경에서 직접 실행할 때
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.enhanced_fallback_manager import get_fallback_manager, EnhancedFallbackManager
    from utils.predictive_quality_optimizer import get_quality_optimizer, PredictiveQualityOptimizer

logger = logging.getLogger(__name__)

class DashboardStatus(Enum):
    """대시보드 상태"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"

class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemAlert:
    """시스템 알림"""
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardMetrics:
    """대시보드 메트릭"""
    timestamp: datetime
    system_status: DashboardStatus
    fallback_manager_metrics: Dict[str, Any]
    quality_optimizer_metrics: Dict[str, Any]
    combined_metrics: Dict[str, Any]
    active_alerts: List[SystemAlert]
    performance_stats: Dict[str, Any]

class RealtimeMonitoringDashboard:
    """실시간 모니터링 대시보드"""
    
    def __init__(self, 
                 update_interval: int = 5,
                 alert_retention_hours: int = 24,
                 max_data_points: int = 1000):
        """
        초기화
        
        Args:
            update_interval: 업데이트 간격 (초)
            alert_retention_hours: 알림 보관 시간 (시간)
            max_data_points: 최대 데이터 포인트 수
        """
        self.update_interval = update_interval
        self.alert_retention_hours = alert_retention_hours
        self.max_data_points = max_data_points
        
        # 컴포넌트 매니저들
        self.fallback_manager = get_fallback_manager()
        self.quality_optimizer = get_quality_optimizer()
        
        # 데이터 저장소
        self.metrics_history: deque = deque(maxlen=max_data_points)
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # WebSocket 연결 관리
        self.websocket_connections: List[WebSocket] = []
        
        # 모니터링 스레드
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        
        # FastAPI 앱
        self.app = FastAPI(title="AuroraQ Monitoring Dashboard", version="1.0.0")
        self._setup_routes()
        
        # 임계값 설정
        self.thresholds = {
            "fallback_rate_warning": 0.7,      # 70% 이상 폴백 발생시 경고
            "fallback_rate_critical": 0.85,    # 85% 이상 폴백 발생시 위험
            "quality_warning": 0.7,            # 70% 미만 품질시 경고  
            "quality_critical": 0.6,           # 60% 미만 품질시 위험
            "response_time_warning": 5.0,      # 5초 이상 응답시간시 경고
            "response_time_critical": 10.0,    # 10초 이상 응답시간시 위험
            "error_rate_warning": 0.05,        # 5% 이상 에러율시 경고
            "error_rate_critical": 0.1         # 10% 이상 에러율시 위험
        }
        
        logger.info(f"Real-time Monitoring Dashboard initialized with {update_interval}s update interval")
    
    def _setup_routes(self):
        """FastAPI 라우트 설정"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_page():
            """대시보드 메인 페이지"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_system_status():
            """시스템 상태 API"""
            try:
                current_metrics = self._collect_current_metrics()
                return JSONResponse({
                    "status": "success",
                    "data": asdict(current_metrics)
                })
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history(hours: int = 1):
            """메트릭 히스토리 API"""
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                history = [
                    asdict(metrics) for metrics in self.metrics_history 
                    if metrics.timestamp >= cutoff_time
                ]
                return JSONResponse({
                    "status": "success",
                    "data": history,
                    "count": len(history)
                })
            except Exception as e:
                logger.error(f"Error getting metrics history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts")
        async def get_alerts(active_only: bool = True):
            """알림 목록 API"""
            try:
                if active_only:
                    alerts = [asdict(alert) for alert in self.active_alerts.values() if not alert.resolved]
                else:
                    alerts = [asdict(alert) for alert in self.alert_history]
                
                return JSONResponse({
                    "status": "success", 
                    "data": alerts,
                    "count": len(alerts)
                })
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """알림 해결 API"""
            try:
                if alert_id in self.active_alerts:
                    self.active_alerts[alert_id].resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    await self._broadcast_update()
                    return JSONResponse({"status": "success", "message": "Alert resolved"})
                else:
                    raise HTTPException(status_code=404, detail="Alert not found")
            except Exception as e:
                logger.error(f"Error resolving alert: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 엔드포인트"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.websocket_connections)}")
            
            try:
                while True:
                    # 주기적으로 ping 메시지 전송
                    await asyncio.sleep(30)
                    await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                logger.info(f"WebSocket disconnected. Total connections: {len(self.websocket_connections)}")
    
    def _get_dashboard_html(self) -> str:
        """대시보드 HTML 페이지 반환"""
        return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AuroraQ Monitoring Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header h1 { 
            color: #2d3748;
            font-size: 2.5em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-left: 15px;
            animation: pulse 2s infinite;
        }
        .status-healthy { background: #48bb78; }
        .status-warning { background: #ed8936; }
        .status-critical { background: #f56565; }
        .status-offline { background: #a0aec0; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        .metric-card:hover { transform: translateY(-2px); }
        .metric-card h3 {
            color: #2d3748;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-good { color: #48bb78; }
        .metric-warning { color: #ed8936; }
        .metric-critical { color: #f56565; }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .alerts-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .alert-item {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-info { border-color: #3182ce; background: #ebf8ff; }
        .alert-warning { border-color: #ed8936; background: #fffaf0; }
        .alert-error { border-color: #f56565; background: #fed7d7; }
        .alert-critical { border-color: #c53030; background: #feb2b2; }
        
        .update-time {
            color: #718096;
            font-size: 0.9em;
            text-align: right;
            margin-top: 10px;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #718096;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🚀 AuroraQ Monitoring Dashboard
                <div id="statusIndicator" class="status-indicator status-offline"></div>
            </h1>
            <p>실시간 폴백 이벤트 및 품질 메트릭 모니터링</p>
            <div class="update-time" id="lastUpdate">마지막 업데이트: 연결 중...</div>
        </div>
        
        <div class="metrics-grid" id="metricsGrid">
            <div class="loading">📊 메트릭 로딩 중...</div>
        </div>
        
        <div class="chart-container">
            <h3>📈 실시간 메트릭 차트</h3>
            <div id="chartArea" style="height: 300px; display: flex; align-items: center; justify-content: center; color: #718096;">
                차트 준비 중...
            </div>
        </div>
        
        <div class="alerts-container">
            <h3>🚨 활성 알림</h3>
            <div id="alertsList">
                <div class="loading">알림 로딩 중...</div>
            </div>
        </div>
    </div>

    <script>
        class DashboardManager {
            constructor() {
                this.ws = null;
                this.reconnectInterval = 5000;
                this.maxReconnectAttempts = 10;
                this.reconnectAttempts = 0;
                this.chartData = [];
                this.maxDataPoints = 50;
                
                this.initializeWebSocket();
                this.fetchInitialData();
                
                // 주기적으로 REST API로 데이터 업데이트
                setInterval(() => this.fetchMetrics(), 10000);
            }
            
            initializeWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('connected');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics_update') {
                        this.updateDashboard(data.data);
                    }
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.updateConnectionStatus('disconnected');
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.initializeWebSocket(), this.reconnectInterval);
                }
            }
            
            async fetchInitialData() {
                await this.fetchMetrics();
                await this.fetchAlerts();
            }
            
            async fetchMetrics() {
                try {
                    const response = await fetch('/api/status');
                    const result = await response.json();
                    if (result.status === 'success') {
                        this.updateDashboard(result.data);
                    }
                } catch (error) {
                    console.error('Error fetching metrics:', error);
                }
            }
            
            async fetchAlerts() {
                try {
                    const response = await fetch('/api/alerts');
                    const result = await response.json();
                    if (result.status === 'success') {
                        this.updateAlerts(result.data);
                    }
                } catch (error) {
                    console.error('Error fetching alerts:', error);
                }
            }
            
            updateConnectionStatus(status) {
                const indicator = document.getElementById('statusIndicator');
                const lastUpdate = document.getElementById('lastUpdate');
                
                if (status === 'connected') {
                    indicator.className = 'status-indicator status-healthy';
                    lastUpdate.textContent = `마지막 업데이트: ${new Date().toLocaleString()}`;
                } else {
                    indicator.className = 'status-indicator status-offline';
                    lastUpdate.textContent = '연결 끊김 - 재연결 시도 중...';
                }
            }
            
            updateDashboard(data) {
                this.updateMetricsGrid(data);
                this.updateChart(data);
                document.getElementById('lastUpdate').textContent = `마지막 업데이트: ${new Date().toLocaleString()}`;
            }
            
            updateMetricsGrid(data) {
                const grid = document.getElementById('metricsGrid');
                const fbMetrics = data.fallback_manager_metrics || {};
                const qoMetrics = data.quality_optimizer_metrics || {};
                const combined = data.combined_metrics || {};
                
                grid.innerHTML = `
                    <div class="metric-card">
                        <h3>🔄 폴백 상태</h3>
                        <div class="metric-value ${this.getMetricClass(fbMetrics.fallback_rate_status)}">
                            ${((fbMetrics.current_fallback_rate || 0) * 100).toFixed(1)}%
                        </div>
                        <div>목표: ${((fbMetrics.target_fallback_rate || 0.6) * 100).toFixed(0)}% 이하</div>
                        <div>상태: ${fbMetrics.fallback_rate_status || 'Unknown'}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>📊 데이터 품질</h3>
                        <div class="metric-value ${this.getMetricClass(fbMetrics.data_quality_status)}">
                            ${((fbMetrics.current_data_quality || 0) * 100).toFixed(1)}%
                        </div>
                        <div>목표: ${((fbMetrics.target_data_quality || 0.8) * 100).toFixed(0)}% 이상</div>
                        <div>상태: ${fbMetrics.data_quality_status || 'Unknown'}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>⚡ 시스템 성능</h3>
                        <div class="metric-value ${this.getSystemStatusClass(data.system_status)}">
                            ${data.system_status || 'Unknown'}
                        </div>
                        <div>전체 작업: ${fbMetrics.total_operations || 0}</div>
                        <div>성공 이벤트: ${fbMetrics.success_events || 0}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🎯 품질 최적화</h3>
                        <div class="metric-value metric-good">
                            ${qoMetrics.total_assessments || 0}
                        </div>
                        <div>품질 개선: ${qoMetrics.quality_improvements || 0}</div>
                        <div>예측 이슈: ${qoMetrics.predicted_issues || 0}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🚨 활성 알림</h3>
                        <div class="metric-value ${(data.active_alerts || []).length > 0 ? 'metric-warning' : 'metric-good'}">
                            ${(data.active_alerts || []).length}
                        </div>
                        <div>총 알림: ${(data.active_alerts || []).length}</div>
                        <div>해결됨: ${(data.active_alerts || []).filter(a => a.resolved).length}</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>📈 종합 점수</h3>
                        <div class="metric-value ${this.getCombinedScoreClass(combined.overall_score || 0)}">
                            ${((combined.overall_score || 0) * 100).toFixed(0)}점
                        </div>
                        <div>안정성: ${((combined.stability_score || 0) * 100).toFixed(0)}%</div>
                        <div>효율성: ${((combined.efficiency_score || 0) * 100).toFixed(0)}%</div>
                    </div>
                `;
            }
            
            updateChart(data) {
                // 차트 데이터 추가
                this.chartData.push({
                    timestamp: new Date(),
                    fallback_rate: (data.fallback_manager_metrics?.current_fallback_rate || 0) * 100,
                    data_quality: (data.fallback_manager_metrics?.current_data_quality || 0) * 100,
                    overall_score: (data.combined_metrics?.overall_score || 0) * 100
                });
                
                // 최대 데이터 포인트 수 제한
                if (this.chartData.length > this.maxDataPoints) {
                    this.chartData.shift();
                }
                
                // 간단한 텍스트 기반 차트 표시
                const chartArea = document.getElementById('chartArea');
                const latest = this.chartData[this.chartData.length - 1];
                if (latest) {
                    chartArea.innerHTML = `
                        <div style="width: 100%; text-align: center;">
                            <div style="margin-bottom: 20px;">
                                <strong>현재 메트릭 (최근 ${this.chartData.length}개 데이터 포인트)</strong>
                            </div>
                            <div style="display: flex; justify-content: space-around;">
                                <div>
                                    <div style="font-size: 1.5em; color: #ed8936;">폴백률</div>
                                    <div style="font-size: 2em; font-weight: bold;">${latest.fallback_rate.toFixed(1)}%</div>
                                </div>
                                <div>
                                    <div style="font-size: 1.5em; color: #48bb78;">데이터품질</div>
                                    <div style="font-size: 2em; font-weight: bold;">${latest.data_quality.toFixed(1)}%</div>
                                </div>
                                <div>
                                    <div style="font-size: 1.5em; color: #3182ce;">종합점수</div>
                                    <div style="font-size: 2em; font-weight: bold;">${latest.overall_score.toFixed(0)}점</div>
                                </div>
                            </div>
                        </div>
                    `;
                }
            }
            
            updateAlerts(alerts) {
                const alertsList = document.getElementById('alertsList');
                if (alerts.length === 0) {
                    alertsList.innerHTML = '<div style="text-align: center; color: #48bb78; padding: 20px;">✅ 활성 알림 없음</div>';
                    return;
                }
                
                alertsList.innerHTML = alerts.map(alert => `
                    <div class="alert-item alert-${alert.level}">
                        <div style="display: flex; justify-content: between; align-items: center;">
                            <div>
                                <strong>${alert.component}</strong>: ${alert.message}
                            </div>
                            <div style="font-size: 0.8em; color: #718096;">
                                ${new Date(alert.timestamp).toLocaleString()}
                            </div>
                        </div>
                        ${!alert.resolved ? `<button onclick="dashboard.resolveAlert('${alert.id}')" style="margin-top: 10px; padding: 5px 10px; background: #48bb78; color: white; border: none; border-radius: 3px; cursor: pointer;">해결</button>` : ''}
                    </div>
                `).join('');
            }
            
            getMetricClass(status) {
                switch(status) {
                    case 'GOOD': return 'metric-good';
                    case 'NEEDS_IMPROVEMENT': return 'metric-warning';
                    default: return 'metric-critical';
                }
            }
            
            getSystemStatusClass(status) {
                switch(status) {
                    case 'healthy': return 'metric-good';
                    case 'warning': return 'metric-warning';
                    case 'critical': return 'metric-critical';
                    default: return 'metric-critical';
                }
            }
            
            getCombinedScoreClass(score) {
                if (score >= 0.8) return 'metric-good';
                if (score >= 0.6) return 'metric-warning';
                return 'metric-critical';
            }
            
            async resolveAlert(alertId) {
                try {
                    const response = await fetch(`/api/alerts/${alertId}/resolve`, {
                        method: 'POST'
                    });
                    if (response.ok) {
                        await this.fetchAlerts();
                    }
                } catch (error) {
                    console.error('Error resolving alert:', error);
                }
            }
        }
        
        // 전역 대시보드 인스턴스
        const dashboard = new DashboardManager();
    </script>
</body>
</html>
        """
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring_thread.is_alive():
            self.monitoring_thread.start()
        logger.info("Real-time monitoring started")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while self.monitoring_active:
            try:
                # 현재 메트릭 수집
                current_metrics = self._collect_current_metrics()
                
                # 메트릭 히스토리에 추가
                self.metrics_history.append(current_metrics)
                
                # 알림 검사 및 생성
                self._check_and_generate_alerts(current_metrics)
                
                # 오래된 알림 정리
                self._cleanup_old_alerts()
                
                # WebSocket으로 실시간 업데이트 브로드캐스트
                asyncio.create_task(self._broadcast_update())
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_current_metrics(self) -> DashboardMetrics:
        """현재 메트릭 수집"""
        
        # 폴백 매니저 메트릭
        fb_metrics = self.fallback_manager.get_current_metrics()
        
        # 품질 최적화기 메트릭
        qo_dashboard_data = self.quality_optimizer.get_quality_dashboard_data()
        qo_metrics = qo_dashboard_data["statistics"]
        
        # 결합 메트릭 계산
        combined_metrics = self._calculate_combined_metrics(fb_metrics, qo_dashboard_data)
        
        # 시스템 상태 결정
        system_status = self._determine_system_status(fb_metrics, qo_dashboard_data, combined_metrics)
        
        # 성능 통계
        performance_stats = self._calculate_performance_stats()
        
        return DashboardMetrics(
            timestamp=datetime.now(),
            system_status=system_status,
            fallback_manager_metrics=fb_metrics,
            quality_optimizer_metrics=qo_metrics,
            combined_metrics=combined_metrics,
            active_alerts=list(self.active_alerts.values()),
            performance_stats=performance_stats
        )
    
    def _calculate_combined_metrics(self, fb_metrics: Dict[str, Any], qo_data: Dict[str, Any]) -> Dict[str, Any]:
        """결합 메트릭 계산"""
        
        # 폴백률 점수 (낮을수록 좋음)
        fallback_rate = fb_metrics.get("current_fallback_rate", 1.0)
        fallback_score = max(0, 1.0 - (fallback_rate / self.fallback_manager.target_fallback_rate))
        
        # 데이터 품질 점수
        quality_score = fb_metrics.get("current_data_quality", 0.0)
        
        # 전체 상태 점수
        qo_overall = qo_data["overall_status"]
        quality_target_achievement = qo_overall["current_quality"] / qo_overall["target_quality"]
        
        # 가중 평균으로 종합 점수 계산
        overall_score = (
            fallback_score * 0.4 +          # 폴백 안정성 40%
            quality_score * 0.4 +           # 데이터 품질 40%  
            quality_target_achievement * 0.2 # 목표 달성도 20%
        )
        
        # 안정성 점수 (폴백 빈도 기반)
        stability_score = max(0, 1.0 - fallback_rate)
        
        # 효율성 점수 (품질 및 성능 기반)
        total_ops = fb_metrics.get("total_operations", 1)
        success_rate = fb_metrics.get("success_events", 0) / total_ops if total_ops > 0 else 1.0
        efficiency_score = (quality_score + success_rate) / 2
        
        return {
            "overall_score": min(1.0, max(0.0, overall_score)),
            "stability_score": stability_score,
            "efficiency_score": efficiency_score,
            "fallback_score": fallback_score,
            "quality_score": quality_score,
            "target_achievement": quality_target_achievement,
            "success_rate": success_rate
        }
    
    def _determine_system_status(self, fb_metrics: Dict[str, Any], qo_data: Dict[str, Any], combined: Dict[str, Any]) -> DashboardStatus:
        """시스템 상태 결정"""
        
        fallback_rate = fb_metrics.get("current_fallback_rate", 0)
        quality = fb_metrics.get("current_data_quality", 1.0)
        overall_score = combined.get("overall_score", 0)
        
        # 위험 상태 체크
        if (fallback_rate >= self.thresholds["fallback_rate_critical"] or 
            quality <= self.thresholds["quality_critical"] or
            overall_score <= 0.5):
            return DashboardStatus.CRITICAL
        
        # 경고 상태 체크  
        if (fallback_rate >= self.thresholds["fallback_rate_warning"] or
            quality <= self.thresholds["quality_warning"] or
            overall_score <= 0.7):
            return DashboardStatus.WARNING
        
        return DashboardStatus.HEALTHY
    
    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 계산"""
        if len(self.metrics_history) < 2:
            return {
                "avg_response_time": 0,
                "metrics_collection_rate": 0,
                "dashboard_uptime": 0
            }
        
        # 최근 메트릭들의 성능 분석
        recent_metrics = list(self.metrics_history)[-10:]  # 최근 10개
        
        # 메트릭 수집 간격 분석
        intervals = []
        for i in range(1, len(recent_metrics)):
            interval = (recent_metrics[i].timestamp - recent_metrics[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        avg_interval = statistics.mean(intervals) if intervals else self.update_interval
        collection_rate = 1.0 / avg_interval if avg_interval > 0 else 0
        
        return {
            "avg_response_time": avg_interval,
            "metrics_collection_rate": collection_rate,
            "dashboard_uptime": len(self.metrics_history) * self.update_interval / 60  # 분 단위
        }
    
    def _check_and_generate_alerts(self, metrics: DashboardMetrics):
        """알림 검사 및 생성"""
        
        fb_metrics = metrics.fallback_manager_metrics
        combined = metrics.combined_metrics
        
        # 폴백률 임계값 체크
        fallback_rate = fb_metrics.get("current_fallback_rate", 0)
        if fallback_rate >= self.thresholds["fallback_rate_critical"]:
            self._create_alert(
                "fallback_rate_critical",
                AlertLevel.CRITICAL,
                "fallback_manager", 
                f"폴백률이 위험 수준입니다: {fallback_rate:.1%} (임계값: {self.thresholds['fallback_rate_critical']:.1%})",
                {"fallback_rate": fallback_rate, "threshold": self.thresholds["fallback_rate_critical"]}
            )
        elif fallback_rate >= self.thresholds["fallback_rate_warning"]:
            self._create_alert(
                "fallback_rate_warning",
                AlertLevel.WARNING,
                "fallback_manager",
                f"폴백률이 경고 수준입니다: {fallback_rate:.1%} (임계값: {self.thresholds['fallback_rate_warning']:.1%})",
                {"fallback_rate": fallback_rate, "threshold": self.thresholds["fallback_rate_warning"]}
            )
        
        # 데이터 품질 임계값 체크
        quality = fb_metrics.get("current_data_quality", 1.0)
        if quality <= self.thresholds["quality_critical"]:
            self._create_alert(
                "quality_critical",
                AlertLevel.CRITICAL,
                "quality_optimizer",
                f"데이터 품질이 위험 수준입니다: {quality:.1%} (임계값: {self.thresholds['quality_critical']:.1%})",
                {"quality": quality, "threshold": self.thresholds["quality_critical"]}
            )
        elif quality <= self.thresholds["quality_warning"]:
            self._create_alert(
                "quality_warning", 
                AlertLevel.WARNING,
                "quality_optimizer",
                f"데이터 품질이 경고 수준입니다: {quality:.1%} (임계값: {self.thresholds['quality_warning']:.1%})",
                {"quality": quality, "threshold": self.thresholds["quality_warning"]}
            )
        
        # 종합 점수 체크
        overall_score = combined.get("overall_score", 0)
        if overall_score <= 0.5:
            self._create_alert(
                "system_critical",
                AlertLevel.CRITICAL,
                "system",
                f"시스템 종합 점수가 위험 수준입니다: {overall_score:.1%}",
                {"overall_score": overall_score}
            )
        elif overall_score <= 0.7:
            self._create_alert(
                "system_warning",
                AlertLevel.WARNING, 
                "system",
                f"시스템 종합 점수가 경고 수준입니다: {overall_score:.1%}",
                {"overall_score": overall_score}
            )
    
    def _create_alert(self, alert_id: str, level: AlertLevel, component: str, message: str, metadata: Dict[str, Any]):
        """알림 생성"""
        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
            return  # 이미 활성 알림이 있음
        
        alert = SystemAlert(
            id=alert_id,
            level=level,
            component=component,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert created: {alert_id} - {message}")
    
    def _cleanup_old_alerts(self):
        """오래된 알림 정리"""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
        
        # 해결된 오래된 알림 제거
        alerts_to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved and alert.timestamp < cutoff_time:
                alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
    
    async def _broadcast_update(self):
        """WebSocket으로 업데이트 브로드캐스트"""
        if not self.websocket_connections:
            return
        
        try:
            current_metrics = self._collect_current_metrics()
            message = {
                "type": "metrics_update",
                "data": asdict(current_metrics),
                "timestamp": datetime.now().isoformat()
            }
            
            # 연결된 모든 클라이언트에게 전송
            disconnected_connections = []
            for ws in self.websocket_connections:
                try:
                    await ws.send_json(message)
                except Exception as e:
                    logger.debug(f"Failed to send to WebSocket: {e}")
                    disconnected_connections.append(ws)
            
            # 끊어진 연결 제거
            for ws in disconnected_connections:
                if ws in self.websocket_connections:
                    self.websocket_connections.remove(ws)
                    
        except Exception as e:
            logger.error(f"Error broadcasting update: {e}")
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """서버 실행"""
        self.start_monitoring()
        logger.info(f"Starting dashboard server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")
    
    def shutdown(self):
        """대시보드 종료"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Real-time Monitoring Dashboard shut down")

# 전역 대시보드 인스턴스
_dashboard: Optional[RealtimeMonitoringDashboard] = None

def get_dashboard() -> RealtimeMonitoringDashboard:
    """전역 대시보드 반환"""
    global _dashboard
    if _dashboard is None:
        _dashboard = RealtimeMonitoringDashboard()
    return _dashboard

# 테스트 및 실행 코드
if __name__ == "__main__":
    async def test_dashboard():
        """대시보드 테스트"""
        print("=== Real-time Monitoring Dashboard Test ===\n")
        
        dashboard = RealtimeMonitoringDashboard(update_interval=2)
        
        # 테스트용 메트릭 수집
        print("1. Testing metrics collection...")
        metrics = dashboard._collect_current_metrics()
        print(f"   System Status: {metrics.system_status.value}")
        print(f"   Active Alerts: {len(metrics.active_alerts)}")
        print(f"   Combined Score: {metrics.combined_metrics['overall_score']:.3f}")
        
        # 알림 시스템 테스트
        print("\n2. Testing alert system...")
        dashboard._create_alert(
            "test_alert",
            AlertLevel.WARNING,
            "test_component",
            "테스트 알림입니다",
            {"test": True}
        )
        print(f"   Alerts created: {len(dashboard.active_alerts)}")
        
        print("\n3. Dashboard ready for server start...")
        print("   Access dashboard at: http://localhost:8000")
        print("   WebSocket endpoint: ws://localhost:8000/ws")
        print("   API endpoints:")
        print("     - GET  /api/status")
        print("     - GET  /api/metrics/history")
        print("     - GET  /api/alerts") 
        print("     - POST /api/alerts/{id}/resolve")
        
        # 서버 시작 (테스트에서는 주석 처리)
        # dashboard.run_server(port=8000)
        
        dashboard.shutdown()
        print("\n✅ Real-time Monitoring Dashboard test completed")
    
    # 직접 실행시 서버 시작
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        dashboard = get_dashboard()
        dashboard.run_server(port=8000)
    else:
        # 테스트 실행
        import sys
        asyncio.run(test_dashboard())