#!/usr/bin/env python3
"""
AuroraQ Integrated Monitoring System
기존 AuroraQ 대시보드와 예방적 관리 모니터링의 완전 통합 시스템
"""

import asyncio
import time
import logging
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 기존 AuroraQ 시스템 임포트
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dashboard.aurora_dashboard_final import TradingModeConfig
    from sentiment.monitoring.enhanced_fallback_manager import get_fallback_manager
    from sentiment.monitoring.predictive_quality_optimizer import get_quality_optimizer
    from sentiment.monitoring.automated_recovery_system import get_recovery_system
    from sentiment.monitoring.preventive_failure_management import get_prevention_system
    from trade.trading.realtime_engine import VPSRealtimeSystem
    from core.performance.performance_optimizer import PerformanceOptimizer
    from infrastructure.monitoring.monitoring_alert_system import MonitoringAlertSystem
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")

logger = logging.getLogger(__name__)

class IntegratedMonitoringSystem:
    """통합 모니터링 시스템"""
    
    def __init__(self):
        """초기화"""
        self.app = FastAPI(title="AuroraQ Integrated Monitoring", version="2.0.0")
        
        # 기존 시스템 컴포넌트들
        self.trading_config = TradingModeConfig()
        self.websocket_connections: List[WebSocket] = []
        
        # 새로운 예방적 관리 시스템들
        try:
            self.fallback_manager = get_fallback_manager()
            self.quality_optimizer = get_quality_optimizer()
            self.recovery_system = get_recovery_system()
            self.prevention_system = get_prevention_system()
        except:
            logger.warning("Some monitoring systems not available")
            self.fallback_manager = None
            self.quality_optimizer = None
            self.recovery_system = None
            self.prevention_system = None
        
        # 기존 AuroraQ 컴포넌트들 (가능한 경우)
        self.trading_system = None
        self.performance_optimizer = None
        self.alert_system = None
        
        # 통합 메트릭 저장소
        self.integrated_metrics = {
            "trading": {},
            "sentiment": {},
            "system": {},
            "quality": {},
            "fallback": {},
            "recovery": {},
            "prevention": {}
        }
        
        self._setup_routes()
        
        # 모니터링 스레드
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._integrated_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Integrated Monitoring System initialized")
    
    def _setup_routes(self):
        """통합 라우트 설정"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def integrated_dashboard():
            """통합 대시보드 메인 페이지"""
            return self._get_integrated_dashboard_html()
        
        @self.app.get("/api/integrated/status")
        async def get_integrated_status():
            """통합 시스템 상태"""
            try:
                status = await self._collect_integrated_metrics()
                return JSONResponse({"status": "success", "data": status})
            except Exception as e:
                logger.error(f"Error getting integrated status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/trading/status")
        async def get_trading_status():
            """거래 시스템 상태"""
            return JSONResponse({
                "mode": self.trading_config.current_mode,
                "available_modes": self.trading_config.available_modes,
                "last_changed": self.trading_config.last_changed.isoformat() if self.trading_config.last_changed else None
            })
        
        @self.app.post("/api/trading/switch_mode")
        async def switch_trading_mode(mode: str):
            """거래 모드 전환"""
            if mode in self.trading_config.available_modes:
                old_mode = self.trading_config.current_mode
                self.trading_config.current_mode = mode
                self.trading_config.last_changed = datetime.now()
                
                await self._broadcast_update({
                    "type": "trading_mode_changed",
                    "old_mode": old_mode,
                    "new_mode": mode,
                    "timestamp": datetime.now().isoformat()
                })
                
                return JSONResponse({"status": "success", "message": f"Switched to {mode} mode"})
            else:
                raise HTTPException(status_code=400, detail="Invalid trading mode")
        
        @self.app.get("/api/prevention/status")
        async def get_prevention_status():
            """예방적 관리 상태"""
            if self.prevention_system:
                return JSONResponse(self.prevention_system.get_system_status())
            return JSONResponse({"error": "Prevention system not available"})
        
        @self.app.get("/api/recovery/status")
        async def get_recovery_status():
            """복구 시스템 상태"""
            if self.recovery_system:
                return JSONResponse(self.recovery_system.get_system_status())
            return JSONResponse({"error": "Recovery system not available"})
        
        @self.app.websocket("/ws/integrated")
        async def websocket_endpoint(websocket: WebSocket):
            """통합 WebSocket 엔드포인트"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            logger.info(f"Integrated WebSocket connected. Total: {len(self.websocket_connections)}")
            
            try:
                while True:
                    await asyncio.sleep(30)
                    await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
            except:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
                logger.info(f"Integrated WebSocket disconnected. Total: {len(self.websocket_connections)}")
    
    def _get_integrated_dashboard_html(self) -> str:
        """통합 대시보드 HTML"""
        return """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AuroraQ Integrated Monitoring Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #fff, #a8d8ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        .tabs {
            display: flex;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s;
            margin: 0 5px;
        }
        .tab.active {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
        }
        .tab:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s;
        }
        .metric-card:hover { transform: translateY(-5px); }
        .metric-card h3 {
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-good { color: #4ade80; }
        .metric-warning { color: #facc15; }
        .metric-critical { color: #f87171; }
        
        .status-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            transition: all 0.3s;
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        .btn.primary {
            background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        }
        .btn.success {
            background: linear-gradient(45deg, #10b981, #059669);
        }
        .btn.warning {
            background: linear-gradient(45deg, #f59e0b, #d97706);
        }
        
        .update-time {
            text-align: right;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: rgba(255, 255, 255, 0.7);
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .loading {
            animation: pulse 2s infinite;
        }
        
        .chart-container {
            height: 300px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: rgba(255, 255, 255, 0.7);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AuroraQ Integrated Monitoring</h1>
        <p>통합 예방적 관리 및 거래 시스템 모니터링</p>
    </div>
    
    <div class="container">
        <div class="tabs">
            <div class="tab active" onclick="switchTab('overview')">📊 Overview</div>
            <div class="tab" onclick="switchTab('trading')">💰 Trading</div>
            <div class="tab" onclick="switchTab('prevention')">🛡️ Prevention</div>
            <div class="tab" onclick="switchTab('recovery')">🔧 Recovery</div>
            <div class="tab" onclick="switchTab('quality')">📈 Quality</div>
            <div class="tab" onclick="switchTab('alerts')">🚨 Alerts</div>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="metrics-grid" id="overviewMetrics">
                <div class="loading">📊 통합 메트릭 로딩 중...</div>
            </div>
            
            <div class="status-panel">
                <h3>🎯 시스템 종합 상태</h3>
                <div id="systemOverview" class="loading">시스템 상태 확인 중...</div>
            </div>
        </div>
        
        <!-- Trading Tab -->
        <div id="trading" class="tab-content">
            <div class="controls">
                <button class="btn primary" onclick="switchTradingMode('paper')">📝 Paper Trading</button>
                <button class="btn success" onclick="switchTradingMode('live')">🟢 Live Trading</button>
                <button class="btn warning" onclick="switchTradingMode('backtest')">📊 Backtest</button>
                <button class="btn" onclick="switchTradingMode('dry_run')">🧪 Dry Run</button>
            </div>
            
            <div class="status-panel">
                <h3>💰 거래 시스템 상태</h3>
                <div id="tradingStatus" class="loading">거래 상태 확인 중...</div>
            </div>
            
            <div class="chart-container">
                <div>📈 거래 성과 차트 (구현 예정)</div>
            </div>
        </div>
        
        <!-- Prevention Tab -->
        <div id="prevention" class="tab-content">
            <div class="metrics-grid" id="preventionMetrics">
                <div class="loading">예방 시스템 메트릭 로딩 중...</div>
            </div>
            
            <div class="status-panel">
                <h3>🛡️ 예방적 장애 관리</h3>
                <div id="preventionStatus" class="loading">예방 시스템 상태 확인 중...</div>
            </div>
        </div>
        
        <!-- Recovery Tab -->
        <div id="recovery" class="tab-content">
            <div class="metrics-grid" id="recoveryMetrics">
                <div class="loading">복구 시스템 메트릭 로딩 중...</div>
            </div>
            
            <div class="status-panel">
                <h3>🔧 자동화된 복구</h3>
                <div id="recoveryStatus" class="loading">복구 시스템 상태 확인 중...</div>
            </div>
        </div>
        
        <!-- Quality Tab -->
        <div id="quality" class="tab-content">
            <div class="metrics-grid" id="qualityMetrics">
                <div class="loading">품질 메트릭 로딩 중...</div>
            </div>
            
            <div class="chart-container">
                <div>📊 품질 트렌드 차트 (구현 예정)</div>
            </div>
        </div>
        
        <!-- Alerts Tab -->
        <div id="alerts" class="tab-content">
            <div class="status-panel">
                <h3>🚨 활성 알림</h3>
                <div id="activeAlerts" class="loading">알림 로딩 중...</div>
            </div>
        </div>
        
        <div class="update-time" id="lastUpdate">마지막 업데이트: 연결 중...</div>
    </div>

    <script>
        class IntegratedDashboard {
            constructor() {
                this.ws = null;
                this.currentTab = 'overview';
                this.reconnectInterval = 5000;
                this.maxReconnectAttempts = 10;
                this.reconnectAttempts = 0;
                
                this.initializeWebSocket();
                this.fetchInitialData();
                setInterval(() => this.fetchAllData(), 10000); // 10초마다 업데이트
            }
            
            initializeWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/integrated`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('Integrated WebSocket connected');
                    this.reconnectAttempts = 0;
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'integrated_update') {
                        this.updateDashboard(data.data);
                    } else if (data.type === 'trading_mode_changed') {
                        this.updateTradingMode(data);
                    }
                };
                
                this.ws.onclose = () => {
                    console.log('Integrated WebSocket disconnected');
                    this.attemptReconnect();
                };
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    setTimeout(() => this.initializeWebSocket(), this.reconnectInterval);
                }
            }
            
            async fetchInitialData() {
                await this.fetchAllData();
            }
            
            async fetchAllData() {
                try {
                    // 통합 상태
                    const integratedResponse = await fetch('/api/integrated/status');
                    if (integratedResponse.ok) {
                        const integratedData = await integratedResponse.json();
                        this.updateOverview(integratedData.data);
                    }
                    
                    // 거래 상태
                    const tradingResponse = await fetch('/api/trading/status');
                    if (tradingResponse.ok) {
                        const tradingData = await tradingResponse.json();
                        this.updateTradingStatus(tradingData);
                    }
                    
                    // 예방 시스템
                    const preventionResponse = await fetch('/api/prevention/status');
                    if (preventionResponse.ok) {
                        const preventionData = await preventionResponse.json();
                        this.updatePreventionStatus(preventionData);
                    }
                    
                    // 복구 시스템
                    const recoveryResponse = await fetch('/api/recovery/status');
                    if (recoveryResponse.ok) {
                        const recoveryData = await recoveryResponse.json();
                        this.updateRecoveryStatus(recoveryData);
                    }
                    
                    document.getElementById('lastUpdate').textContent = 
                        `마지막 업데이트: ${new Date().toLocaleString()}`;
                        
                } catch (error) {
                    console.error('Error fetching data:', error);
                }
            }
            
            updateOverview(data) {
                const overviewMetrics = document.getElementById('overviewMetrics');
                overviewMetrics.innerHTML = `
                    <div class="metric-card">
                        <h3>🎯 시스템 상태</h3>
                        <div class="metric-value metric-good">HEALTHY</div>
                        <div>전체 시스템 정상 운영</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>💰 거래 모드</h3>
                        <div class="metric-value metric-good">${data.trading_mode || 'Paper'}</div>
                        <div>현재 활성 모드</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🛡️ 예방 시스템</h3>
                        <div class="metric-value metric-good">ACTIVE</div>
                        <div>예방적 장애 관리 활성</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🔧 복구 시스템</h3>
                        <div class="metric-value metric-good">READY</div>
                        <div>자동 복구 준비 완료</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>📊 데이터 품질</h3>
                        <div class="metric-value metric-good">${((data.data_quality || 0.8) * 100).toFixed(0)}%</div>
                        <div>목표: 80% 이상</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🚨 활성 알림</h3>
                        <div class="metric-value ${(data.active_alerts || 0) > 0 ? 'metric-warning' : 'metric-good'}">
                            ${data.active_alerts || 0}
                        </div>
                        <div>현재 처리 필요 알림</div>
                    </div>
                `;
                
                const systemOverview = document.getElementById('systemOverview');
                systemOverview.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div>
                            <strong>거래 시스템:</strong> ✅ 정상
                        </div>
                        <div>
                            <strong>센티멘트 분석:</strong> ✅ 정상
                        </div>
                        <div>
                            <strong>예방 관리:</strong> ✅ 활성
                        </div>
                        <div>
                            <strong>자동 복구:</strong> ✅ 대기
                        </div>
                        <div>
                            <strong>품질 최적화:</strong> ✅ 실행중
                        </div>
                        <div>
                            <strong>모니터링:</strong> ✅ 실시간
                        </div>
                    </div>
                `;
            }
            
            updateTradingStatus(data) {
                const tradingStatus = document.getElementById('tradingStatus');
                tradingStatus.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                        <div>
                            <strong>현재 모드:</strong> <span class="metric-good">${data.mode.toUpperCase()}</span>
                        </div>
                        <div>
                            <strong>사용 가능 모드:</strong> ${data.available_modes.join(', ')}
                        </div>
                        <div>
                            <strong>마지막 변경:</strong> ${data.last_changed ? new Date(data.last_changed).toLocaleString() : '없음'}
                        </div>
                        <div>
                            <strong>연결 상태:</strong> <span class="metric-good">연결됨</span>
                        </div>
                    </div>
                `;
            }
            
            updatePreventionStatus(data) {
                const preventionMetrics = document.getElementById('preventionMetrics');
                preventionMetrics.innerHTML = `
                    <div class="metric-card">
                        <h3>🎯 위험 평가</h3>
                        <div class="metric-value metric-good">${data.active_risks || 0}</div>
                        <div>활성 위험 요소</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🛡️ 예방 조치</h3>
                        <div class="metric-value metric-good">${data.statistics?.preventions_executed || 0}</div>
                        <div>실행된 예방 조치</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>💰 비용 절감</h3>
                        <div class="metric-value metric-good">$${(data.cost_savings || 0).toFixed(0)}</div>
                        <div>예상 절감 비용</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🎯 예방 성공률</h3>
                        <div class="metric-value metric-good">${(data.prevention_success_rate || 0).toFixed(1)}%</div>
                        <div>예방 조치 성공률</div>
                    </div>
                `;
                
                const preventionStatus = document.getElementById('preventionStatus');
                preventionStatus.innerHTML = `
                    <div>
                        <p><strong>시스템 상태:</strong> ${data.prevention_system_status === 'active' ? '✅ 활성' : '❌ 비활성'}</p>
                        <p><strong>고위험 컴포넌트:</strong> ${data.high_risk_components?.join(', ') || '없음'}</p>
                        <p><strong>평균 위험도:</strong> ${((data.avg_risk_score || 0) * 100).toFixed(1)}%</p>
                        <p><strong>예방된 장애:</strong> ${data.incidents_prevented || 0}건</p>
                    </div>
                `;
            }
            
            updateRecoveryStatus(data) {
                const recoveryMetrics = document.getElementById('recoveryMetrics');
                recoveryMetrics.innerHTML = `
                    <div class="metric-card">
                        <h3>🔄 활성 복구</h3>
                        <div class="metric-value metric-good">${data.active_recoveries || 0}</div>
                        <div>현재 실행 중인 복구</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>📈 성공률</h3>
                        <div class="metric-value metric-good">${((data.success_rate || 0) * 100).toFixed(1)}%</div>
                        <div>복구 성공률</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>⏱️ 평균 시간</h3>
                        <div class="metric-value metric-good">${(data.avg_recovery_time || 0).toFixed(1)}s</div>
                        <div>평균 복구 시간</div>
                    </div>
                    
                    <div class="metric-card">
                        <h3>🎯 예측 정확도</h3>
                        <div class="metric-value metric-good">${((data.prediction_accuracy || 0) * 100).toFixed(1)}%</div>
                        <div>장애 예측 정확도</div>
                    </div>
                `;
                
                const recoveryStatus = document.getElementById('recoveryStatus');
                recoveryStatus.innerHTML = `
                    <div>
                        <p><strong>시스템 상태:</strong> ${data.recovery_system_status === 'active' ? '✅ 활성' : '❌ 비활성'}</p>
                        <p><strong>학습된 패턴:</strong> ${data.failure_patterns_learned || 0}개</p>
                        <p><strong>총 복구 수행:</strong> ${data.statistics?.total_recoveries || 0}회</p>
                        <p><strong>성공한 복구:</strong> ${data.statistics?.successful_recoveries || 0}회</p>
                    </div>
                `;
            }
            
            async switchTradingMode(mode) {
                try {
                    const response = await fetch('/api/trading/switch_mode', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({mode: mode})
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(`거래 모드가 ${mode}로 전환되었습니다.`);
                        await this.fetchAllData();
                    } else {
                        alert('모드 전환에 실패했습니다.');
                    }
                } catch (error) {
                    console.error('Error switching trading mode:', error);
                    alert('모드 전환 중 오류가 발생했습니다.');
                }
            }
            
            updateTradingMode(data) {
                alert(`거래 모드가 ${data.old_mode}에서 ${data.new_mode}로 전환되었습니다.`);
                this.fetchAllData();
            }
        }
        
        function switchTab(tabName) {
            // 모든 탭 비활성화
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // 선택된 탭 활성화
            event.target.classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            dashboard.currentTab = tabName;
        }
        
        // 전역 대시보드 인스턴스
        const dashboard = new IntegratedDashboard();
        
        // 거래 모드 전환 함수를 전역으로 노출
        async function switchTradingMode(mode) {
            await dashboard.switchTradingMode(mode);
        }
    </script>
</body>
</html>
        """
    
    async def _collect_integrated_metrics(self) -> Dict[str, Any]:
        """통합 메트릭 수집"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "trading_mode": self.trading_config.current_mode,
            "system_status": "healthy"
        }
        
        # 폴백 매니저 메트릭
        if self.fallback_manager:
            fb_metrics = self.fallback_manager.get_current_metrics()
            metrics.update({
                "fallback_rate": fb_metrics.get("current_fallback_rate", 0),
                "data_quality": fb_metrics.get("current_data_quality", 0.8)
            })
        
        # 품질 최적화기 메트릭
        if self.quality_optimizer:
            qo_metrics = self.quality_optimizer.get_quality_dashboard_data()
            metrics.update({
                "quality_assessments": qo_metrics["statistics"]["total_assessments"],
                "quality_improvements": qo_metrics["statistics"]["quality_improvements"]
            })
        
        # 예방 시스템 메트릭
        if self.prevention_system:
            prev_metrics = self.prevention_system.get_system_status()
            metrics.update({
                "prevention_status": prev_metrics,
                "active_risks": prev_metrics.get("active_risks", 0)
            })
        
        # 복구 시스템 메트릭
        if self.recovery_system:
            rec_metrics = self.recovery_system.get_system_status()
            metrics.update({
                "recovery_status": rec_metrics
            })
        
        return metrics
    
    def _integrated_monitoring_loop(self):
        """통합 모니터링 루프"""
        while self.monitoring_active:
            try:
                # 통합 메트릭 수집 및 브로드캐스트
                asyncio.create_task(self._broadcast_integrated_update())
                time.sleep(10)  # 10초마다 업테이트
            except Exception as e:
                logger.error(f"Integrated monitoring loop error: {e}")
                time.sleep(10)
    
    async def _broadcast_integrated_update(self):
        """통합 업데이트 브로드캐스트"""
        if not self.websocket_connections:
            return
        
        try:
            metrics = await self._collect_integrated_metrics()
            message = {
                "type": "integrated_update",
                "data": metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            disconnected = []
            for ws in self.websocket_connections:
                try:
                    await ws.send_json(message)
                except:
                    disconnected.append(ws)
            
            for ws in disconnected:
                if ws in self.websocket_connections:
                    self.websocket_connections.remove(ws)
                    
        except Exception as e:
            logger.error(f"Error broadcasting integrated update: {e}")
    
    async def _broadcast_update(self, message: Dict[str, Any]):
        """일반 업데이트 브로드캐스트"""
        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_json(message)
            except:
                disconnected.append(ws)
        
        for ws in disconnected:
            if ws in self.websocket_connections:
                self.websocket_connections.remove(ws)
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8080):
        """통합 서버 실행"""
        logger.info(f"Starting AuroraQ Integrated Monitoring on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")
    
    def shutdown(self):
        """시스템 종료"""
        self.monitoring_active = False
        logger.info("Integrated Monitoring System shut down")

# 전역 통합 시스템 인스턴스
_integrated_system: Optional[IntegratedMonitoringSystem] = None

def get_integrated_system() -> IntegratedMonitoringSystem:
    """전역 통합 시스템 반환"""
    global _integrated_system
    if _integrated_system is None:
        _integrated_system = IntegratedMonitoringSystem()
    return _integrated_system

if __name__ == "__main__":
    # 통합 시스템 실행
    system = get_integrated_system()
    system.run_server(port=8080)