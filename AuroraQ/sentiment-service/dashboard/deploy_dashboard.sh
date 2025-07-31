#!/bin/bash

# AuroraQ Terminal Dashboard Deployment Script
# Deploy and run the dashboard on VPS

set -e

echo "🚀 Deploying AuroraQ Terminal Dashboard..."

# Configuration
DASHBOARD_DIR="/opt/aurora-sentiment/dashboard"
SERVICE_NAME="aurora-dashboard"
PYTHON_BIN="/opt/aurora-sentiment/venv/bin/python"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Create dashboard directory
echo "📁 Creating dashboard directory..."
mkdir -p $DASHBOARD_DIR

# Copy dashboard files
echo "📋 Copying dashboard files..."
cp aurora_terminal_dashboard.py $DASHBOARD_DIR/
cp aurora_dashboard_advanced.py $DASHBOARD_DIR/
cp dashboard_config.json $DASHBOARD_DIR/

# Install additional dependencies
echo "📦 Installing dependencies..."
$PYTHON_BIN -m pip install wcwidth colorama psutil

# Create systemd service
echo "⚙️ Creating systemd service..."
cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=AuroraQ Terminal Dashboard
After=network.target redis.service

[Service]
Type=simple
User=root
WorkingDirectory=$DASHBOARD_DIR
Environment="PYTHONUNBUFFERED=1"
ExecStart=$PYTHON_BIN $DASHBOARD_DIR/aurora_dashboard_advanced.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create run script for terminal
echo "📝 Creating terminal run script..."
cat > $DASHBOARD_DIR/run_dashboard.sh << 'EOF'
#!/bin/bash
# Run dashboard in current terminal

cd /opt/aurora-sentiment/dashboard

# Check if in SSH session
if [ -n "$SSH_TTY" ]; then
    # Set terminal for better compatibility
    export TERM=xterm-256color
fi

# Run with proper Python environment
/opt/aurora-sentiment/venv/bin/python aurora_dashboard_advanced.py
EOF

chmod +x $DASHBOARD_DIR/run_dashboard.sh

# Create screen session script
echo "📺 Creating screen session script..."
cat > $DASHBOARD_DIR/run_in_screen.sh << 'EOF'
#!/bin/bash
# Run dashboard in screen session

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    apt-get update && apt-get install -y screen
fi

# Kill existing dashboard screen session if exists
screen -S aurora-dashboard -X quit 2>/dev/null

# Start new screen session
screen -dmS aurora-dashboard bash -c "cd /opt/aurora-sentiment/dashboard && /opt/aurora-sentiment/venv/bin/python aurora_dashboard_advanced.py"

echo "Dashboard started in screen session 'aurora-dashboard'"
echo "To attach: screen -r aurora-dashboard"
echo "To detach: Ctrl+A then D"
EOF

chmod +x $DASHBOARD_DIR/run_in_screen.sh

# Create tmux session script
echo "🖥️ Creating tmux session script..."
cat > $DASHBOARD_DIR/run_in_tmux.sh << 'EOF'
#!/bin/bash
# Run dashboard in tmux session

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux..."
    apt-get update && apt-get install -y tmux
fi

# Kill existing dashboard tmux session if exists
tmux kill-session -t aurora-dashboard 2>/dev/null

# Start new tmux session
tmux new-session -d -s aurora-dashboard "cd /opt/aurora-sentiment/dashboard && /opt/aurora-sentiment/venv/bin/python aurora_dashboard_advanced.py"

echo "Dashboard started in tmux session 'aurora-dashboard'"
echo "To attach: tmux attach -t aurora-dashboard"
echo "To detach: Ctrl+B then D"
EOF

chmod +x $DASHBOARD_DIR/run_in_tmux.sh

# Update configuration with VPS settings
echo "⚙️ Updating configuration..."
cat > $DASHBOARD_DIR/dashboard_config.json << EOF
{
  "api_url": "http://localhost:8080",
  "redis_url": "redis://localhost:6379",
  "update_interval": 1,
  "enable_animations": true,
  "enable_alerts": true,
  "enable_sound": false,
  "thresholds": {
    "sentiment_extreme": 0.8,
    "sentiment_rapid_change": 0.3,
    "cpu_high": 80,
    "memory_high": 85,
    "api_slow": 500,
    "cache_hit_low": 60
  },
  "display": {
    "width": 120,
    "height": 40,
    "refresh_rate": 1,
    "color_scheme": "dos",
    "font_size": "auto"
  }
}
EOF

# Create convenience aliases
echo "🔗 Creating convenience commands..."
cat >> ~/.bashrc << 'EOF'

# AuroraQ Dashboard aliases
alias aurora-dash='cd /opt/aurora-sentiment/dashboard && ./run_dashboard.sh'
alias aurora-dash-screen='cd /opt/aurora-sentiment/dashboard && ./run_in_screen.sh'
alias aurora-dash-tmux='cd /opt/aurora-sentiment/dashboard && ./run_in_tmux.sh'
alias aurora-dash-attach='screen -r aurora-dashboard || tmux attach -t aurora-dashboard'
EOF

# Reload systemd
systemctl daemon-reload

echo "✅ Dashboard deployment complete!"
echo ""
echo "📌 Usage Options:"
echo ""
echo "1. Run directly in terminal:"
echo "   aurora-dash"
echo ""
echo "2. Run in screen (recommended for SSH):"
echo "   aurora-dash-screen"
echo "   screen -r aurora-dashboard  # to attach"
echo ""
echo "3. Run in tmux:"
echo "   aurora-dash-tmux"
echo "   tmux attach -t aurora-dashboard  # to attach"
echo ""
echo "4. Run as system service:"
echo "   systemctl start $SERVICE_NAME"
echo "   systemctl enable $SERVICE_NAME  # auto-start on boot"
echo ""
echo "5. View logs:"
echo "   journalctl -u $SERVICE_NAME -f"
echo ""
echo "Note: For best experience over SSH, use screen or tmux!"