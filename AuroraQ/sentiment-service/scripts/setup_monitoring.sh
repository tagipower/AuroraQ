#!/bin/bash
# Setup 24/7 monitoring for AuroraQ Sentiment Service

echo "🚀 Setting up 24/7 monitoring for AuroraQ Sentiment Service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Copy service file
echo "📋 Installing systemd service..."
cp /opt/aurora-sentiment/scripts/aurora-sentiment.service /etc/systemd/system/
chmod 644 /etc/systemd/system/aurora-sentiment.service

# Copy monitoring script
echo "📋 Installing health monitor script..."
cp /opt/aurora-sentiment/scripts/health_monitor.sh /usr/local/bin/aurora-health-check
chmod +x /usr/local/bin/aurora-health-check

# Create log file
touch /var/log/aurora-sentiment-monitor.log
chmod 666 /var/log/aurora-sentiment-monitor.log

# Setup log rotation
echo "📋 Setting up log rotation..."
cat > /etc/logrotate.d/aurora-sentiment << EOF
/var/log/aurora-sentiment-monitor.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0666 root root
}
EOF

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable and start service
echo "✅ Enabling auto-start on boot..."
systemctl enable aurora-sentiment.service
systemctl start aurora-sentiment.service

# Setup cron job for health monitoring (every 5 minutes)
echo "⏰ Setting up health check cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * /usr/local/bin/aurora-health-check") | crontab -

# Setup daily restart (optional - at 3 AM)
echo "⏰ Setting up daily restart at 3 AM..."
(crontab -l 2>/dev/null; echo "0 3 * * * systemctl restart aurora-sentiment.service") | crontab -

# Setup disk space monitoring
echo "💾 Setting up disk space monitoring..."
cat > /usr/local/bin/aurora-disk-check << 'EOF'
#!/bin/bash
THRESHOLD=90
USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

if [ $USAGE -gt $THRESHOLD ]; then
    # Clean Docker system
    docker system prune -af --volumes
    
    # Clean old logs
    find /var/log -name "*.log" -mtime +7 -delete
    
    # Send notification
    if [ -f /opt/aurora-sentiment/.env ]; then
        source /opt/aurora-sentiment/.env
        if [ ! -z "$TELEGRAM_BOT_TOKEN" ] && [ ! -z "$TELEGRAM_CHAT_ID" ]; then
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d "chat_id=${TELEGRAM_CHAT_ID}" \
                -d "text=⚠️ Disk usage is at ${USAGE}% - Automatic cleanup performed" \
                -d "parse_mode=HTML"
        fi
    fi
fi
EOF

chmod +x /usr/local/bin/aurora-disk-check

# Add disk check to cron (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/aurora-disk-check") | crontab -

# Create service status script
echo "📊 Creating status check script..."
cat > /usr/local/bin/aurora-status << 'EOF'
#!/bin/bash
echo "=== AuroraQ Sentiment Service Status ==="
echo ""
echo "🐳 Docker Containers:"
docker ps -a | grep aurora
echo ""
echo "🔍 Service Status:"
systemctl status aurora-sentiment.service --no-pager | head -10
echo ""
echo "💾 Disk Usage:"
df -h /
echo ""
echo "📊 Memory Usage:"
free -h
echo ""
echo "🌐 Service Health:"
curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "Service not responding"
echo ""
echo "📝 Recent Logs:"
tail -5 /var/log/aurora-sentiment-monitor.log
EOF

chmod +x /usr/local/bin/aurora-status

echo ""
echo "✅ 24/7 Monitoring Setup Complete!"
echo ""
echo "📋 Summary:"
echo "  - Systemd service: aurora-sentiment.service"
echo "  - Health checks: Every 5 minutes"
echo "  - Daily restart: 3:00 AM"
echo "  - Disk cleanup: 2:00 AM (when >90% full)"
echo "  - Status command: aurora-status"
echo ""
echo "🔧 Useful commands:"
echo "  - Check status: aurora-status"
echo "  - View logs: journalctl -u aurora-sentiment -f"
echo "  - Restart service: systemctl restart aurora-sentiment"
echo "  - Stop service: systemctl stop aurora-sentiment"
echo ""