#!/bin/bash
# AuroraQ Sentiment Service Health Monitor
# Checks service health and restarts if needed

SERVICE_URL="http://localhost:8000/health"
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
LOG_FILE="/var/log/aurora-sentiment-monitor.log"
MAX_RETRIES=3
RETRY_DELAY=30

# Load environment variables
if [ -f /opt/aurora-sentiment/.env ]; then
    source /opt/aurora-sentiment/.env
fi

# Function to send Telegram notification
send_telegram_notification() {
    local message="$1"
    if [ ! -z "$TELEGRAM_BOT_TOKEN" ] && [ ! -z "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=🚨 AuroraQ Alert: ${message}" \
            -d "parse_mode=HTML" > /dev/null 2>&1
    fi
}

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to check service health
check_health() {
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "$SERVICE_URL")
    if [ "$response" == "200" ]; then
        return 0
    else
        return 1
    fi
}

# Function to restart service
restart_service() {
    log_message "Attempting to restart sentiment service..."
    cd /opt/aurora-sentiment
    
    # Stop containers
    docker compose down
    sleep 5
    
    # Start containers
    docker compose up -d
    
    # Wait for service to come up
    sleep 30
    
    # Check if service is now healthy
    if check_health; then
        log_message "Service successfully restarted"
        send_telegram_notification "✅ Sentiment service successfully restarted after health check failure"
        return 0
    else
        log_message "Service failed to restart properly"
        send_telegram_notification "❌ Sentiment service failed to restart - manual intervention required"
        return 1
    fi
}

# Main monitoring loop
main() {
    log_message "Starting sentiment service health monitor"
    
    # Initial health check
    if ! check_health; then
        log_message "Initial health check failed"
        send_telegram_notification "⚠️ Sentiment service health check failed - attempting restart"
        
        # Try to restart with retries
        retry_count=0
        while [ $retry_count -lt $MAX_RETRIES ]; do
            if restart_service; then
                exit 0
            fi
            
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log_message "Retry $retry_count/$MAX_RETRIES in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        done
        
        log_message "All restart attempts failed"
        send_telegram_notification "🔴 CRITICAL: Sentiment service is down after $MAX_RETRIES restart attempts!"
        exit 1
    else
        log_message "Health check passed"
    fi
}

# Run main function
main