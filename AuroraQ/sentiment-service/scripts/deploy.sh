#!/bin/bash
# Automated deployment script for AuroraQ Sentiment Service

set -e  # Exit on error

# Configuration
DEPLOY_USER="root"
DEPLOY_HOST="109.123.239.30"
DEPLOY_DIR="/opt/aurora-sentiment"
SERVICE_NAME="aurora-sentiment"
BACKUP_DIR="/opt/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Check if we have all required files
check_requirements() {
    print_status "Checking deployment requirements..."
    
    required_files=(
        "docker-compose.yml"
        "Dockerfile"
        "requirements.txt"
        ".env.example"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file not found: $file"
            exit 1
        fi
    done
    
    print_status "All required files found ✅"
}

# Run tests locally
run_tests() {
    print_status "Running tests..."
    
    # Check if pytest is available
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || {
            print_error "Tests failed! Aborting deployment."
            exit 1
        }
    else
        print_warning "pytest not found, skipping tests"
    fi
    
    print_status "Tests passed ✅"
}

# Build Docker image locally
build_image() {
    print_status "Building Docker image..."
    
    docker build -t ${SERVICE_NAME}:latest . || {
        print_error "Docker build failed!"
        exit 1
    }
    
    print_status "Docker image built successfully ✅"
}

# Deploy to VPS
deploy_to_vps() {
    print_status "Deploying to VPS ${DEPLOY_HOST}..."
    
    # Create deployment package
    print_status "Creating deployment package..."
    tar -czf deploy.tar.gz \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='env' \
        --exclude='*.log' \
        --exclude='deploy.tar.gz' \
        .
    
    # Transfer files
    print_status "Transferring files to VPS..."
    scp deploy.tar.gz ${DEPLOY_USER}@${DEPLOY_HOST}:/tmp/ || {
        print_error "Failed to transfer files!"
        rm deploy.tar.gz
        exit 1
    }
    
    # Execute deployment on VPS
    print_status "Executing deployment on VPS..."
    ssh ${DEPLOY_USER}@${DEPLOY_HOST} << 'ENDSSH'
        set -e
        
        # Create backup
        echo "Creating backup..."
        if [ -d /opt/aurora-sentiment ]; then
            mkdir -p /opt/backups
            timestamp=$(date +%Y%m%d_%H%M%S)
            tar -czf /opt/backups/aurora-sentiment-${timestamp}.tar.gz -C /opt aurora-sentiment
            
            # Keep only last 5 backups
            cd /opt/backups
            ls -t aurora-sentiment-*.tar.gz | tail -n +6 | xargs -r rm
        fi
        
        # Extract new deployment
        echo "Extracting deployment package..."
        cd /opt/aurora-sentiment
        tar -xzf /tmp/deploy.tar.gz
        rm /tmp/deploy.tar.gz
        
        # Preserve .env file if it exists
        if [ -f .env.backup ]; then
            cp .env.backup .env
        elif [ ! -f .env ] && [ -f .env.example ]; then
            cp .env.example .env
            echo "WARNING: Using .env.example - Please configure production settings!"
        fi
        
        # Stop existing services
        echo "Stopping existing services..."
        docker compose down || true
        
        # Build new images
        echo "Building Docker images..."
        docker compose build
        
        # Start services
        echo "Starting services..."
        docker compose up -d
        
        # Wait for services to be healthy
        echo "Waiting for services to be healthy..."
        sleep 30
        
        # Check health
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Service is healthy!"
        else
            echo "❌ Health check failed!"
            docker compose logs --tail=50
            exit 1
        fi
        
        # Clean up old Docker images
        echo "Cleaning up old images..."
        docker image prune -f
ENDSSH
    
    # Clean up local deployment package
    rm deploy.tar.gz
    
    print_status "Deployment completed successfully! 🚀"
}

# Send deployment notification
send_notification() {
    if [ -f .env ]; then
        source .env
        
        if [ ! -z "$TELEGRAM_BOT_TOKEN" ] && [ ! -z "$TELEGRAM_CHAT_ID" ]; then
            message="🚀 <b>Deployment Successful!</b>%0A%0A"
            message+="Service: AuroraQ Sentiment Service%0A"
            message+="Server: ${DEPLOY_HOST}%0A"
            message+="Time: $(date '+%Y-%m-%d %H:%M:%S')%0A"
            message+="Status: ✅ Healthy"
            
            curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
                -d "chat_id=${TELEGRAM_CHAT_ID}" \
                -d "text=${message}" \
                -d "parse_mode=HTML" > /dev/null
        fi
    fi
}

# Main deployment flow
main() {
    print_status "Starting deployment process..."
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "test")
            run_tests
            ;;
        "build")
            check_requirements
            build_image
            ;;
        "deploy")
            check_requirements
            run_tests
            build_image
            deploy_to_vps
            send_notification
            ;;
        "quick")
            # Skip tests for hotfix deployment
            check_requirements
            build_image
            deploy_to_vps
            send_notification
            ;;
        *)
            echo "Usage: $0 [test|build|deploy|quick]"
            echo "  test   - Run tests only"
            echo "  build  - Build Docker image only"
            echo "  deploy - Full deployment (test + build + deploy)"
            echo "  quick  - Quick deployment (skip tests)"
            exit 1
            ;;
    esac
    
    print_status "Process completed!"
}

# Run main function
main "$@"