#!/bin/bash
# ============================================================================
# EC2 First-Time Setup Script
# BNM Policy RAG Agent with ChromaDB (using OpenRouter API)
# ============================================================================
#
# Usage: bash setup-ec2.sh
#
# This script:
# 1. Installs Docker and Docker Compose
# 2. Configures the user for Docker access
# 3. Creates the application directory structure
# 4. Downloads the production docker-compose file
#
# Requirements:
# - Ubuntu 22.04 LTS or Amazon Linux 2023
# - At least 4GB RAM (for ChromaDB + HuggingFace embeddings)
# - At least 10GB disk space
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "BNM RAG Agent - EC2 Setup Script"
echo "=============================================="
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS. Exiting."
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# ============================================
# Step 1: Install Docker
# ============================================
echo "[1/4] Installing Docker..."

if command -v docker &> /dev/null; then
    echo "Docker already installed: $(docker --version)"
else
    if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        # Ubuntu/Debian
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg
        sudo install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/$OS/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        sudo chmod a+r /etc/apt/keyrings/docker.gpg

        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    elif [ "$OS" = "amzn" ]; then
        # Amazon Linux
        sudo yum update -y
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Install Docker Compose
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi

    # Add current user to docker group
    sudo usermod -aG docker $USER
    echo "Docker installed successfully!"
fi

# ============================================
# Step 2: Start Docker Service
# ============================================
echo ""
echo "[2/4] Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker
echo "Docker service is running."

# ============================================
# Step 3: Create Application Directory
# ============================================
echo ""
echo "[3/4] Creating application directory..."

APP_DIR=~/bnm-rag-agent
mkdir -p $APP_DIR
cd $APP_DIR

# Create data directories
mkdir -p data chroma_db

echo "Application directory created: $APP_DIR"

# ============================================
# Step 4: Download Docker Compose File
# ============================================
echo ""
echo "[4/4] Setting up Docker Compose..."

# Create production docker-compose.yml
cat > docker-compose.yml << 'COMPOSE_EOF'
# Production Docker Compose for AWS EC2
# BNM Policy RAG Agent with ChromaDB (using OpenRouter API)

services:
  app:
    image: ghcr.io/${GITHUB_REPOSITORY:-your-username/bank-policy-rag-agent}:latest
    container_name: bnm-rag-app
    ports:
      - "7860:7860"
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
      - USE_OPENROUTER=true
    volumes:
      - app-data:/app/data
    depends_on:
      chromadb:
        condition: service_healthy
    restart: always
    networks:
      - rag-network

  chromadb:
    image: chromadb/chroma:latest
    container_name: bnm-chromadb
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: always
    networks:
      - rag-network

volumes:
  app-data:
  chroma-data:

networks:
  rag-network:
    driver: bridge
COMPOSE_EOF

# Create .env template
cat > .env.example << 'ENV_EOF'
# OpenRouter API Key (REQUIRED for LLM access)
OPENROUTER_API_KEY=your_key_here

# GitHub repository (for image pulls)
GITHUB_REPOSITORY=your-username/bank-policy-rag-agent
ENV_EOF

echo "Docker Compose file created."

# ============================================
# Final Instructions
# ============================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Log out and log back in (to apply Docker group permissions)"
echo "   Or run: newgrp docker"
echo ""
echo "2. Create your .env file:"
echo "   cd $APP_DIR"
echo "   cp .env.example .env"
echo "   nano .env  # Add your OpenRouter API key"
echo ""
echo "3. Update GITHUB_REPOSITORY in .env to your repo:"
echo "   GITHUB_REPOSITORY=your-username/bank-policy-rag-agent"
echo ""
echo "4. Log in to GitHub Container Registry:"
echo "   echo \$GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin"
echo ""
echo "5. Start the application:"
echo "   cd $APP_DIR"
echo "   docker compose up -d"
echo ""
echo "6. View logs:"
echo "   docker compose logs -f"
echo ""
echo "7. Access the app at: http://YOUR_EC2_IP:7860"
echo ""
echo "=============================================="
