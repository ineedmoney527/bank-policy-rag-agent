# AWS EC2 Setup Guide for BNM Policy RAG Agent

This guide walks you through creating an EC2 instance on AWS Console and deploying the BNM Policy RAG Agent.

## Prerequisites

- AWS Account with EC2 access
- GitHub account (for container registry)
- OpenRouter API key

---

## Part 1: Create EC2 Instance

### Step 1: Launch Instance

1. Go to [AWS Console](https://console.aws.amazon.com/) → **EC2** → **Instances** → **Launch instances**

2. **Name**: `bnm-rag-agent`

### Step 2: Choose AMI (Amazon Machine Image)

1. Select **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
   - Architecture: **64-bit (x86)**

> 💡 Ubuntu is recommended for easier Docker installation

### Step 3: Choose Instance Type

| Workload | Instance Type | vCPU | RAM | Monthly Cost* |
|----------|---------------|------|-----|---------------|
| Light (testing) | `t3.small` | 2 | 2 GB | ~$15 |
| **Recommended** | `t3.medium` | 2 | 4 GB | ~$30 |
| Heavy traffic | `t3.large` | 2 | 8 GB | ~$60 |

*Costs are approximate for us-east-1 region

> ⚠️ The app uses HuggingFace embeddings which load into memory (~500MB). Choose at least `t3.medium` for production.

### Step 4: Create Key Pair

1. Click **Create new key pair**
2. **Key pair name**: `bnm-rag-key`
3. **Key pair type**: RSA
4. **Private key file format**: `.pem` (for Mac/Linux) or `.ppk` (for Windows/PuTTY)
5. Click **Create key pair** - the file will download automatically

> 🔐 **IMPORTANT**: Save this file securely! You cannot download it again.

### Step 5: Network Settings

Click **Edit** and configure:

1. **VPC**: Default VPC (or your preferred VPC)
2. **Subnet**: No preference (auto-assign)
3. **Auto-assign public IP**: **Enable**

#### Security Group Rules

Create a new security group with these rules:

| Type | Protocol | Port Range | Source | Description |
|------|----------|------------|--------|-------------|
| SSH | TCP | 22 | My IP | SSH access |
| Custom TCP | TCP | 7860 | 0.0.0.0/0 | Gradio app |
| Custom TCP | TCP | 8000 | My IP | ChromaDB (optional) |

> 🔒 For production, restrict port 7860 to specific IPs or use a load balancer

### Step 6: Configure Storage

1. **Size**: `20 GiB` (minimum)
2. **Volume type**: `gp3` (General Purpose SSD)

> The Docker images and ChromaDB data need space. 20GB is minimum, 30GB recommended.

### Step 7: Launch Instance

1. Review your configuration
2. Click **Launch instance**
3. Wait for instance state to show **Running**

---

## Part 2: Connect to Your Instance

### Get Your Instance Details

1. Go to **EC2** → **Instances**
2. Select your instance
3. Copy the **Public IPv4 address** (e.g., `54.123.45.67`)

### Connect via SSH

**Mac/Linux:**
```bash
# Set correct permissions on key file (required)
chmod 400 ~/Downloads/bnm-rag-key.pem

# Connect to instance
ssh -i ~/Downloads/bnm-rag-key.pem ubuntu@YOUR_PUBLIC_IP
```

**Windows (PowerShell):**
```powershell
ssh -i C:\Users\YourName\Downloads\bnm-rag-key.pem ubuntu@YOUR_PUBLIC_IP
```

---

## Part 3: Run Setup Script

Once connected to your instance:

### Option A: Download and Run Script

```bash
# Download the setup script
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/bank-policy-rag-agent/main/deploy/setup-ec2.sh -o setup-ec2.sh

# Make it executable
chmod +x setup-ec2.sh

# Run the script
bash setup-ec2.sh
```

### Option B: Clone Repo and Run

```bash
# Install git if needed
sudo apt-get update && sudo apt-get install -y git

# Clone your repository
git clone https://github.com/YOUR_USERNAME/bank-policy-rag-agent.git
cd bank-policy-rag-agent

# Run setup script
bash deploy/setup-ec2.sh
```

### After Setup Completes

```bash
# IMPORTANT: Log out and log back in to apply Docker permissions
exit

# SSH back in
ssh -i ~/Downloads/bnm-rag-key.pem ubuntu@YOUR_PUBLIC_IP
```

---

## Part 4: Configure and Deploy

### Step 1: Create Environment File

```bash
cd ~/bnm-rag-agent

# Copy example env file
cp .env.example .env

# Edit with your values
nano .env
```

Add your configuration:
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
GITHUB_REPOSITORY=your-username/bank-policy-rag-agent
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

### Step 2: Login to GitHub Container Registry

1. Create a GitHub Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click **Generate new token (classic)**
   - Select scopes: `read:packages`, `write:packages`
   - Copy the token

2. Login on EC2:
```bash
echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

### Step 3: Pull and Start Application

```bash
cd ~/bnm-rag-agent

# Pull the latest image
docker compose pull

# Start in detached mode
docker compose up -d
```

### Step 4: Verify Deployment

```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f app

# Test the endpoint
curl http://localhost:7860
```

---

## Part 5: Access Your Application

Open your browser and go to:
```
http://YOUR_PUBLIC_IP:7860
```

You should see the BNM Policy RAG Agent interface!

---

## Part 6: Set Up GitHub Actions (Automated Deployment)

### Add GitHub Secrets

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value | How to Get It |
|-------------|-------|---------------|
| `EC2_HOST` | `54.123.45.67` | Your EC2 public IP |
| `EC2_USER` | `ubuntu` | Default for Ubuntu AMI |
| `EC2_SSH_KEY` | (paste entire .pem content) | Open with text editor, copy all |
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | From OpenRouter dashboard |

### For EC2_SSH_KEY

1. Open your `.pem` file with a text editor
2. Copy **everything** including:
   ```
   -----BEGIN RSA PRIVATE KEY-----
   MIIEowIBAAKCAQEA...
   ...
   -----END RSA PRIVATE KEY-----
   ```
3. Paste as the secret value

### Test Deployment

1. Make a small change to your code
2. Push to `main` branch
3. Go to **Actions** tab to watch the workflow
4. Once complete, your EC2 will have the latest version!

---

## Troubleshooting

### Cannot SSH to Instance

```bash
# Check security group allows port 22 from your IP
# Your IP might have changed - update security group "My IP"
```

### Docker Permission Denied

```bash
# Add yourself to docker group
sudo usermod -aG docker $USER

# Log out and back in
exit
ssh -i key.pem ubuntu@YOUR_IP
```

### Container Won't Start

```bash
# Check logs
docker compose logs app
docker compose logs chromadb

# Restart containers
docker compose restart
```

### Out of Disk Space

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a
```

### Application Not Accessible

1. Check security group has port 7860 open
2. Check instance has public IP
3. Verify containers are running: `docker compose ps`

---

## Cost Optimization Tips

1. **Use Spot Instances**: Up to 90% savings for non-critical workloads
2. **Stop When Not in Use**: Stop instance overnight saves ~70%
3. **Reserved Instances**: 1-year commitment for ~40% savings
4. **Elastic IP**: Allocate to avoid IP changes when stopping/starting

---

## Quick Reference Commands

```bash
# Start application
docker compose up -d

# Stop application
docker compose down

# View logs
docker compose logs -f

# Restart specific service
docker compose restart app

# Update to latest image
docker compose pull && docker compose up -d

# Check resource usage
docker stats
```
