#!/bin/bash
# ============================================================================
# Quat Generator Pro - macOS/Linux Setup Script
# ============================================================================
# This script sets up the development environment on macOS or Linux.
# Prerequisites: Python 3.11+, Node.js 18+ LTS
# ============================================================================

set -e  # Exit on error

echo ""
echo "============================================================"
echo "   Quat Generator Pro - Setup Script for macOS/Linux"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed."
    echo "Please install Python 3.11+ using:"
    echo "  macOS: brew install python@3.11"
    echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "  Fedora: sudo dnf install python3.11"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_ok "Found Python $PYTHON_VERSION"

# Check Python version is 3.11+
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    print_warning "Python 3.11+ recommended. Found $PYTHON_VERSION"
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed."
    echo "Please install Node.js LTS using:"
    echo "  macOS: brew install node"
    echo "  Ubuntu/Debian: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt install nodejs"
    echo "  Or use nvm: https://github.com/nvm-sh/nvm"
    exit 1
fi

NODE_VERSION=$(node --version)
print_ok "Found Node.js $NODE_VERSION"

# Check if npm is available
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed."
    exit 1
fi

NPM_VERSION=$(npm --version)
print_ok "Found npm $NPM_VERSION"

echo ""
echo "============================================================"
echo "   Step 1: Setting up Backend (Python)"
echo "============================================================"
echo ""

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    print_ok "Virtual environment created."
else
    print_ok "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing Python dependencies (this may take several minutes)..."
echo "Note: First run will download ML models (~1.5-2 GB)"
echo ""

# Use --break-system-packages for newer Python versions if needed
pip install -r requirements.txt || {
    print_warning "Some packages may have failed to install."
    echo "This is often due to optional dependencies."
    echo "The application may still work."
}

# Create data directories
mkdir -p data/chembl_cache
mkdir -p models

print_ok "Backend setup complete."

cd ..

echo ""
echo "============================================================"
echo "   Step 2: Setting up Frontend (Node.js)"
echo "============================================================"
echo ""

cd frontend

# Install npm dependencies
echo "Installing Node.js dependencies..."
npm install

print_ok "Frontend setup complete."

cd ..

echo ""
echo "============================================================"
echo "   Setup Complete!"
echo "============================================================"
echo ""
echo "To start the application, run: ./start_mac.sh"
echo ""
echo "First run notes:"
echo "  - Backend will download ML models on first start (~1.5-2 GB)"
echo "  - This initial download takes 10-15 minutes"
echo "  - Subsequent starts will be much faster"
echo ""
echo "Optional: For GPU acceleration (NVIDIA), install PyTorch with CUDA:"
echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
echo ""
echo "For Apple Silicon Macs, PyTorch will use MPS acceleration automatically."
echo ""
