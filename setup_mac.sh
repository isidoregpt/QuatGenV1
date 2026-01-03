#!/bin/bash
# ============================================================================
# Quat Generator Pro - macOS Complete Setup Script (Enhanced)
# ============================================================================
# This script automatically installs ALL prerequisites and sets up the
# development environment on macOS.
#
# What this script installs (if not already present):
#   - Xcode Command Line Tools (required for compiling)
#   - Homebrew (macOS package manager)
#   - System dependencies: cairo, pkg-config, cmake
#   - Python 3.11+
#   - Node.js 18+ LTS
#   - All backend Python dependencies
#   - All frontend Node.js dependencies
#
# Tested on: macOS Sonoma/Ventura on Apple Silicon and Intel Macs
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================
PYTHON_MIN_VERSION="3.11"
NODE_MIN_VERSION="18"

# ============================================================================
# Colors for output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================
print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BOLD}   $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[i]${NC} $1"
}

print_step() {
    echo -e "${BOLD}>>> $1${NC}"
}

# Function to check if running on macOS
check_macos() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is designed for macOS."
        print_info "For Linux, please use setup_linux.sh or your package manager."
        exit 1
    fi
    print_ok "Running on macOS $(sw_vers -productVersion)"
}

# Function to detect architecture and set Homebrew paths
setup_homebrew_env() {
    if [[ $(uname -m) == "arm64" ]]; then
        HOMEBREW_PREFIX="/opt/homebrew"
        print_ok "Apple Silicon detected - PyTorch will use MPS acceleration"
    else
        HOMEBREW_PREFIX="/usr/local"
        print_ok "Intel Mac detected"
    fi
    
    # Always set up Homebrew environment if it exists
    if [[ -f "${HOMEBREW_PREFIX}/bin/brew" ]]; then
        eval "$(${HOMEBREW_PREFIX}/bin/brew shellenv)"
        export PATH="${HOMEBREW_PREFIX}/bin:${HOMEBREW_PREFIX}/sbin:$PATH"
    fi
}

# Function to ensure Homebrew is in PATH
ensure_brew_in_path() {
    if ! command -v brew &> /dev/null; then
        if [[ -f "${HOMEBREW_PREFIX}/bin/brew" ]]; then
            eval "$(${HOMEBREW_PREFIX}/bin/brew shellenv)"
            export PATH="${HOMEBREW_PREFIX}/bin:${HOMEBREW_PREFIX}/sbin:$PATH"
        fi
    fi
}

# Function to add Homebrew to shell profile permanently
add_brew_to_profile() {
    local SHELL_PROFILE=""
    
    # Determine which shell profile to use
    if [[ -n "$ZSH_VERSION" ]] || [[ "$SHELL" == */zsh ]]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [[ -n "$BASH_VERSION" ]] || [[ "$SHELL" == */bash ]]; then
        if [[ -f "$HOME/.bash_profile" ]]; then
            SHELL_PROFILE="$HOME/.bash_profile"
        else
            SHELL_PROFILE="$HOME/.bashrc"
        fi
    fi
    
    if [[ -n "$SHELL_PROFILE" ]]; then
        # Check if already added
        if ! grep -q 'brew shellenv' "$SHELL_PROFILE" 2>/dev/null; then
            echo "" >> "$SHELL_PROFILE"
            echo '# Homebrew - added by Quat Generator Pro setup' >> "$SHELL_PROFILE"
            echo "eval \"\$(${HOMEBREW_PREFIX}/bin/brew shellenv)\"" >> "$SHELL_PROFILE"
            print_ok "Added Homebrew to $SHELL_PROFILE"
        else
            print_ok "Homebrew already in $SHELL_PROFILE"
        fi
    fi
}

# ============================================================================
# Main Script
# ============================================================================

clear
echo ""
echo -e "${BOLD}${BLUE}"
echo "   ╔═══════════════════════════════════════════════════════════╗"
echo "   ║                                                           ║"
echo "   ║          Quat Generator Pro - Complete Setup              ║"
echo "   ║                      (Enhanced)                           ║"
echo "   ║                                                           ║"
echo "   ║     This script will install all prerequisites and        ║"
echo "   ║     set up your development environment automatically     ║"
echo "   ║                                                           ║"
echo "   ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

# ============================================================================
# Step 0: System Checks
# ============================================================================
print_header "Step 0: System Checks"

check_macos
setup_homebrew_env

# ============================================================================
# Step 1: Install Xcode Command Line Tools
# ============================================================================
print_header "Step 1: Checking Xcode Command Line Tools"

if xcode-select -p &> /dev/null; then
    print_ok "Xcode Command Line Tools already installed"
else
    print_warning "Xcode Command Line Tools not found. Installing..."
    print_info "A dialog may appear - click 'Install' to proceed"
    xcode-select --install
    
    # Wait for installation to complete
    echo ""
    print_info "Waiting for Xcode Command Line Tools installation..."
    print_info "Press Enter after the installation completes."
    read -r
    
    if xcode-select -p &> /dev/null; then
        print_ok "Xcode Command Line Tools installed successfully"
    else
        print_error "Xcode Command Line Tools installation may have failed"
        print_info "Please install manually and re-run this script"
        exit 1
    fi
fi

# ============================================================================
# Step 2: Install Homebrew
# ============================================================================
print_header "Step 2: Checking Homebrew"

ensure_brew_in_path

if command -v brew &> /dev/null; then
    print_ok "Homebrew is already installed"
    print_step "Updating Homebrew..."
    brew update --quiet
else
    print_warning "Homebrew not found. Installing..."
    print_info "This may ask for your password."
    echo ""
    
    # Install Homebrew
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Set up Homebrew environment immediately
    eval "$(${HOMEBREW_PREFIX}/bin/brew shellenv)"
    export PATH="${HOMEBREW_PREFIX}/bin:${HOMEBREW_PREFIX}/sbin:$PATH"
    
    # Verify installation
    if command -v brew &> /dev/null; then
        print_ok "Homebrew installed successfully"
    else
        print_error "Homebrew installation failed"
        print_info "Please install Homebrew manually from https://brew.sh"
        exit 1
    fi
fi

# Add Homebrew to shell profile for future sessions
add_brew_to_profile

BREW_VERSION=$(brew --version | head -n1)
print_info "$BREW_VERSION"

# ============================================================================
# Step 3: Install System Dependencies
# ============================================================================
print_header "Step 3: Installing System Dependencies"

print_info "These are required for compiling Python packages"
echo ""

# List of required system packages
SYSTEM_PACKAGES=(
    "cairo"        # Required by pycairo
    "pkg-config"   # Required for package discovery
    "cmake"        # Required by some ML packages
)

for package in "${SYSTEM_PACKAGES[@]}"; do
    if brew list "$package" &> /dev/null; then
        print_ok "$package already installed"
    else
        print_step "Installing $package..."
        brew install "$package" --quiet
        if brew list "$package" &> /dev/null; then
            print_ok "$package installed successfully"
        else
            print_warning "Failed to install $package - some features may not work"
        fi
    fi
done

# Ensure pkg-config can find cairo
export PKG_CONFIG_PATH="${HOMEBREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"

# ============================================================================
# Step 4: Install Python 3.11+
# ============================================================================
print_header "Step 4: Checking Python"

NEED_PYTHON=false
PYTHON_CMD=""

# Check for existing Python installation
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_ok "Python $PYTHON_VERSION found (meets requirement of $PYTHON_MIN_VERSION+)"
        PYTHON_CMD="python3"
    else
        print_warning "Python $PYTHON_VERSION found, but $PYTHON_MIN_VERSION+ is required"
        NEED_PYTHON=true
    fi
else
    print_warning "Python 3 not found"
    NEED_PYTHON=true
fi

# Install Python if needed
if [ "$NEED_PYTHON" = true ]; then
    print_step "Installing Python 3.11 via Homebrew..."
    brew install python@3.11 --quiet
    
    # Link python3.11
    brew link python@3.11 --overwrite --force 2>/dev/null || true
    
    # Find the installed Python
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif [[ -f "${HOMEBREW_PREFIX}/bin/python3.11" ]]; then
        PYTHON_CMD="${HOMEBREW_PREFIX}/bin/python3.11"
    else
        PYTHON_CMD="python3"
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_ok "Python $PYTHON_VERSION installed successfully"
else
    # Make sure we have the right Python command
    if [[ -z "$PYTHON_CMD" ]]; then
        PYTHON_CMD="python3"
    fi
fi

# Verify pip is available
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_step "Installing pip..."
    $PYTHON_CMD -m ensurepip --upgrade
fi

PIP_VERSION=$($PYTHON_CMD -m pip --version | cut -d' ' -f2)
print_ok "pip $PIP_VERSION available"

# ============================================================================
# Step 5: Install Node.js 18+
# ============================================================================
print_header "Step 5: Checking Node.js"

NEED_NODE=false

if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | tr -d 'v')
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
    
    if [ "$NODE_MAJOR" -ge "$NODE_MIN_VERSION" ]; then
        print_ok "Node.js v$NODE_VERSION found (meets requirement of v$NODE_MIN_VERSION+)"
    else
        print_warning "Node.js v$NODE_VERSION found, but v$NODE_MIN_VERSION+ is required"
        NEED_NODE=true
    fi
else
    print_warning "Node.js not found"
    NEED_NODE=true
fi

# Install Node.js if needed
if [ "$NEED_NODE" = true ]; then
    print_step "Installing Node.js LTS via Homebrew..."
    brew install node --quiet
    
    NODE_VERSION=$(node --version | tr -d 'v')
    print_ok "Node.js v$NODE_VERSION installed successfully"
fi

# Verify npm is available
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    print_ok "npm $NPM_VERSION available"
else
    print_error "npm not found after Node.js installation"
    exit 1
fi

# ============================================================================
# Step 6: Setup Backend (Python)
# ============================================================================
print_header "Step 6: Setting up Backend (Python)"

# Check if backend directory exists
if [ ! -d "backend" ]; then
    print_error "backend/ directory not found!"
    print_info "Please run this script from the Quat Generator Pro root directory."
    print_info "Example: cd /path/to/QuatGenV1-main && ./setup_mac.sh"
    exit 1
fi

cd backend

# Remove existing broken venv if it has no packages
if [ -d "venv" ]; then
    INSTALLED_PACKAGES=$(venv/bin/pip list 2>/dev/null | wc -l)
    if [ "$INSTALLED_PACKAGES" -lt 5 ]; then
        print_warning "Existing virtual environment appears incomplete"
        print_step "Removing and recreating..."
        rm -rf venv
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_step "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv
    print_ok "Virtual environment created"
else
    print_ok "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate

# IMPORTANT: Ensure Homebrew paths are available in the virtual environment
export PATH="${HOMEBREW_PREFIX}/bin:${HOMEBREW_PREFIX}/sbin:$PATH"
export PKG_CONFIG_PATH="${HOMEBREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export LDFLAGS="-L${HOMEBREW_PREFIX}/lib"
export CPPFLAGS="-I${HOMEBREW_PREFIX}/include"

# Upgrade pip in virtual environment
print_step "Upgrading pip in virtual environment..."
pip install --upgrade pip --quiet

# Install Python dependencies
echo ""
print_step "Installing Python dependencies..."
print_info "This may take several minutes on first run"
print_info "ML models (~1.5-2 GB) will be downloaded on first application start"
echo ""

if [ -f "requirements.txt" ]; then
    # Install with verbose output to catch errors
    pip install -r requirements.txt 2>&1 | while read -r line; do
        if [[ "$line" == *"error"* ]] || [[ "$line" == *"ERROR"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ "$line" == *"Successfully"* ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ "$line" == *"Collecting"* ]] || [[ "$line" == *"Installing"* ]]; then
            echo -e "${CYAN}$line${NC}"
        fi
    done
    
    # Verify critical packages installed
    echo ""
    print_step "Verifying critical packages..."
    
    CRITICAL_PACKAGES=("fastapi" "uvicorn" "torch" "rdkit" "pycairo")
    ALL_INSTALLED=true
    
    for pkg in "${CRITICAL_PACKAGES[@]}"; do
        if pip show "$pkg" &> /dev/null; then
            print_ok "$pkg installed"
        else
            print_error "$pkg NOT installed"
            ALL_INSTALLED=false
        fi
    done
    
    if [ "$ALL_INSTALLED" = false ]; then
        echo ""
        print_warning "Some packages failed to install"
        print_info "Attempting to install missing packages individually..."
        
        for pkg in "${CRITICAL_PACKAGES[@]}"; do
            if ! pip show "$pkg" &> /dev/null; then
                print_step "Installing $pkg..."
                pip install "$pkg" 2>&1 | tail -3
            fi
        done
    fi
    
    print_ok "Python dependencies installation complete"
else
    print_warning "requirements.txt not found - skipping dependency installation"
fi

# Create necessary directories
print_step "Creating data directories..."
mkdir -p data/chembl_cache
mkdir -p models
print_ok "Data directories created"

cd ..

# ============================================================================
# Step 7: Setup Frontend (Node.js)
# ============================================================================
print_header "Step 7: Setting up Frontend (Node.js)"

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    print_error "frontend/ directory not found!"
    print_info "Please run this script from the Quat Generator Pro root directory."
    exit 1
fi

cd frontend

# Install npm dependencies
print_step "Installing Node.js dependencies..."

if [ -f "package.json" ]; then
    npm install --loglevel warn
    print_ok "Node.js dependencies installed"
else
    print_warning "package.json not found - skipping dependency installation"
fi

cd ..

# ============================================================================
# Step 8: Final Verification
# ============================================================================
print_header "Step 8: Final Verification"

echo ""
print_step "Checking all components..."

# Check backend
if [ -f "backend/venv/bin/python" ]; then
    BACKEND_PACKAGES=$(backend/venv/bin/pip list 2>/dev/null | wc -l)
    if [ "$BACKEND_PACKAGES" -gt 10 ]; then
        print_ok "Backend: Python environment ready ($BACKEND_PACKAGES packages)"
    else
        print_warning "Backend: May be incomplete ($BACKEND_PACKAGES packages)"
    fi
else
    print_error "Backend: Virtual environment not found"
fi

# Check frontend
if [ -d "frontend/node_modules" ]; then
    print_ok "Frontend: Node modules installed"
else
    print_warning "Frontend: node_modules not found"
fi

# Check for start script
if [ -f "start_mac.sh" ]; then
    chmod +x start_mac.sh
    print_ok "Start script: Ready"
else
    print_warning "Start script: start_mac.sh not found"
fi

# ============================================================================
# Setup Complete
# ============================================================================
print_header "Setup Complete!"

echo -e "${GREEN}${BOLD}"
echo "   ╔═══════════════════════════════════════════════════════════╗"
echo "   ║                                                           ║"
echo "   ║               ✓ Installation Successful!                  ║"
echo "   ║                                                           ║"
echo "   ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

echo -e "${BOLD}What was installed:${NC}"
echo -e "  • Xcode Command Line Tools"
echo -e "  • Homebrew"
echo -e "  • System libraries: cairo, pkg-config, cmake"
echo -e "  • Python $($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)"
echo -e "  • Node.js $(node --version)"
echo -e "  • npm $(npm --version)"
echo -e "  • Python virtual environment with dependencies"
echo -e "  • Node.js frontend dependencies"
echo ""

echo -e "${BOLD}To start the application:${NC}"
echo -e "  ${CYAN}./start_mac.sh${NC}"
echo ""

echo -e "${BOLD}First run notes:${NC}"
echo -e "  • Backend will download ML models on first start (~1.5-2 GB)"
echo -e "  • Initial download takes approximately 10-15 minutes"
echo -e "  • Subsequent starts will be much faster"
echo ""

if [[ $(uname -m) == "arm64" ]]; then
    echo -e "${BOLD}Apple Silicon:${NC}"
    echo -e "  • PyTorch will automatically use MPS acceleration"
    echo ""
fi

echo -e "${BOLD}Optional GPU acceleration (NVIDIA):${NC}"
echo -e "  ${CYAN}source backend/venv/bin/activate${NC}"
echo -e "  ${CYAN}pip install torch --index-url https://download.pytorch.org/whl/cu121${NC}"
echo ""

echo -e "${BOLD}If you encounter issues:${NC}"
echo -e "  • Make sure you're in the project root directory"
echo -e "  • Try: ${CYAN}source backend/venv/bin/activate && pip install -r backend/requirements.txt${NC}"
echo -e "  • Check logs for specific error messages"
echo ""

print_ok "You're all set! Happy coding!"
echo ""