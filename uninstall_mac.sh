#!/bin/bash
# ============================================================================
# Quat Generator Pro - macOS/Linux Uninstaller
# ============================================================================
# This script removes Quat Generator Pro components from your system.
# You can choose which components to remove.
# All actions are logged to uninstall_log.txt
# ============================================================================

# Store the starting directory
ROOT_DIR="$(pwd)"
LOG_FILE="${ROOT_DIR}/uninstall_log.txt"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# Logging Functions
# ============================================================================

init_log() {
    echo "============================================================================" > "$LOG_FILE"
    echo "Quat Generator Pro - Uninstall Log" >> "$LOG_FILE"
    echo "============================================================================" >> "$LOG_FILE"
    echo "Started: $(date)" >> "$LOG_FILE"
    echo "Working Directory: $ROOT_DIR" >> "$LOG_FILE"
    echo "Operating System: $(uname -s) $(uname -r)" >> "$LOG_FILE"
    echo "Architecture: $(uname -m)" >> "$LOG_FILE"
    echo "============================================================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

log() {
    echo "$1" >> "$LOG_FILE"
}

log_step() {
    echo "[STEP] $1" >> "$LOG_FILE"
}

log_ok() {
    echo "[OK] $1" >> "$LOG_FILE"
}

log_error() {
    echo "[ERROR] $1" >> "$LOG_FILE"
}

log_info() {
    echo "[INFO] $1" >> "$LOG_FILE"
}

print_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
    log_ok "$1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log_error "$1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$LOG_FILE"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
    log_info "$1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
    log_step "$1"
}

# ============================================================================
# Helper Functions
# ============================================================================

get_dir_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "Not installed"
    fi
}

confirm_action() {
    local prompt="$1"
    local response
    read -p "$prompt (yes/no): " response
    if [[ "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
        return 0
    else
        return 1
    fi
}

# ============================================================================
# Removal Functions
# ============================================================================

remove_venv() {
    echo ""
    print_step "Removing Python virtual environment..."
    
    if [ -d "backend/venv" ]; then
        rm -rf "backend/venv" 2>> "$LOG_FILE"
        if [ -d "backend/venv" ]; then
            print_error "Failed to remove backend/venv"
            echo "         It may be in use. Deactivate it first and try again."
        else
            print_ok "Removed backend/venv"
        fi
    else
        print_info "backend/venv not found - skipping"
    fi
}

remove_node_modules() {
    echo ""
    print_step "Removing Node.js dependencies..."
    
    if [ -d "frontend/node_modules" ]; then
        rm -rf "frontend/node_modules" 2>> "$LOG_FILE"
        if [ -d "frontend/node_modules" ]; then
            print_error "Failed to remove frontend/node_modules"
        else
            print_ok "Removed frontend/node_modules"
        fi
    else
        print_info "frontend/node_modules not found - skipping"
    fi
    
    # Also remove package-lock.json for clean reinstall
    if [ -f "frontend/package-lock.json" ]; then
        rm -f "frontend/package-lock.json" 2>> "$LOG_FILE"
        print_ok "Removed frontend/package-lock.json"
    fi
}

remove_data() {
    echo ""
    print_step "Removing application data..."
    
    if [ -d "backend/data" ]; then
        rm -rf "backend/data" 2>> "$LOG_FILE"
        if [ -d "backend/data" ]; then
            print_error "Failed to remove backend/data"
        else
            print_ok "Removed backend/data"
        fi
    else
        print_info "backend/data not found - skipping"
    fi
}

remove_models() {
    echo ""
    print_step "Removing local ML models..."
    
    if [ -d "backend/models" ]; then
        rm -rf "backend/models" 2>> "$LOG_FILE"
        if [ -d "backend/models" ]; then
            print_error "Failed to remove backend/models"
        else
            print_ok "Removed backend/models"
        fi
    else
        print_info "backend/models not found - skipping"
    fi
}

remove_hf_cache() {
    echo ""
    print_warning "This will remove the Hugging Face cache used by ALL applications!"
    echo ""
    
    if ! confirm_action "Are you sure?"; then
        print_info "Skipping Hugging Face cache removal"
        return
    fi
    
    print_step "Removing Hugging Face cache..."
    
    local HF_CACHE="$HOME/.cache/huggingface"
    
    if [ -d "$HF_CACHE" ]; then
        rm -rf "$HF_CACHE" 2>> "$LOG_FILE"
        if [ -d "$HF_CACHE" ]; then
            print_error "Failed to remove Hugging Face cache"
        else
            print_ok "Removed $HF_CACHE"
        fi
    else
        print_info "Hugging Face cache not found - skipping"
    fi
}

remove_logs() {
    echo ""
    print_step "Removing log files..."
    
    if [ -f "setup_log.txt" ]; then
        rm -f "setup_log.txt"
        print_ok "Removed setup_log.txt"
    fi
    
    print_info "uninstall_log.txt will be kept for your records"
}

# ============================================================================
# Main Menu
# ============================================================================

show_menu() {
    # Calculate sizes
    VENV_SIZE=$(get_dir_size "backend/venv")
    NODE_MODULES_SIZE=$(get_dir_size "frontend/node_modules")
    DATA_SIZE=$(get_dir_size "backend/data")
    MODELS_SIZE=$(get_dir_size "backend/models")
    HF_CACHE_SIZE=$(get_dir_size "$HOME/.cache/huggingface")
    
    echo ""
    echo "============================================================"
    echo "   What would you like to uninstall?"
    echo "============================================================"
    echo ""
    echo "   [1] Python Virtual Environment (backend/venv)"
    echo "       - Removes all installed Python packages"
    echo "       - Size: $VENV_SIZE"
    echo ""
    echo "   [2] Node.js Dependencies (frontend/node_modules)"
    echo "       - Removes all installed npm packages"
    echo "       - Size: $NODE_MODULES_SIZE"
    echo ""
    echo "   [3] Application Data (backend/data)"
    echo "       - Removes database, ChEMBL cache, generated molecules"
    echo "       - Size: $DATA_SIZE"
    echo ""
    echo "   [4] Downloaded ML Models (backend/models)"
    echo "       - Removes locally stored models"
    echo "       - Size: $MODELS_SIZE"
    echo ""
    echo "   [5] Hugging Face Cache (global)"
    echo "       - Removes cached ML models from ~/.cache/huggingface"
    echo "       - This affects ALL applications using Hugging Face"
    echo "       - Size: $HF_CACHE_SIZE"
    echo ""
    echo "   [6] Log Files"
    echo "       - Removes setup_log.txt"
    echo ""
    echo "   [7] EVERYTHING (Full Uninstall)"
    echo "       - Removes all of the above"
    echo ""
    echo "   [8] Custom Selection"
    echo "       - Choose multiple components individually"
    echo ""
    echo "   [0] Cancel and Exit"
    echo ""
    echo "============================================================"
}

# ============================================================================
# Main Script
# ============================================================================

# Initialize log
init_log

echo ""
echo "============================================================"
echo "   Quat Generator Pro - Uninstaller for macOS/Linux"
echo "============================================================"
echo ""
echo -e "${CYAN}[INFO]${NC} All actions will be logged to: uninstall_log.txt"
echo ""

# Verify we're in the correct directory
if [ ! -d "backend" ] && [ ! -d "frontend" ]; then
    print_error "This doesn't appear to be the QuatGenV1 directory."
    echo "Please run this script from the QuatGenV1 repository root."
    echo ""
    log_error "Neither backend nor frontend folder found"
    exit 1
fi

print_info "Found QuatGenV1 installation at: $ROOT_DIR"
echo ""

# Show menu and get choice
while true; do
    show_menu
    
    read -p "Enter your choice (0-8): " CHOICE
    
    log ""
    log "User selected option: $CHOICE"
    log ""
    
    case $CHOICE in
        0)
            echo ""
            print_info "Uninstall cancelled."
            log "Uninstall cancelled by user"
            exit 0
            ;;
        1)
            remove_venv
            break
            ;;
        2)
            remove_node_modules
            break
            ;;
        3)
            remove_data
            break
            ;;
        4)
            remove_models
            break
            ;;
        5)
            remove_hf_cache
            break
            ;;
        6)
            remove_logs
            break
            ;;
        7)
            echo ""
            echo "============================================================"
            print_warning "This will remove ALL Quat Generator Pro components!"
            echo "============================================================"
            echo ""
            echo "The following will be deleted:"
            echo "  - backend/venv (Python packages)"
            echo "  - frontend/node_modules (Node.js packages)"
            echo "  - backend/data (database, cache, molecules)"
            echo "  - backend/models (local ML models)"
            echo "  - Hugging Face cache (~1.5-2 GB)"
            echo "  - Log files"
            echo ""
            read -p "Type 'DELETE ALL' to confirm: " CONFIRM_ALL
            
            if [ "$CONFIRM_ALL" != "DELETE ALL" ]; then
                echo ""
                print_info "Uninstall cancelled."
                log "Full uninstall cancelled by user"
                break
            fi
            
            log "[STEP] Starting full uninstall..."
            
            remove_venv
            remove_node_modules
            remove_data
            remove_models
            
            # For full uninstall, skip the confirmation for HF cache
            print_step "Removing Hugging Face cache..."
            HF_CACHE="$HOME/.cache/huggingface"
            if [ -d "$HF_CACHE" ]; then
                rm -rf "$HF_CACHE" 2>> "$LOG_FILE"
                if [ -d "$HF_CACHE" ]; then
                    print_error "Failed to remove Hugging Face cache"
                else
                    print_ok "Removed $HF_CACHE"
                fi
            else
                print_info "Hugging Face cache not found - skipping"
            fi
            
            remove_logs
            
            echo ""
            print_ok "Full uninstall complete!"
            break
            ;;
        8)
            echo ""
            echo "============================================================"
            echo "   Custom Selection - Choose components to remove"
            echo "============================================================"
            echo ""
            echo "Answer yes/no for each component:"
            echo ""
            
            if confirm_action "Remove Python virtual environment?"; then
                log "[SELECTED] Python virtual environment"
                remove_venv
            fi
            
            echo ""
            if confirm_action "Remove Node.js dependencies?"; then
                log "[SELECTED] Node.js dependencies"
                remove_node_modules
            fi
            
            echo ""
            if confirm_action "Remove application data?"; then
                log "[SELECTED] Application data"
                remove_data
            fi
            
            echo ""
            if confirm_action "Remove local ML models?"; then
                log "[SELECTED] Local ML models"
                remove_models
            fi
            
            echo ""
            if confirm_action "Remove Hugging Face cache (affects all apps)?"; then
                log "[SELECTED] Hugging Face cache"
                remove_hf_cache
            fi
            
            echo ""
            if confirm_action "Remove log files?"; then
                log "[SELECTED] Log files"
                remove_logs
            fi
            
            echo ""
            print_ok "Custom uninstall complete!"
            break
            ;;
        *)
            echo ""
            print_error "Invalid choice. Please enter 0-8."
            echo ""
            ;;
    esac
done

# ============================================================
# Done
# ============================================================

echo ""
echo "============================================================"
echo "   Uninstall Complete"
echo "============================================================"
echo ""
echo "Actions logged to: $LOG_FILE"
echo ""
echo "NOTE: This uninstaller does NOT remove:"
echo "  - Python (system installation)"
echo "  - Node.js (system installation)"
echo "  - The QuatGenV1 repository folder itself"
echo ""
echo "To completely remove the application, delete the QuatGenV1 folder:"
echo "  rm -rf $ROOT_DIR"
echo ""

log ""
log "============================================================"
log "Uninstall completed: $(date)"
log "============================================================"

echo "Press Enter to exit..."
read -r
