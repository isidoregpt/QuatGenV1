#!/bin/bash
# ============================================================================
# Quat Generator Pro - macOS/Linux Startup Script
# ============================================================================
# This script starts both the backend API and frontend development server.
# Run ./setup_mac.sh first if you haven't already.
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

echo ""
echo "============================================================"
echo "   Quat Generator Pro - Starting Application"
echo "============================================================"
echo ""

# Check if backend venv exists
if [ ! -d "backend/venv" ]; then
    print_error "Backend not set up. Please run ./setup_mac.sh first."
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    print_error "Frontend not set up. Please run ./setup_mac.sh first."
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    print_info "Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend
print_info "Starting Backend API server..."
echo "   URL: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""

cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
print_info "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    print_error "Backend failed to start. Check the logs above."
    exit 1
fi
print_ok "Backend started (PID: $BACKEND_PID)"

# Start frontend
echo ""
print_info "Starting Frontend development server..."
echo "   URL: http://localhost:5173"
echo ""

cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend failed to start. Check the logs above."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
print_ok "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "============================================================"
echo "   Application Running!"
echo "============================================================"
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Frontend UI:  http://localhost:5173"
echo ""
echo "First run note:"
echo "  The backend will download ML models on first start."
echo "  This takes 10-15 minutes. Watch the logs for progress."
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""
echo "============================================================"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
