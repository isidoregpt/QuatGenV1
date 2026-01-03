@echo off
REM ============================================================================
REM Quat Generator Pro - Windows Startup Script
REM ============================================================================
REM This script starts both the backend API and frontend development server.
REM Run setup_windows.bat first if you haven't already.
REM ============================================================================

echo.
echo ============================================================
echo    Quat Generator Pro - Starting Application
echo ============================================================
echo.

REM Check if backend venv exists
if not exist "backend\venv" (
    echo [ERROR] Backend not set up. Please run setup_windows.bat first.
    pause
    exit /b 1
)

REM Check if frontend node_modules exists
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend not set up. Please run setup_windows.bat first.
    pause
    exit /b 1
)

echo Starting Backend API server...
echo   URL: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.

REM Start backend in a new window
start "Quat Generator Pro - Backend" cmd /k "cd backend && call venv\Scripts\activate.bat && python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo Starting Frontend development server...
echo   URL: http://localhost:5173
echo.

REM Start frontend in a new window
start "Quat Generator Pro - Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ============================================================
echo    Application Starting!
echo ============================================================
echo.
echo Two terminal windows have been opened:
echo   1. Backend API (http://localhost:8000)
echo   2. Frontend UI (http://localhost:5173)
echo.
echo Wait for both servers to start, then open:
echo   http://localhost:5173
echo.
echo To stop the application:
echo   Close both terminal windows, or press Ctrl+C in each.
echo.
echo First run note:
echo   The backend will download ML models on first start.
echo   This takes 10-15 minutes. Watch the backend terminal for progress.
echo.
pause
