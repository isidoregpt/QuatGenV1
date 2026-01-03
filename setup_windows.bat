@echo off
REM ============================================================================
REM Quat Generator Pro - Windows Setup Script
REM ============================================================================
REM This script sets up the development environment on Windows.
REM Prerequisites: Python 3.11+, Node.js 18+ LTS
REM ============================================================================

echo.
echo ============================================================
echo    Quat Generator Pro - Setup Script for Windows
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2 delims= " %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo Please install Node.js LTS from https://nodejs.org/
    pause
    exit /b 1
)

for /f "tokens=1 delims= " %%i in ('node --version') do set NODE_VERSION=%%i
echo [OK] Found Node.js %NODE_VERSION%

REM Check if npm is available
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not installed or not in PATH.
    pause
    exit /b 1
)

for /f "tokens=1 delims= " %%i in ('npm --version') do set NPM_VERSION=%%i
echo [OK] Found npm %NPM_VERSION%

echo.
echo ============================================================
echo    Step 1: Setting up Backend (Python)
echo ============================================================
echo.

cd backend

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        cd ..
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing Python dependencies (this may take several minutes)...
echo Note: First run will download ML models (~1.5-2 GB)
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages may have failed to install.
    echo This is often due to optional dependencies.
    echo The application may still work.
)

REM Create data directories
if not exist "data" mkdir data
if not exist "data\chembl_cache" mkdir data\chembl_cache
if not exist "models" mkdir models

echo.
echo [OK] Backend setup complete.

cd ..

echo.
echo ============================================================
echo    Step 2: Setting up Frontend (Node.js)
echo ============================================================
echo.

cd frontend

REM Install npm dependencies
echo Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    cd ..
    pause
    exit /b 1
)

echo.
echo [OK] Frontend setup complete.

cd ..

echo.
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo To start the application, run: start_windows.bat
echo.
echo First run notes:
echo   - Backend will download ML models on first start (~1.5-2 GB)
echo   - This initial download takes 10-15 minutes
echo   - Subsequent starts will be much faster
echo.
echo Optional: For GPU acceleration, install PyTorch with CUDA:
echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
echo.
pause
