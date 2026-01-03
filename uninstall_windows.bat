@echo off
REM ============================================================================
REM Quat Generator Pro - Windows Uninstaller
REM ============================================================================
REM This script removes Quat Generator Pro components from your system.
REM You can choose which components to remove.
REM All actions are logged to uninstall_log.txt
REM ============================================================================

REM Store the starting directory
set "ROOT_DIR=%CD%"
set "LOG_FILE=%ROOT_DIR%\uninstall_log.txt"

REM Initialize log file
echo ============================================================================ > "%LOG_FILE%"
echo Quat Generator Pro - Uninstall Log >> "%LOG_FILE%"
echo ============================================================================ >> "%LOG_FILE%"
echo Started: %DATE% %TIME% >> "%LOG_FILE%"
echo Working Directory: %ROOT_DIR% >> "%LOG_FILE%"
echo ============================================================================ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo.
echo ============================================================
echo    Quat Generator Pro - Uninstaller for Windows
echo ============================================================
echo.
echo [INFO] All actions will be logged to: uninstall_log.txt
echo.

REM Verify we're in the correct directory
if not exist "backend" (
    if not exist "frontend" (
        echo [ERROR] This doesn't appear to be the QuatGenV1 directory.
        echo [ERROR] Neither 'backend' nor 'frontend' folder found. >> "%LOG_FILE%"
        echo Please run this script from the QuatGenV1 repository root.
        echo.
        goto :end
    )
)

echo [INFO] Found QuatGenV1 installation at: %ROOT_DIR%
echo [INFO] Found QuatGenV1 installation at: %ROOT_DIR% >> "%LOG_FILE%"
echo.

REM ============================================================
REM Calculate sizes before showing menu
REM ============================================================

echo Calculating component sizes...
echo.

set "VENV_SIZE=Not installed"
set "NODE_MODULES_SIZE=Not installed"
set "DATA_SIZE=Not installed"
set "MODELS_SIZE=Not installed"
set "CACHE_SIZE=Unknown"

if exist "backend\venv" (
    for /f "tokens=3" %%a in ('dir /s "backend\venv" 2^>nul ^| find "File(s)"') do set "VENV_SIZE=%%a bytes"
)

if exist "frontend\node_modules" (
    for /f "tokens=3" %%a in ('dir /s "frontend\node_modules" 2^>nul ^| find "File(s)"') do set "NODE_MODULES_SIZE=%%a bytes"
)

if exist "backend\data" (
    for /f "tokens=3" %%a in ('dir /s "backend\data" 2^>nul ^| find "File(s)"') do set "DATA_SIZE=%%a bytes"
)

if exist "backend\models" (
    for /f "tokens=3" %%a in ('dir /s "backend\models" 2^>nul ^| find "File(s)"') do set "MODELS_SIZE=%%a bytes"
)

REM ============================================================
REM Display Menu
REM ============================================================

:menu
echo ============================================================
echo    What would you like to uninstall?
echo ============================================================
echo.
echo    [1] Python Virtual Environment (backend\venv)
echo        - Removes all installed Python packages
echo        - Size: %VENV_SIZE%
echo.
echo    [2] Node.js Dependencies (frontend\node_modules)
echo        - Removes all installed npm packages
echo        - Size: %NODE_MODULES_SIZE%
echo.
echo    [3] Application Data (backend\data)
echo        - Removes database, ChEMBL cache, generated molecules
echo        - Size: %DATA_SIZE%
echo.
echo    [4] Downloaded ML Models (backend\models)
echo        - Removes locally stored models
echo        - Size: %MODELS_SIZE%
echo.
echo    [5] Hugging Face Cache (global)
echo        - Removes cached ML models from %%USERPROFILE%%\.cache\huggingface
echo        - This affects ALL applications using Hugging Face
echo        - Size: ~1.5-2 GB typically
echo.
echo    [6] Log Files
echo        - Removes setup_log.txt and uninstall_log.txt
echo.
echo    [7] EVERYTHING (Full Uninstall)
echo        - Removes all of the above
echo.
echo    [8] Custom Selection
echo        - Choose multiple components individually
echo.
echo    [0] Cancel and Exit
echo.
echo ============================================================

set /p CHOICE="Enter your choice (0-8): "

echo. >> "%LOG_FILE%"
echo User selected option: %CHOICE% >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

if "%CHOICE%"=="0" goto :cancel
if "%CHOICE%"=="1" goto :remove_venv
if "%CHOICE%"=="2" goto :remove_node_modules
if "%CHOICE%"=="3" goto :remove_data
if "%CHOICE%"=="4" goto :remove_models
if "%CHOICE%"=="5" goto :remove_hf_cache
if "%CHOICE%"=="6" goto :remove_logs
if "%CHOICE%"=="7" goto :remove_all
if "%CHOICE%"=="8" goto :custom_selection

echo.
echo [ERROR] Invalid choice. Please enter 0-8.
echo.
goto :menu

REM ============================================================
REM Individual Removal Functions
REM ============================================================

:remove_venv
echo.
echo [STEP] Removing Python virtual environment...
echo [STEP] Removing Python virtual environment... >> "%LOG_FILE%"

if exist "backend\venv" (
    rmdir /s /q "backend\venv" >> "%LOG_FILE%" 2>&1
    if exist "backend\venv" (
        echo [ERROR] Failed to remove backend\venv
        echo [ERROR] Failed to remove backend\venv >> "%LOG_FILE%"
        echo         It may be in use. Close any terminals using it and try again.
    ) else (
        echo [OK] Removed backend\venv
        echo [OK] Removed backend\venv >> "%LOG_FILE%"
    )
) else (
    echo [INFO] backend\venv not found - skipping
    echo [INFO] backend\venv not found - skipping >> "%LOG_FILE%"
)

if "%BATCH_MODE%"=="1" goto :eof
goto :done

:remove_node_modules
echo.
echo [STEP] Removing Node.js dependencies...
echo [STEP] Removing Node.js dependencies... >> "%LOG_FILE%"

if exist "frontend\node_modules" (
    rmdir /s /q "frontend\node_modules" >> "%LOG_FILE%" 2>&1
    if exist "frontend\node_modules" (
        echo [ERROR] Failed to remove frontend\node_modules
        echo [ERROR] Failed to remove frontend\node_modules >> "%LOG_FILE%"
    ) else (
        echo [OK] Removed frontend\node_modules
        echo [OK] Removed frontend\node_modules >> "%LOG_FILE%"
    )
) else (
    echo [INFO] frontend\node_modules not found - skipping
    echo [INFO] frontend\node_modules not found - skipping >> "%LOG_FILE%"
)

REM Also remove package-lock.json for clean reinstall
if exist "frontend\package-lock.json" (
    del /q "frontend\package-lock.json" >> "%LOG_FILE%" 2>&1
    echo [OK] Removed frontend\package-lock.json
    echo [OK] Removed frontend\package-lock.json >> "%LOG_FILE%"
)

if "%BATCH_MODE%"=="1" goto :eof
goto :done

:remove_data
echo.
echo [STEP] Removing application data...
echo [STEP] Removing application data... >> "%LOG_FILE%"

if exist "backend\data" (
    rmdir /s /q "backend\data" >> "%LOG_FILE%" 2>&1
    if exist "backend\data" (
        echo [ERROR] Failed to remove backend\data
        echo [ERROR] Failed to remove backend\data >> "%LOG_FILE%"
    ) else (
        echo [OK] Removed backend\data
        echo [OK] Removed backend\data >> "%LOG_FILE%"
    )
) else (
    echo [INFO] backend\data not found - skipping
    echo [INFO] backend\data not found - skipping >> "%LOG_FILE%"
)

if "%BATCH_MODE%"=="1" goto :eof
goto :done

:remove_models
echo.
echo [STEP] Removing local ML models...
echo [STEP] Removing local ML models... >> "%LOG_FILE%"

if exist "backend\models" (
    rmdir /s /q "backend\models" >> "%LOG_FILE%" 2>&1
    if exist "backend\models" (
        echo [ERROR] Failed to remove backend\models
        echo [ERROR] Failed to remove backend\models >> "%LOG_FILE%"
    ) else (
        echo [OK] Removed backend\models
        echo [OK] Removed backend\models >> "%LOG_FILE%"
    )
) else (
    echo [INFO] backend\models not found - skipping
    echo [INFO] backend\models not found - skipping >> "%LOG_FILE%"
)

if "%BATCH_MODE%"=="1" goto :eof
goto :done

:remove_hf_cache
echo.
echo [WARNING] This will remove the Hugging Face cache used by ALL applications!
echo [WARNING] This will remove the Hugging Face cache >> "%LOG_FILE%"
echo.
set /p CONFIRM_HF="Are you sure? (yes/no): "

if /i not "%CONFIRM_HF%"=="yes" (
    echo [INFO] Skipping Hugging Face cache removal
    echo [INFO] User cancelled Hugging Face cache removal >> "%LOG_FILE%"
    if "%BATCH_MODE%"=="1" goto :eof
    goto :done
)

echo [STEP] Removing Hugging Face cache...
echo [STEP] Removing Hugging Face cache... >> "%LOG_FILE%"

if exist "%USERPROFILE%\.cache\huggingface" (
    rmdir /s /q "%USERPROFILE%\.cache\huggingface" >> "%LOG_FILE%" 2>&1
    if exist "%USERPROFILE%\.cache\huggingface" (
        echo [ERROR] Failed to remove Hugging Face cache
        echo [ERROR] Failed to remove Hugging Face cache >> "%LOG_FILE%"
    ) else (
        echo [OK] Removed %USERPROFILE%\.cache\huggingface
        echo [OK] Removed Hugging Face cache >> "%LOG_FILE%"
    )
) else (
    echo [INFO] Hugging Face cache not found - skipping
    echo [INFO] Hugging Face cache not found - skipping >> "%LOG_FILE%"
)

if "%BATCH_MODE%"=="1" goto :eof
goto :done

:remove_logs
echo.
echo [STEP] Removing log files...
echo [STEP] Removing log files... >> "%LOG_FILE%"

if exist "setup_log.txt" (
    del /q "setup_log.txt"
    echo [OK] Removed setup_log.txt
)

echo [INFO] uninstall_log.txt will be kept for your records
echo [INFO] uninstall_log.txt kept for records >> "%LOG_FILE%"

if "%BATCH_MODE%"=="1" goto :eof
goto :done

REM ============================================================
REM Remove Everything
REM ============================================================

:remove_all
echo.
echo ============================================================
echo [WARNING] This will remove ALL Quat Generator Pro components!
echo ============================================================
echo.
echo The following will be deleted:
echo   - backend\venv (Python packages)
echo   - frontend\node_modules (Node.js packages)
echo   - backend\data (database, cache, molecules)
echo   - backend\models (local ML models)
echo   - Hugging Face cache (~1.5-2 GB)
echo   - Log files
echo.
set /p CONFIRM_ALL="Type 'DELETE ALL' to confirm: "

if not "%CONFIRM_ALL%"=="DELETE ALL" (
    echo.
    echo [INFO] Uninstall cancelled.
    echo [INFO] Full uninstall cancelled by user >> "%LOG_FILE%"
    goto :done
)

echo.
echo [STEP] Starting full uninstall... >> "%LOG_FILE%"

set "BATCH_MODE=1"
call :remove_venv
call :remove_node_modules
call :remove_data
call :remove_models
call :remove_hf_cache
call :remove_logs
set "BATCH_MODE="

echo.
echo [OK] Full uninstall complete!
echo [OK] Full uninstall complete >> "%LOG_FILE%"
goto :done

REM ============================================================
REM Custom Selection
REM ============================================================

:custom_selection
echo.
echo ============================================================
echo    Custom Selection - Choose components to remove
echo ============================================================
echo.
echo Answer yes/no for each component:
echo.

set "BATCH_MODE=1"

set /p REMOVE_VENV="Remove Python virtual environment? (yes/no): "
if /i "%REMOVE_VENV%"=="yes" (
    echo [SELECTED] Python virtual environment >> "%LOG_FILE%"
    call :remove_venv
)

echo.
set /p REMOVE_NODE="Remove Node.js dependencies? (yes/no): "
if /i "%REMOVE_NODE%"=="yes" (
    echo [SELECTED] Node.js dependencies >> "%LOG_FILE%"
    call :remove_node_modules
)

echo.
set /p REMOVE_DATA="Remove application data? (yes/no): "
if /i "%REMOVE_DATA%"=="yes" (
    echo [SELECTED] Application data >> "%LOG_FILE%"
    call :remove_data
)

echo.
set /p REMOVE_MODELS="Remove local ML models? (yes/no): "
if /i "%REMOVE_MODELS%"=="yes" (
    echo [SELECTED] Local ML models >> "%LOG_FILE%"
    call :remove_models
)

echo.
set /p REMOVE_HF="Remove Hugging Face cache (affects all apps)? (yes/no): "
if /i "%REMOVE_HF%"=="yes" (
    echo [SELECTED] Hugging Face cache >> "%LOG_FILE%"
    call :remove_hf_cache
)

echo.
set /p REMOVE_LOGS="Remove log files? (yes/no): "
if /i "%REMOVE_LOGS%"=="yes" (
    echo [SELECTED] Log files >> "%LOG_FILE%"
    call :remove_logs
)

set "BATCH_MODE="

echo.
echo [OK] Custom uninstall complete!
echo [OK] Custom uninstall complete >> "%LOG_FILE%"
goto :done

REM ============================================================
REM Cancel
REM ============================================================

:cancel
echo.
echo [INFO] Uninstall cancelled.
echo [INFO] Uninstall cancelled by user >> "%LOG_FILE%"
goto :end

REM ============================================================
REM Done
REM ============================================================

:done
echo.
echo ============================================================
echo    Uninstall Complete
echo ============================================================
echo.
echo Actions logged to: %LOG_FILE%
echo.
echo NOTE: This uninstaller does NOT remove:
echo   - Python (system installation)
echo   - Node.js (system installation)
echo   - The QuatGenV1 repository folder itself
echo.
echo To completely remove the application, delete the QuatGenV1 folder.
echo.

echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo Uninstall completed: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

:end
echo.
echo Press any key to exit...
pause > nul
