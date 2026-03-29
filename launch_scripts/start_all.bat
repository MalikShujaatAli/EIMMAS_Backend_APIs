@echo off
title Enterprise AI Master Launcher
color 0F

echo ===================================================
echo SYSTEM STARTUP: EimmAi Master Architecture
echo ===================================================

:: Ensure we are in the launch_scripts directory
cd /d D:\oldtestpcdesktop\aa\Shahroz\FYP\launch_scripts

echo [1/4] Booting Audio Microservice (Port 8000)...
start "Audio API" cmd /c "start_audio.bat"
timeout /t 5 /nobreak >nul

echo [2/4] Booting Text Microservice (Port 8001)...
start "Text API" cmd /c "start_text.bat"
timeout /t 5 /nobreak >nul

echo [3/4] Booting Vision Microservice (Port 8002)...
start "Vision API" cmd /c "start_vision.bat"
timeout /t 5 /nobreak >nul

echo [4/4] Booting Fusion Orchestrator (Port 8003)...
start "Fusion Orchestrator" cmd /c "start_fusion.bat"

echo.
echo ===================================================
echo ✅ All Systems Online! 
echo 4 terminal windows have been opened in the background.
echo You can safely close this master launcher window.
echo ===================================================
pause