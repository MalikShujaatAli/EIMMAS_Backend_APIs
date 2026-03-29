@echo off
title Fusion API Orchestrator (Port 8003)
color 0A

echo ===================================================
echo Starting Master Extraction ^& Fusion Orchestrator
echo ===================================================

:: Navigate to the fusion API directory
cd /d D:\oldtestpcdesktop\aa\Shahroz\FYP\services\fusion_api

:: Activate the virtual environment using the standard .bat activator
call venv\Scripts\activate.bat

:: Run the FastAPI application
python orchestrator_v3.py

pause