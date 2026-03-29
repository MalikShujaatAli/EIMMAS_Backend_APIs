@echo off
title Text Emotion API (Port 8001)
color 0D

echo ===================================================
echo Starting Text Emotion API Microservice
echo ===================================================

:: Navigate to the text API directory
cd /d D:\oldtestpcdesktop\aa\Shahroz\FYP\services\text_api

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the FastAPI application using your new filename
python main_text.py

pause