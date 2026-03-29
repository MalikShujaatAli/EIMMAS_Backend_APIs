@echo off
title Audio Emotion API (Port 8000)
color 0B

echo ===================================================
echo Starting Audio Emotion API Microservice
echo ===================================================

:: Navigate to the audio API directory
cd /d D:\oldtestpcdesktop\aa\Shahroz\FYP\services\audio_api

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the FastAPI application
python main_audio.py

pause