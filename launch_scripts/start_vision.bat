@echo off
title Vision API (Port 8002)
color 0E

echo ===================================================
echo Starting Image/Video Emotion API Microservice
echo ===================================================

:: Navigate to the vision API directory
cd /d D:\oldtestpcdesktop\aa\Shahroz\FYP\services\image_video_api

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the FastAPI application
python main_video.py

pause