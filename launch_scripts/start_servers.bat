@echo off
echo =======================================================
echo EIMMAS SYSTEM: STARTING MICROSERVICES (PRODUCTION MODE)
echo =======================================================

REM Execute paths relative to the bat file's physical location
set ROOT_DIR=%~dp0..
set PYTHONPATH=%ROOT_DIR%

echo [1/4] Starting Audio API (Port 8000)...
cd "%ROOT_DIR%\services\audio_api"
start /B "Audio_API" cmd /c "call venv\Scripts\activate.bat && uvicorn main_audio:app --host 0.0.0.0 --port 8000 --workers 4"

echo [2/4] Starting Text API (Port 8001)...
cd "%ROOT_DIR%\services\text_api"
start /B "Text_API" cmd /c "call venv\Scripts\activate.bat && uvicorn main_text:app --host 0.0.0.0 --port 8001 --workers 4"

echo [3/4] Starting Image/Video API (Port 8002)...
cd "%ROOT_DIR%\services\image_video_api"
start /B "Video_API" cmd /c "call venv\Scripts\activate.bat && uvicorn main_video:app --host 0.0.0.0 --port 8002 --workers 4"

echo [4/4] Starting Fusion Orchestrator (Port 8003)...
cd "%ROOT_DIR%\services\fusion_api"
start /B "Fusion_API" cmd /c "call venv\Scripts\activate.bat && uvicorn orchestrator_v3:app --host 0.0.0.0 --port 8003 --workers 4"

cd "%ROOT_DIR%\launch_scripts"
echo =======================================================
echo ALL SERVERS SEQUENCES INITIATED IN BACKGROUND.
echo You may close this window or leave it open.
echo =======================================================
pause
