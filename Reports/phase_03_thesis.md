# Phase 3 Thesis: Desktop Real-Time Prototypes (Hardware-Locked)

## Strategic Intent
Phase 3 asked: "Can the trained models run in real-time on live hardware input?" The goal was to prove that emotion detection could work interactively — a human speaks or shows their face, and the system responds within seconds. This was the first attempt at building a user-facing application rather than a training script. The success criterion was qualitative: does the detected emotion match what the human intended?

## Scope & Boundaries
Three scripts were developed, each bound to physical hardware peripherals. `phase03_vision_webcam_scaled.py` opened the laptop webcam via `cv2.VideoCapture(0)` and ran face emotion detection in a continuous loop with live bounding box overlays. `phase03_audio_live_vosk.py` opened the laptop microphone via `sounddevice.RawInputStream` and ran continuous speech recording with Vosk-based live subtitles, predicting emotion only after the user pressed Ctrl+C. `phase03_audio_push_to_talk.py` refined `phase03_audio_live_vosk.py` by adding push-to-talk via `keyboard.is_pressed("space")`, a volume "heartbeat bar" visualizer, a minimum speech volume threshold, and per-emotion color coding in the terminal.

## Failure Analysis
Phase 3 achieved its qualitative goal — the emotion detection "worked" in demos — but the architecture was fundamentally undeployable. All three scripts required physical hardware access (webcam, microphone, keyboard) that does not exist on cloud servers. The infinite `while True` loops blocked any possibility of serving multiple users. The `sounddevice` library captured raw PCM from the OS audio driver, making it impossible to accept uploaded audio files from a mobile app. `keyboard.is_pressed()` required root/admin privileges on Linux. There was no HTTP API, no JSON response format, no network interface. The recognition that these scripts needed to become FastAPI endpoints drove the transition to Phase 6.
