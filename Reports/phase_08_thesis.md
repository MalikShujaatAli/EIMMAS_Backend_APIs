# Phase 8 Thesis: The Pre-Production API Bridge

## Core Objective
The primary aim of Phase 8 was to refactor the separated, flawed APIs of Phase 7 into stable, deployable microservices before introducing the complexities of a central orchestrator. The goal was to eliminate catastrophic latency bottlenecks (specifically in video processing) and remove hard disk I/O dependency for media files.

## Architectural State
- **Decoupled Pre-Production**: Three independent FastAPI applications (`main_audio.py`, `main_video.py`, `main_text.py` from `FYP old/`) running on separate ports (8000, 8002, 8001).
- **No Orchestration**: External clients (or testing scripts) had to communicate with each API directly. No centralized fusion, authentication, or LLM chat existed.
- **In-Memory Transformation**: Audio processing successfully shifted from `NamedTemporaryFile` disk writes to `io.BytesIO` streams, marking the first time the system achieved "zero-disk" operation for a modality.

## Pivotal Technical Inventions
1. **Dual-Threshold Filter (Text)**: Transitioned from hardcoded emotion keyword matching to a dynamic confidence filter (`max_prob < 0.50` OR `max_prob - second_prob < 0.15`). This allowed the model's native intelligence to handle nuance rather than relying on brittle dictionaries.
2. **Keras 3 Math Ops**: Migrated the `AttentionLayer` from deprecated `tf.keras.backend` methods to backend-agnostic `keras.ops`.
3. **Tasks Vision API**: Upgraded facial detection from legacy `mp.solutions` to the MediaPipe Tasks Vision API (`vision.FaceDetector`), increasing the confidence threshold to 0.75 to ruthlessly eliminate false positives.

## Why The Phase Ended
While the individual microservices were technically functional, they were structurally incomplete for a final product. The video service still lacked batched processing (analyzing frame-by-frame instead of as a tensor stack), and none of the APIs utilized asynchronous thread offloading (`asyncio.to_thread`), meaning heavy ML inference could still block the FastAPI event loop under concurrent load. Furthermore, a multimodal system inherently requires a fusion layer to combine the signals into a unified psychological profile—necessitating the creation of Phase 9's master orchestrator.

## Legacy Impact
Phase 8 represents the "dress rehearsal" for production. The specific file forms created here (`main_audio.py`, `main_video.py`, `main_text.py`) survived almost intact into the final `services/` directory, requiring only the final performance optimizations (`@tf.function` and thread pooling) to achieve enterprise-grade stability.
