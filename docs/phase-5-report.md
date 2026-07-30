# Phase 5 — Local Whisper Transcription

Phase 5 adds caption-first local speech transcription using `faster-whisper`.
The implementation remains offline-testable and keeps engine objects at the
adapter boundary.

## Architecture

- `WhisperAdapter` lazily imports and loads faster-whisper, selects CPU/CUDA and
  compute type, applies VAD settings, converts generated segments, reports
  progress, checks cancellation, translates errors, and releases the model.
- `TranscriptionService` inspects once, reuses a matching caption unless forced,
  invokes Phase 4 audio preparation otherwise, converts transcription segments
  to the common subtitle model, exports them, and cleans the per-job workspace.
- `TranscriptionResult` and `TranscriptionSegment` carry stable engine-neutral
  language, probability, timing, index, and optional confidence information.
- The job model now represents preparing, loading, transcribing,
  post-processing, exporting, completed, cancelled, and failed states.

## Defaults and configuration

The default model is `small`; automatic selection uses CUDA with `float16` when
available and CPU with `int8` otherwise. The device, compute type, beam size,
VAD toggle, minimum silence duration, language, and model download directory
are configurable with the `CAPTIONFORGE_WHISPER_*` environment variables
documented in the README.

An explicit unavailable CUDA request fails clearly instead of silently using
CPU. Model loading and downloads happen only when transcription is actually
needed. The doctor command performs local package and hardware checks without
downloading a model.

## CLI workflow

`captionforge transcribe VIDEO_URL` supports repeated output formats, language,
model, device, compute type, destination, forced transcription, retained audio,
timestamped TXT, and overwrite behavior. Existing captions use the Phase 3
retrieval/export path; forced or missing captions use Phase 4 audio plus
Whisper. Generated paths and selected transcription settings are printed.

## Error handling and cleanup

CaptionForge translates missing dependencies, unavailable CUDA, invalid compute
types, unsupported models, loading/download failures, GPU memory exhaustion,
unreadable audio, empty speech, and cancellation into concise application
errors. Temporary per-job audio is removed after success, failure, or
cancellation unless `--keep-audio` or `CAPTIONFORGE_KEEP_TEMP_FILES=true` is
selected.

## Verification

The default suite mocks all Whisper model activity. It covers CPU/CUDA and
compute selection, loading, Arabic/VAD options, conversion, empty output,
progress, cancellation, loading and GPU errors, caption-first and forced
workflows, the CLI, cleanup, and SRT/VTT/TXT/JSON export. A real-model smoke
test is marked `integration` and requires `CAPTIONFORGE_INTEGRATION_AUDIO`, so
normal test runs never download a model.
