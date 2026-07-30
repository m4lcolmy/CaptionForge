# Phase 4 Report — Audio Fallback and FFmpeg Integration

## Scope

Phase 4 prepares transcription-friendly audio when the requested caption
language is unavailable. It does not transcribe, generate subtitles, translate,
process playlists, or add a GUI.

## Workflow

`prepare-audio` validates and inspects an individual YouTube video through the
existing services. A matching manual or automatic caption stops processing by
default; `--force` explicitly bypasses that guard.

yt-dlp requests `bestaudio`, with playlist processing disabled, into a
unique job directory. FFmpeg produces 16-bit PCM WAV at 16 kHz with one channel
by default. No full video is downloaded.

On success, the prepared output remains in the configured temporary root while
the source and job directory are removed. On failure, the job directory is
removed. `--keep-temp` (or configuration) preserves the job directory, source,
and converted file for debugging.

## Configuration

- `AUDIO_FORMAT` (`wav`)
- `AUDIO_SAMPLE_RATE` (`16000`)
- `AUDIO_CHANNELS` (`1`)
- `TEMP_DIRECTORY` (`temp`)
- `KEEP_TEMP_FILES` (`false`)
- `FFMPEG_EXECUTABLE` (`ffmpeg`)

Environment names use the `CAPTIONFORGE_` prefix.

## Errors and diagnostics

Raw subprocess and yt-dlp messages are hidden from CLI users. Domain errors
cover missing FFmpeg, unavailable audio, download/conversion failure, disk
capacity, permissions, invalid temporary paths, interruption, and cleanup.

`captionforge doctor` checks the configured FFmpeg binary and version, the
installed yt-dlp package/version, and temporary-folder write access.

## Verification

Offline tests mock yt-dlp and subprocess execution. They cover audio-only
options and progress, command construction, conversion failures, workspace
creation, cleanup and preservation, caption/force behavior, CLI output, and
doctor checks. Fixtures and docs use:

`https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk`
