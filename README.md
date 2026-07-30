# CaptionForge

CaptionForge downloads and exports existing YouTube captions. When no matching
caption exists, it prepares mono 16 kHz audio and transcribes it locally with
`faster-whisper`. It never downloads the full video.

## Features

- Inspect video metadata and available caption tracks
- Select captions by preferred language
- Export SRT, VTT, TXT, or JSON
- Generate multiple formats in one command
- Create plain or timestamped TXT transcripts
- Preserve Arabic words, punctuation, and diacritics
- Produce safe filenames and prevent accidental overwrites
- Prepare audio-only fallback input with yt-dlp and FFmpeg
- Clean per-job intermediate files, or preserve them on request
- Automatically select CPU or NVIDIA CUDA and a suitable compute type
- Transcribe locally with VAD, timestamps, progress, and cancellation support
- Conservatively clean Arabic, Latin, and mixed-language subtitle text
- Repair timing, remove repetition, and format subtitles to two readable lines
- Retry temporary network failures without retrying invalid input
- Persist validated settings and write output files atomically
- Keep rotating technical logs separate from concise CLI errors

CaptionForge currently works with individual, non-live YouTube videos.

## Installation

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[transcription]"
```

## Usage

Inspect a video and its caption tracks:

```bash
captionforge inspect "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" --language ar
```

Download captions using the configured default formats:

```bash
captionforge extract "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" --language ar
```

Choose one or more formats and an output directory:

```bash
captionforge extract "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" \
  --format srt \
  --format txt \
  --output ./output
```

Export captions when available, otherwise transcribe locally:

```bash
captionforge transcribe "https://youtu.be/qJFbKl6RjLU" \
  --language ar --model small --device auto --compute-type auto \
  --format srt --format txt --output ./output
```

The command reuses a suitable YouTube caption by default. Use `--force` to run
Whisper anyway, `--keep-audio` to preserve prepared audio, and `--overwrite` to
replace output files. The default model is `small`, avoiding an impractical
large-model default on low-resource machines.

`captionforge prepare-audio` remains available for audio-only diagnostics.
Run `captionforge doctor` to check FFmpeg, yt-dlp, faster-whisper, CUDA, the
detected GPU, recommendations, and folder access. Doctor never downloads a model.

Post-processing is enabled for both downloaded captions and Whisper results. Use
`--no-postprocess` with `extract` or `transcribe` when source segmentation must be
retained. Clean an existing file without downloading or transcribing:

```bash
captionforge clean captions.srt
captionforge clean captions.vtt --output ./output
```

The clean command preserves the input format and writes `*.cleaned.srt` or
`*.cleaned.vtt` unless another destination is supplied.

## Configuration

Settings are persisted in
`$XDG_CONFIG_HOME/captionforge/config.json` (normally
`~/.config/captionforge/config.json`). Environment variables override persisted
values.

```bash
captionforge config show
captionforge config set retry_count 4
captionforge config set retry_delay_seconds 2
captionforge config reset
```

```text
CAPTIONFORGE_AUDIO_FORMAT=wav
CAPTIONFORGE_AUDIO_SAMPLE_RATE=16000
CAPTIONFORGE_AUDIO_CHANNELS=1
CAPTIONFORGE_TEMP_DIRECTORY=temp
CAPTIONFORGE_KEEP_TEMP_FILES=false
CAPTIONFORGE_FFMPEG_EXECUTABLE=ffmpeg
CAPTIONFORGE_DEFAULT_WHISPER_MODEL=small
CAPTIONFORGE_WHISPER_DEVICE=auto
CAPTIONFORGE_WHISPER_COMPUTE_TYPE=auto
CAPTIONFORGE_WHISPER_BEAM_SIZE=5
CAPTIONFORGE_WHISPER_VAD_ENABLED=true
CAPTIONFORGE_WHISPER_MIN_SILENCE_DURATION_MS=500
CAPTIONFORGE_WHISPER_LANGUAGE=
CAPTIONFORGE_WHISPER_MODEL_DOWNLOAD_DIRECTORY=
CAPTIONFORGE_MAXIMUM_CHARACTERS_PER_LINE=42
CAPTIONFORGE_MAXIMUM_SUBTITLE_LINES=2
CAPTIONFORGE_MINIMUM_SUBTITLE_DURATION=0.8
CAPTIONFORGE_MAXIMUM_SUBTITLE_DURATION=7.0
CAPTIONFORGE_SUBTITLE_MERGE_THRESHOLD=1.0
CAPTIONFORGE_DUPLICATE_DETECTION_THRESHOLD=0.9
CAPTIONFORGE_REMOVE_DIACRITICS=false
CAPTIONFORGE_NORMALIZE_ARABIC_LETTERS=false
CAPTIONFORGE_NORMALIZE_ARABIC_INDIC_DIGITS=false
CAPTIONFORGE_RETRY_COUNT=3
CAPTIONFORGE_RETRY_DELAY_SECONDS=1.0
CAPTIONFORGE_MINIMUM_FREE_DISK_BYTES=104857600
```

`CAPTIONFORGE_CONFIG_FILE` may point to another config file. Invalid persisted
values fall back individually to safe defaults; invalid values passed to
`config set` are rejected.

## Errors, retries, and logs

Normal CLI output contains a short actionable error, never raw yt-dlp, FFmpeg,
Whisper, CUDA, or Python details. Technical causes, job identifiers, stages,
selected methods, model/device choices, retries, output paths, and durations are
written to `logs/`. Logs rotate at 10 MB and are retained for 14 days.

CaptionForge retries translated temporary metadata, caption, audio-download, and
model-load failures. Invalid URLs, unsupported resources, unavailable videos,
bad timestamps, missing FFmpeg, invalid model names, and invalid output paths
are not retried. Interrupting a job cancels it and removes temporary and partial
files unless preservation was requested.

### Troubleshooting

- Run `captionforge doctor` to check FFmpeg, CUDA, dependencies, disk paths, and
  write access.
- Increase `retry_count` or `retry_delay_seconds` for throttling and unstable
  connections.
- For GPU memory errors, use a smaller model, `--compute-type int8`, or
  `--device cpu`.
- Use a writable `--output` directory and ensure both output and temporary
  filesystems have enough free space.
- Inspect the newest file in `logs/` when the CLI asks for technical details.

Useful options:

```text
--language ar       Preferred caption language
--format FORMAT     srt, vtt, txt, or json; may be repeated
--output PATH       Output directory
--timestamped-txt   Add timestamps to TXT output
--overwrite         Replace existing output files
--no-postprocess    Bypass Phase 6 processing
```

Run `captionforge --help` or `captionforge extract --help` for the complete
command reference.

## Caption selection

CaptionForge prefers tracks in this order:

1. Exact manual language match
2. Manual base-language match
3. Exact automatic language match
4. Automatic base-language match

For example, requesting `ar-EG` can match another Arabic variant when an exact
track is unavailable.

## Development

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=app --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check app tests
.venv/bin/python -m mypy app
```

The default test suite is offline. The optional live YouTube integration test
requires `CAPTIONFORGE_INTEGRATION_VIDEO_URL`. The optional real Whisper smoke
test requires `CAPTIONFORGE_INTEGRATION_AUDIO`; it is excluded by default.

Run optional integrations explicitly:

```bash
.venv/bin/python -m pytest -m integration
```

## Limitations

- No full-video downloading
- No live streams, playlists, translation, or GUI
- No authenticated or cookie-based access
- No speaker diarization, translation, or aggressive spelling/grammar rewriting

For implementation details, see the
[Phase 6 report](docs/phase-6-report.md),
[Phase 7 report](docs/phase-7-report.md),
[Phase 5 report](docs/phase-5-report.md),
[Phase 4 report](docs/phase-4-report.md),
[Phase 3 report](docs/phase-3-report.md), and
[Phase 2 report](docs/phase-2-report.md).
