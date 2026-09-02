# CaptionForge

CaptionForge downloads and exports existing YouTube captions. When no matching
caption exists, it prepares mono 16 kHz audio and transcribes it locally with
`faster-whisper`. It never downloads the full video.

## Features

- Run the whole workflow from a browser on your own machine
- Inspect video metadata and available caption tracks
- Select captions by preferred language
- Export SRT, VTT, TXT, JSON, or DOCX
- Generate multiple formats in one command
- Create plain or timestamped TXT transcripts
- Save a Word transcript as plain prose, with right-to-left support
- Preserve Arabic words, punctuation, and diacritics
- Produce safe filenames and never replace existing output
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
python -m pip install -e ".[web]"
```

## Web interface

Serve the interface to a browser on this computer:

```bash
captionforge web
```

CaptionForge prints a `http://127.0.0.1:PORT/?t=TOKEN` link and opens it. Paste a
video link, pick a language and formats, and download the results. Nothing is
uploaded anywhere: yt-dlp, FFmpeg, and faster-whisper all run locally, exactly as
they do for the commands below.

```bash
captionforge web --port 8800    # bind a fixed port instead of a free one
captionforge web --no-open      # print the link without opening a browser
```

The page remembers the language, formats, model, device, options, and vocabulary
hint you last used, and opens with them next time. They are kept in
`$XDG_CONFIG_HOME/captionforge/web-preferences.json`, beside `config.json` but
separate from it: interface choices never change what the CLI does. Delete that
file to start from the configured defaults again. Browser storage is not used,
because `captionforge web` binds a new port each run and that changes the page's
origin.

The server binds `127.0.0.1` only, never `0.0.0.0`. Every `/api` call must carry
the per-process token from that link, requests addressed to any other hostname
are refused, and downloads are restricted to the files the job actually wrote.
The page loads no fonts, scripts, or styles from the internet, so it works with
the network unplugged once a caption track or model is already local.

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

Save a Word document instead of a subtitle file:

```bash
captionforge extract "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" \
  --language ar --format docx --output ./output
```

The `docx` format writes the transcript as plain readable prose: no cue numbers
and no timestamps. Consecutive cues are joined into paragraphs that break at a
silent gap, at a sentence boundary, or at a length limit. The video title becomes
the document heading, and Arabic (or any other right-to-left language) is written
right-aligned with the correct text direction. `--timestamped-txt` affects TXT
only and never adds timings to the Word document.

Export captions when available, otherwise transcribe locally:

```bash
captionforge transcribe "https://youtu.be/qJFbKl6RjLU" \
  --language ar --model small --device auto --compute-type auto \
  --format srt --format txt --output ./output
```

The command reuses a suitable YouTube caption by default. Use `--force` to run
Whisper anyway, `--keep-audio` to preserve prepared audio, and `--overwrite` to
replace output files.

Existing output is never replaced or refused. When the natural filename is
taken, the whole set of generated files moves to the next free variant —
`title (2).srt`, `title (3).srt` — so a long transcription is never discarded
over a name clash. Pass `--overwrite` to replace the originals in place
instead. The default model is `small`, avoiding an impractical
large-model default on low-resource machines.

Transcription quality options:

- `--prompt` supplies a vocabulary hint. Proper nouns and domain terms in the
  prompt measurably improve how Whisper spells them.
- `whisper_word_timestamps` (default on) makes CaptionForge cut subtitle lines
  on real word boundaries instead of estimating them from character counts. It
  costs a little speed and is what `whisper_hallucination_silence_threshold`
  needs in order to work.
- `whisper_condition_on_previous_text` defaults to **false**, unlike the
  faster-whisper default. Carrying decoded text between windows is the usual
  cause of repetition loops and drift on long recordings; the cost is slightly
  less cross-window context. Set it to `true` to restore the engine default.
- `whisper_compression_ratio_threshold`, `whisper_log_prob_threshold` and
  `whisper_no_speech_threshold` are the engine's degeneracy guards, now tunable.

`captionforge prepare-audio` remains available for audio-only diagnostics.
Run `captionforge doctor` to check FFmpeg, yt-dlp, faster-whisper, python-docx,
CUDA, the detected GPU, recommendations, and folder access. Doctor never
downloads a model.

Post-processing is enabled for both downloaded captions and Whisper results. Use
`--no-postprocess` with `extract` or `transcribe` when source segmentation must be
retained. Post-processing removes redundant cues but never rewrites speech:
collapsing an adjacent repeated phrase is opt-in via `collapse_repeated_phrases`,
because deliberate rhetorical repetition is indistinguishable from an artifact. Clean an existing file without downloading or transcribing:

```bash
captionforge clean captions.srt
captionforge clean captions.vtt --output ./output
```

The clean command preserves the input format and writes `*.cleaned.srt` or
`*.cleaned.vtt` unless another destination is supplied; it does not convert to
DOCX.

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
CAPTIONFORGE_WHISPER_VAD_THRESHOLD=0.5
CAPTIONFORGE_WHISPER_VAD_SPEECH_PAD_MS=400
CAPTIONFORGE_WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
CAPTIONFORGE_WHISPER_INITIAL_PROMPT=
CAPTIONFORGE_WHISPER_WORD_TIMESTAMPS=true
CAPTIONFORGE_WHISPER_COMPRESSION_RATIO_THRESHOLD=2.4
CAPTIONFORGE_WHISPER_LOG_PROB_THRESHOLD=-1.0
CAPTIONFORGE_WHISPER_NO_SPEECH_THRESHOLD=0.6
CAPTIONFORGE_WHISPER_HALLUCINATION_SILENCE_THRESHOLD=
CAPTIONFORGE_WHISPER_LANGUAGE=
CAPTIONFORGE_WHISPER_MODEL_DOWNLOAD_DIRECTORY=
CAPTIONFORGE_MAXIMUM_CHARACTERS_PER_LINE=42
CAPTIONFORGE_MAXIMUM_SUBTITLE_LINES=2
CAPTIONFORGE_MINIMUM_SUBTITLE_DURATION=0.8
CAPTIONFORGE_MAXIMUM_SUBTITLE_DURATION=7.0
CAPTIONFORGE_SUBTITLE_MERGE_THRESHOLD=1.0
CAPTIONFORGE_DUPLICATE_DETECTION_THRESHOLD=0.9
CAPTIONFORGE_REMOVE_DIACRITICS=false
CAPTIONFORGE_COLLAPSE_REPEATED_PHRASES=false
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
- **HTTP 403 on audio or captions** means yt-dlp can no longer read YouTube's
  current site. Retrying never fixes it; upgrade instead:

  ```bash
  python -m pip install --upgrade yt-dlp
  ```

  yt-dlp also wants a JavaScript runtime (`deno`) for signature extraction and
  has deprecated working without one. `doctor` reports whether you have it.
- **"a required NVIDIA runtime library is missing"** means a GPU was detected
  but CUDA's math libraries could not be loaded. Install them:

  ```bash
  python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  ```

  These wheels place their shared objects in `site-packages/nvidia/*/lib`, a
  directory the dynamic loader never searches, so being installed is not enough
  on its own. CaptionForge loads them from there by absolute path at startup,
  which is why no `LD_LIBRARY_PATH` is needed. If `doctor` still reports them as
  missing, the wheels are genuinely absent or the GPU is unusable; run with
  `--device cpu` in the meantime. `doctor` names the specific libraries it could
  not load.
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
--format FORMAT     srt, vtt, txt, json, or docx; may be repeated
--output PATH       Output directory
--timestamped-txt   Add timestamps to TXT output
--overwrite         Replace existing output files instead of numbering
--no-postprocess    Bypass Phase 6 processing
--allow-translated  Also consider machine-translated caption tracks
--prompt TEXT       Vocabulary hint for Whisper (names, terms)
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

YouTube also publishes machine translations of its automatic transcription for
roughly 150 languages. These are translations of a transcription and rank
*below* having no track at all, so `transcribe` falls back to local Whisper
instead of exporting them. `inspect` marks them in a `Translated` column, and
`--allow-translated` opts back in.

Caption tracks are downloaded as `json3` where available. YouTube's automatic
VTT is a rolling format in which each cue repeats the previous line, roughly
tripling the word count and losing the true start of the first phrase; `json3`
carries one cue per phrase and needs no repair.

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
- No live streams, playlists, or translation
- The web interface covers `extract` and `transcribe` only; settings, `clean`, `doctor`, and local file input stay on the command line
- No authenticated or cookie-based access
- No speaker diarization, translation, or aggressive spelling/grammar rewriting

For implementation details, see the
[Phase 6 report](docs/phase-6-report.md),
[Phase 7 report](docs/phase-7-report.md),
[Phase 5 report](docs/phase-5-report.md),
[Phase 4 report](docs/phase-4-report.md),
[Phase 3 report](docs/phase-3-report.md), and
[Phase 2 report](docs/phase-2-report.md).
