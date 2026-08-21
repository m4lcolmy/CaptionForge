# CaptionForge — How the App Works

A working guide to the internals: what each layer does, what happens on every
command, and which rules decide the output. Version 0.7.0, Python 3.12+.

---

## 1. What the app actually does

CaptionForge turns a single YouTube video URL into subtitle/transcript files
(`srt`, `vtt`, `txt`, `json`, `docx`). It has two sources of text and always
prefers the cheap one:

1. **Existing YouTube captions** — downloaded as a caption track only, no media.
2. **Local transcription** — only when no caption matches the requested
   language (or `--force` is passed). Audio-only stream is downloaded,
   converted to mono 16 kHz PCM WAV, and fed to `faster-whisper` on the local
   machine.

The video stream is never downloaded in any path. There is no network call to
any service other than YouTube (via `yt-dlp`) and the Whisper model host on
first model download.

---

## 2. Layer map

```
app/
├── main.py              entry point → app.interfaces.cli:app  (console script "captionforge")
├── interfaces/cli.py    Typer commands, Rich rendering, error→exit-code mapping
├── services/            orchestration; the only place where a workflow is decided
│   ├── video_service.py         URL validation + live/availability guards + discovery
│   ├── subtitle_service.py      track selection, caption parsing, minimal cleanup
│   ├── audio_service.py         job workspace, audio download, FFmpeg conversion
│   ├── transcription_service.py caption-first workflow, Whisper fallback, export
│   ├── postprocessing_service.py  timing + text normalization (shared by all sources)
│   └── export_service.py        format validation, filename, atomic multi-format write
├── adapters/            everything that talks to the outside world
│   ├── ytdlp_adapter.py    metadata, caption download, audio download, error translation
│   ├── ffmpeg_adapter.py   subprocess (no shell), conversion, error translation
│   └── whisper_adapter.py  lazy faster-whisper import, device/compute selection
├── models/              frozen Pydantic contracts (VideoMetadata, SubtitleTrack, …)
├── exporters/           pure render functions: segments → text
├── utils/               pure helpers (URL, time, language, filenames, Arabic text)
└── core/                config, constants, exception hierarchy, retry, logging
```

The dependency direction is strict: `cli → services → adapters → models/utils`.
Adapters never import services. Exporters are pure functions with no I/O — the
`ExportService` owns the writing. This is what makes the whole thing testable
offline: every adapter takes an injectable factory/runner
(`ExtractorFactory`, `ProcessRunner`, `model_factory`, `cuda_detector`).

---

## 3. Configuration resolution

`Config` ([app/core/config.py](../app/core/config.py)) is a frozen Pydantic model
with `extra="forbid"`. `Config.load()` merges four sources, in this order of
increasing precedence:

1. Field defaults on the model.
2. `.env` in the working directory (via `dotenv_values`, not injected into
   `os.environ`).
3. The persisted user file — `$CAPTIONFORGE_CONFIG_FILE`, else
   `$XDG_CONFIG_HOME/captionforge/config.json`, else
   `~/.config/captionforge/config.json`.
4. Process environment variables.

**One subtlety worth knowing:** every field is looked up as
`CAPTIONFORGE_<FIELD>` *first*, and only then as the bare `<field>` name. The
persisted JSON file uses bare names. So a `CAPTIONFORGE_RETRY_COUNT` present in
`.env` will outrank a `retry_count` written by `captionforge config set`. If a
setting seems to be ignored, that is almost always why.

**Damage tolerance.** A corrupt JSON config is silently skipped. If the merged
value set fails validation as a whole, `load()` retries field by field and keeps
only the individually valid ones — one stale value cannot make every command
unusable. Only a total failure raises `ConfigurationError`.

`config set` is stricter than loading: `Config.parse_setting` validates the
single key/value and rejects unknown keys or invalid values outright, then the
whole model is re-validated and written atomically.

Validated constraints that matter in practice:

| Setting | Rule |
|---|---|
| `whisper_device` | `auto` \| `cpu` \| `cuda` |
| `whisper_compute_type` | `auto`, `default`, `int8`, `int8_float16`, `int8_float32`, `int16`, `float16`, `float32`, `bfloat16` |
| `maximum_subtitle_lines` | 1 or 2 only |
| `maximum_subtitle_duration` | must be ≥ `minimum_subtitle_duration` (cross-field validator) |
| `default_output_formats` | comma string or tuple; must be non-empty and a subset of srt/vtt/txt/json/docx |
| `whisper_language`, `whisper_model_download_directory` | blank string is coerced to `None` (unset) |
| `retry_count` | 1–10 |

---

## 4. Commands and what each one triggers

| Command | Network | Writes files | Core path |
|---|---|---|---|
| `version` | no | no | prints constant |
| `config show/set/reset` | no | user config only | `Config` |
| `doctor` | no | creates output/temp dirs to test them | local probes |
| `inspect` | metadata only | no | `VideoService.inspect` |
| `extract` | metadata + caption track | yes | captions → parse → post-process → export |
| `transcribe` | metadata + (caption **or** audio) | yes | caption-first, Whisper fallback |
| `prepare-audio` | metadata + audio | WAV only | `AudioService.prepare` |
| `clean` | none | yes | local file → parse → post-process → export |

Every command runs through `_run_with_config`, which does the same four things:
load config, configure logging, run the action, and translate exceptions into a
short user-facing message plus an exit code. Raw `yt-dlp`, FFmpeg, CUDA or
Python text never reaches stdout/stderr — it goes to the log file.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | generic failure (incl. unexpected exceptions) |
| 2 | invalid or unsupported YouTube URL |
| 3 | video unavailable or live stream |
| 4 | metadata retrieval failure |
| 130 | `KeyboardInterrupt` |

---

## 5. The inspection path (shared by everything that touches YouTube)

1. **`extract_youtube_video_id`** ([app/utils/url_utils.py](../app/utils/url_utils.py))
   — pure parsing, no network. Accepts `youtube.com`, `www`, `m`, `music`, and
   `youtu.be` hosts; `/watch?v=`, `/shorts/`, `/embed/`, `/v/`, and bare
   `youtu.be/<id>`. Rejects `/playlist`, `/channel`, `/user`, `/c/`, `/@`,
   `/results`, `/feed` with `UnsupportedYouTubeUrlError`; everything else with
   `InvalidYouTubeUrlError`. The ID must match `^[A-Za-z0-9_-]{11}$`. A scheme
   is added if missing.
2. **Canonicalization** — the parsed ID is rebuilt into
   `https://www.youtube.com/watch?v=<id>`. Tracking parameters, `si=`, playlist
   context and timestamps are discarded before yt-dlp ever sees the URL.
3. **`YtDlpAdapter.inspect`** — one `extract_info(download=False)` call with
   `skip_download`, `noplaylist`, `quiet`. Wrapped in `retry_call`.
4. **Error translation** — `DownloadError` text is matched to
   `PrivateVideoError`, `VideoUnavailableError` (removed / members-only /
   age-restricted / region-blocked), or `MetadataRetrievalError`.
5. **Mapping** — raw dict → frozen `VideoMetadata`. Caption dictionaries
   (`subtitles`, `automatic_captions`) → `SubtitleTrack` tuples, sorted by
   normalized language code, with unparseable language codes dropped.
6. **Guards** in `VideoService` — `is_live` or `live_status in {is_live,
   is_upcoming}` → `LiveStreamNotSupportedError`; `availability in {private,
   subscriber_only, premium_only}` → `VideoUnavailableError`.
7. **Selection** — handed to `SubtitleService.discover`.

### Track selection rules

`SubtitleService.select_track` applies four tiers in strict order and stops at
the first non-empty tier:

1. Manual track, exact normalized match (`ar-EG` == `ar-EG`)
2. Manual track, same base language (`ar-EG` → any `ar*`)
3. Automatic track, exact match
4. Automatic track, same base language

Within a tier, ties break by preferring the bare base code (`ar` over `ar-EG`),
then alphabetically. Language codes are normalized first
([app/utils/language_utils.py](../app/utils/language_utils.py)): `_` → `-`,
language lowercased, region uppercased, script title-cased — `AR_eg` becomes
`ar-EG`. Malformed codes return `None` and are skipped rather than raising.

---

## 6. `extract` — captions to files

```
inspect → select track → download that track only → parse → post-process → export
```

The caption download uses a `TemporaryDirectory` with `subtitlesformat:
"vtt/best"` and `subtitleslangs: [track.language_code]`, and sets exactly one of
`writesubtitles` / `writeautomaticsub` based on `track.is_automatic`. It reads
the resulting file with `utf-8-sig` (BOM-tolerant) and returns a `RawSubtitle`.

**Parsing** handles two shapes:

- `vtt` / `srt` — line scan for `START --> END` (extra cue settings after the
  timestamps are ignored), then all following non-blank lines become the text
  block. Timestamps go through `parse_timestamp`, which accepts optional hours
  and both `,` and `.` as the millisecond separator, and rejects
  minutes/seconds > 59.
- `json3` — YouTube's JSON caption format; `events[].segs[].utf8` concatenated,
  `tStartMs` + `dDurationMs` as timing.

Anything else raises `SubtitleParseError`.

With `--no-postprocess`, a minimal cleanup path runs instead
(`_clean_segments`): markup stripped, exact consecutive duplicates dropped,
overlaps trimmed — but no merging, splitting, or re-wrapping. Use it when the
source segmentation must be preserved verbatim.

---

## 7. `transcribe` — caption-first with a Whisper fallback

`TranscriptionService.process` is the only place that decides between the two
sources.

```
inspect (5%)
  ├── track found and not --force  → download captions (25%) → export (85%) → done
  └── otherwise
        prepare audio (10% → 25%)
          ├── create job dir + disk check
          ├── yt-dlp bestaudio download
          └── ffmpeg → mono 16 kHz PCM WAV
        load model (30%)
        transcribe (40% → 85%)
        post-process (88%)
        export (92%)
        done (100%)
```

The progress percentages are real and deterministic — the audio sub-progress is
remapped with `10 + percent * 0.15`, and Whisper's segment loop maps elapsed
audio time onto 40–85 (`40 + min(45, end/duration*45)`), falling back to a
slow `min(84, 40 + segment_index)` crawl when the duration is unknown.

### Audio preparation ([app/services/audio_service.py](../app/services/audio_service.py))

- A `Job` is created with a UUID; the workspace is
  `<temp_directory>/captionforge-<uuid>` created with mode `0o700` and probed
  with a write test.
- Disk space required is `max(minimum_free_disk_bytes, duration_seconds *
  64000)` — roughly twice the PCM footprint, to cover the compressed download
  plus the WAV output. Unknown duration falls back to 600 s.
- `yt-dlp` with `format: "bestaudio"`. A "requested format is not available"
  error becomes `AudioFormatUnavailableError`; everything else becomes a
  retryable `AudioDownloadError`.
- FFmpeg is invoked as an argument list, never through a shell:
  `-y -i <src> -vn -acodec pcm_s16le -ar 16000 -ac 1 <dst>`. The output is
  verified to exist and be non-empty afterwards. Note that `audio_format` only
  changes the file extension — the codec is always `pcm_s16le`, which is what
  Whisper wants.
- On success without preservation, the WAV is moved out to
  `<temp>/captionforge-<uuid>.wav` and the job directory is deleted. With
  preservation, everything stays inside the job directory.
- `KeyboardInterrupt` → job `CANCELLED` + workspace removed +
  `ProcessingInterruptedError`. Any `CaptionForgeError` → job `FAILED` +
  workspace removed.

Standalone `prepare-audio` refuses to run when a matching caption exists,
unless `--force` — it is a diagnostic command, not part of the normal flow.
Inside `transcribe` it is always called with `force=True` (the decision was
already made) and `keep_temp=True` (the caller owns cleanup, in its `finally`).

### Whisper adapter ([app/adapters/whisper_adapter.py](../app/adapters/whisper_adapter.py))

- `faster_whisper` is imported **lazily**, via `importlib`, at transcription
  time. The rest of the app works without it installed; a missing package
  surfaces as `WhisperNotInstalledError`.
- Device: `auto` → `cuda` if `ctranslate2.get_cuda_device_count()` is non-zero,
  else `cpu`. An explicit `cuda` with no GPU raises `CudaUnavailableError`
  rather than silently degrading.
- Compute type: `auto` → `float16` on CUDA, `int8` on CPU.
- VAD is on by default with `min_silence_duration_ms=500`.
- Cancellation is checked before loading and on every produced segment.
- Engine objects never escape: segments are converted to frozen
  `TranscriptionSegment` models inside the loop, the generator is closed, and
  `gc.collect()` runs in `finally` to release GPU memory promptly.
- Error translation is string-based on the exception message: "out of memory" →
  `GpuMemoryError`; "compute type"/"quantization" → `InvalidComputeTypeError`;
  "invalid model"/"model not found"/"repository not found" during load →
  `UnsupportedModelError`; any other load-time failure → retryable
  `ModelLoadError`; any runtime failure → `AudioTranscriptionError`.
- `confidence` is `exp(avg_logprob)` clamped to 0–1. It is a useful ordering
  score, deliberately not presented as a calibrated probability.

Zero usable segments raises `EmptyTranscriptionError` — an empty file is never
written.

---

## 8. Post-processing — the part that shapes the output

`PostProcessingService._process`
([app/services/postprocessing_service.py](../app/services/postprocessing_service.py))
runs the same six stages for downloaded captions and Whisper output alike, in
this fixed order:

**1. Text cleanup** (`clean_caption_text`)
HTML entities unescaped, zero-width spaces removed, `<...>` tags stripped,
whitespace collapsed. A cue that is *entirely* `[...]` or `(...)` is dropped.
Punctuation spacing is normalized — space removed before `، ؛ ؟ , . ! ? : ; ٪ %`
and added after, **except** between digits, so `3.14` and `1,000` survive.
Diacritic removal, alef/ya normalization and Arabic-Indic digit conversion are
all **off by default**; the letters the speaker said are not altered unless you
ask. Then adjacent repeated phrases of ≥2 words are collapsed
(`_collapse_repeated_phrase`). A recognized silence cue (music/applause/
silence/موسيقى/تصفيق/صمت) is dropped only if `no_speech_probability ≥ 0.9`.

**2. Duplicate removal**
Compared against the previous segment only, using a case-folded cleaned key.
Identical → previous segment's end is extended and the new one dropped.
Otherwise `SequenceMatcher.ratio() ≥ duplicate_detection_threshold` (0.9) or a
detected leading-phrase overlap triggers a fix: the repeated leading phrase is
trimmed from the current text, or — if the current text is not longer — it is
absorbed into the previous segment. This is what kills the rolling-repetition
artifact typical of YouTube auto-captions.

**3. Timing repair (pass 1, no minimum duration)**
Negative starts clamped to 0; an overlap either shortens the previous segment or
pushes the current start forward; durations clamped to
`maximum_subtitle_duration`.

**4. Merge short**
Two neighbours merge when *all* hold: the gap ≤ `subtitle_merge_threshold`
(1.0 s), the combined length ≤ `maximum_characters_per_line ×
maximum_subtitle_lines` (84 by default), at least one side is shorter than
`minimum_subtitle_duration` or is a single word, and the previous text does not
already end a sentence (`. ! ? ؟ ؛ …`).

**5. Split long**
Splits by character budget *and* by `maximum_subtitle_duration`, preferring
sentence ends once past half the limit. New timings come from
`distribute_duration`, which allocates the range proportionally to part length
using cumulative boundaries — so parts are contiguous with no floating-point
gaps or overlaps.

**6. Timing repair (pass 2, minimum duration enforced)** then **line wrapping**
`_wrap` finds the word boundary that most evenly balances the two halves and
splits there — but only if *both* halves fit within
`maximum_characters_per_line`. Otherwise the line is left long rather than
broken badly.

`--no-postprocess` skips all of this for `extract` and `transcribe`. The `clean`
command is post-processing applied to a file you already have.

---

## 9. Export and file safety

`ExportService.export` ([app/services/export_service.py](../app/services/export_service.py)):

1. Formats are lowercased and de-duplicated in order (`dict.fromkeys`);
   unknown formats raise before anything is written.
2. Output directory is created and checked for `W_OK`.
3. Filename stem comes from `sanitize_filename(video.title)`: NFC-normalized,
   `<>:"/\|?*` and control characters replaced with `_`, whitespace collapsed,
   trimmed of leading/trailing spaces and dots, truncated to 180 characters.
   Empty results or Windows reserved names (`CON`, `NUL`, `COM1`…) fall back to
   the video ID. Arabic, Turkish and other Unicode is deliberately preserved.
4. **All** target paths are checked for existence *before* rendering. Without
   `--overwrite`, the command aborts having written nothing.
5. Contents are rendered in memory, the total UTF-8 byte size is checked against
   free disk space, then each file is written with `atomic_write_text`:
   `mkstemp` in the same directory → write → `flush` → `os.fsync` →
   `Path.replace`. A reader never sees a partial file, and a crash cannot
   corrupt an existing one.
6. If a later file in a multi-format export fails, files newly created by that
   same call are unlinked — no half-finished sets.

Renderers ([app/exporters/](../app/exporters/)):

- **SRT** — 1-based index, `HH:MM:SS,mmm`, re-indexed on export.
- **VTT** — `WEBVTT` header, `HH:MM:SS.mmm`, no cue identifiers.
- **TXT** — one segment per line; `--timestamped-txt` prefixes
  `[HH:MM:SS.mmm]\t`.
- **JSON** — full `VideoMetadata`, selected language, caption source type, and
  every segment field including `confidence` and `no_speech_probability`.
  `ensure_ascii=False`, so Arabic stays readable in the file.

---

## 10. Errors, retries, logging

**Hierarchy.** Everything expected derives from `CaptionForgeError`, which
carries a user-facing `message` and a technical `details`. The base class
exposes `retryable = False`; only four exception types override it to `True`:

- `MetadataRetrievalError`
- `SubtitleDownloadError`
- `AudioDownloadError` (and its subclass `AudioFormatUnavailableError`)
- `ModelLoadError` (and its subclass `UnsupportedModelError`)

**Retry.** `retry_call` ([app/core/retry.py](../app/core/retry.py)) re-runs an
operation only when the *translated* exception is retryable. Invalid URLs,
unavailable videos, missing FFmpeg, bad timestamps and invalid output paths fail
on the first attempt. Retries are logged with the attempt number and technical
cause. `retry_count` and `retry_delay_seconds` are configurable; the delay is
fixed, not exponential.

Note that `UnsupportedModelError` inherits `retryable = True` from
`ModelLoadError`, so a genuinely bad model name is retried `retry_count` times
before failing.

**Logging.** `configure_logging` removes Loguru's default handler and adds a
**file sink only** — `logs/captionforge_YYYY-MM-DD.log`, rotating at 10 MB,
retained 14 days, UTF-8, `enqueue=True`, with backtrace and diagnose disabled so
no local variables leak into the file. Nothing from the logger reaches the
terminal; user-facing text is printed separately via Rich. Logged facts include
job IDs, stages, selected method, model/device/compute choices, retry attempts,
output paths and durations.

**Cancellation.** `Ctrl-C` propagates as `KeyboardInterrupt`, is converted to
`ProcessingInterruptedError` / `TranscriptionCancelledError`, the job is marked
`CANCELLED`, temporary and partial files are removed unless preservation was
requested, and the process exits 130.

---

## 11. The `Job` model

`Job` ([app/models/job.py](../app/models/job.py)) is the in-process record for
one run: UUID, source URL, language, formats, status, timestamps, workspace and
audio paths, progress, current stage, failure stage and total duration.
`transition()` stamps `started_at` on the first non-pending state and, on
`COMPLETED`/`FAILED`/`CANCELLED`, stamps `finished_at`, computes
`duration_seconds`, and records `failure_stage` on failure. It is not persisted
across runs — it exists to give logs a consistent, correlatable shape.

---

## 12. Development

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=app --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check app tests
.venv/bin/python -m mypy app
```

The default suite is fully offline — `addopts = "-ra -m 'not integration'"`.
Live tests are opt-in and require `CAPTIONFORGE_INTEGRATION_VIDEO_URL` (YouTube)
or `CAPTIONFORGE_INTEGRATION_AUDIO` (real Whisper); run them with
`pytest -m integration`.

Offline testing is possible because every external boundary is injectable:
`YtDlpAdapter(extractor_factory=…)`, `FFmpegAdapter(runner=…)`,
`WhisperAdapter(model_factory=…, cuda_detector=…)`, and `retry_call(sleep=…)`.

---

## 13. Extending it

| Goal | Touch |
|---|---|
| New output format | add a render function in `app/exporters/`, register it in `ExportService.export`, add it to `SUPPORTED_OUTPUT_FORMATS` |
| Different transcription engine | new adapter returning `TranscriptionResult`; `TranscriptionService` needs no change |
| Another video source | new adapter with the same `inspect`/`download_subtitle`/`download_audio` shape |
| Different subtitle style rules | `PostProcessingService` + the related `Config` fields |
| New language display name | `_LANGUAGE_NAMES` in `app/utils/language_utils.py` |

---

## 14. Known limitations and rough edges

Deliberate limits: individual non-live videos only; no playlists, channels,
live streams, translation, diarization, cookie/authenticated access, or GUI; no
aggressive spelling or grammar rewriting.

Rough edges in the current code, worth knowing:

- `prepare-audio` still prints "Transcription will be implemented in Phase 5",
  and `extract` still says the transcription fallback "will be added in a later
  phase" when no caption matches. Both are stale — `transcribe` implements it
  today.
- `FFmpegAdapter.build_conversion_command` has a `codec = "pcm_s16le" if … else
  "pcm_s16le"` branch; `audio_format` only affects the file extension.
- `configure_logging`'s docstring mentions console and file handlers, but only
  the file handler is registered.
- The `clean` command builds a placeholder `VideoMetadata` with the synthetic ID
  `localclean1` so it can reuse `ExportService`; the file is then renamed to the
  requested destination.
