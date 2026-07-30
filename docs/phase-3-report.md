# CaptionForge Phase 3 Implementation Report

## Status and scope

Phase 3 is implemented. CaptionForge now downloads the caption track chosen by
the Phase 2 selection rules and exports it as SRT, VTT, TXT, JSON, or any
combination of those formats. Both manual subtitles and YouTube-generated
captions are supported.

No audio or video is downloaded: the yt-dlp extraction configuration retains
`skip_download=True` and enables only the subtitle option corresponding to the
selected track. FFmpeg, Whisper, transcription fallback, translation, GUI, and
playlist work remain outside this phase.

## Design

- `YtDlpAdapter.download_subtitle` is the raw acquisition boundary. It requests
  one selected language, prefers VTT, uses an isolated temporary directory, and
  returns UTF-8 text as `RawSubtitle`.
- `SubtitleService` preserves its Phase 2 selection API and now separately
  parses VTT/SRT/JSON3 data and performs conservative cleanup.
- `SubtitleSegment` is the common representation: index, start seconds, end
  seconds, text, and language.
- `ExportService` validates formats and destinations, preflights overwrite
  conflicts, and dispatches to format-specific renderers.
- SRT, VTT, TXT, and JSON rendering are independent exporter modules.
- Time, Arabic-safe text cleanup, and safe filename handling are pure utility
  modules.

## Cleanup behavior

Cleanup removes empty segments, simple caption markup, cue-only annotations,
and exact consecutive duplicates. It decodes HTML entities and collapses
whitespace. Negative, invalid, and overlapping timings are corrected while
retaining a positive segment duration. Indices are rebuilt after cleanup.

Arabic letters, diacritics, words, and punctuation are preserved. No spelling,
letter-shape, hamza, alef, or ya normalization is performed.

## CLI

```bash
captionforge extract VIDEO_URL
captionforge extract VIDEO_URL --language ar
captionforge extract VIDEO_URL --format srt --format txt
captionforge extract VIDEO_URL --output ./output --timestamped-txt
captionforge extract VIDEO_URL --format json --overwrite
```

The command inspects the video, selects a track, retrieves and parses it,
exports the requested formats, then prints every generated absolute path. If no
track matches, it explains that transcription fallback will arrive in a later
phase.

## Output safety

Output directories are created when possible and rejected when they are not
writable directories. Filename stems preserve Unicode while removing
cross-platform forbidden characters, device names, trailing dots, and excessive
length. Files are UTF-8 with normalized line endings. Every requested path is
checked before writing, and existing files require explicit `--overwrite`.

JSON output includes the complete stable video metadata model, normalized
selected language, manual/automatic source type, and all cleaned segments.

## Offline test coverage

Phase 3 tests cover:

- manual and automatic yt-dlp caption options;
- proof that `skip_download` remains enabled;
- VTT, SRT, and JSON3 parsing;
- Arabic UTF-8 and punctuation preservation;
- markup/empty-cue cleanup and duplicate removal;
- invalid and overlapping timestamp correction;
- portable filename sanitization;
- SRT, VTT, plain TXT, timestamped TXT, and JSON output;
- multiple-format CLI extraction;
- no-matching-caption messaging;
- invalid output paths;
- overwrite refusal and explicit overwrite.

The existing Phase 1 and Phase 2 suite remains green.

## Validation

```bash
.venv/bin/python -m ruff format app tests
.venv/bin/python -m ruff check app tests
.venv/bin/python -m pytest -q
```

Latest result:

```text
77 passed, 1 deselected
```

The deselected test is the opt-in live YouTube integration test. Default tests
remain offline and use mocked yt-dlp responses.
