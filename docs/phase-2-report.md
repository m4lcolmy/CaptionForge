# CaptionForge Phase 2 Implementation Report

## Implementation status

Phase 2 is implemented. CaptionForge validates individual YouTube URLs,
retrieves metadata through yt-dlp without downloading content, discovers manual
and automatic caption tracks, normalizes language identifiers, and selects a
preferred track. Phase 3 functionality was not implemented.

## Files created

- `app/adapters/ytdlp_adapter.py`
- `app/services/video_service.py`
- `app/services/subtitle_service.py`
- `app/utils/url_utils.py`
- `app/utils/language_utils.py`
- `tests/conftest.py`
- `tests/unit/test_url_utils.py`
- `tests/unit/test_language_utils.py`
- `tests/unit/test_subtitle_service.py`
- `tests/unit/test_ytdlp_adapter.py`
- `tests/unit/test_video_service.py`
- `tests/integration/test_youtube_inspection.py`
- `docs/phase-2-report.md`

## Files modified

- `app/core/constants.py`
- `app/core/exceptions.py`
- `app/interfaces/cli.py`
- `app/models/__init__.py`
- `app/models/video.py`
- `app/models/subtitle.py`
- `pyproject.toml`
- `requirements.txt`
- `README.md`
- `tests/unit/test_cli.py`

## Dependencies added

`yt-dlp>=2025.6` is the only new runtime dependency. The implementation uses
its Python API and never constructs a yt-dlp shell command.

## Architecture decisions

- Raw yt-dlp mappings are confined to the adapter.
- The adapter is instantiated per service graph and accepts an extractor
  factory for offline testing.
- Public results are immutable Pydantic models and do not retain raw metadata.
- `VideoService` owns URL validation and supported-content rules.
- `SubtitleService` owns language selection independently of yt-dlp.
- URL and language operations are pure utilities.
- CLI code constructs dependencies and renders models; it contains no
  extraction or selection rules.
- Exit codes are centralized in `ExitCode`.

The adapter explicitly sets `skip_download`, `noplaylist`, and all caption and
thumbnail write options to safe metadata-only values, and passes
`download=False` to extraction.

## URL formats supported

- `youtube.com/watch?v=VIDEO_ID`, including `www` and mobile hosts
- `youtu.be/VIDEO_ID`
- `youtube.com/shorts/VIDEO_ID`
- `youtube.com/embed/VIDEO_ID`
- `youtube.com/v/VIDEO_ID`
- Extra timestamp, tracking, and playlist query parameters when an individual
  video ID is present

Video identifiers must be exactly eleven valid YouTube ID characters.

## Unsupported cases

Playlist-only, channel, user, handle, search, feed, malformed, and non-YouTube
URLs are rejected before metadata retrieval. Active and upcoming streams,
private videos, removed videos, and inaccessible or restricted videos are
translated into application errors with concise CLI messages.

## Subtitle-selection algorithm

Language codes are trimmed, underscores become hyphens, the base language is
lowercase, and two-letter regions are uppercase. Selection priority is:

1. manual exact match;
2. manual base-language match;
3. automatic exact match;
4. automatic base-language match;
5. no match.

Within a priority, the generic base language is preferred; remaining tracks are
sorted by normalized code and then original code. Arabic regions naturally
match through their normalized `ar` base rather than a fixed region list.

## CLI commands added

```bash
captionforge inspect VIDEO_URL
captionforge inspect VIDEO_URL --language ar-EG
captionforge inspect VIDEO_URL --json
```

The existing help, version, config, doctor, and module entry points remain
available. Doctor now reports local yt-dlp installation and version information
without making a network request.

## Tests added

The offline suite covers URL forms and rejection cases, language normalization,
Arabic selection, all selection priorities and ties, metadata mapping, missing
fields, malformed caption entries, duplicate formats, private/unavailable and
generic extractor errors, live-video rejection, CLI help, human and JSON
output, no-subtitle results, exit codes, and Phase 1 regressions.

An optional marked integration test accepts
`CAPTIONFORGE_INTEGRATION_VIDEO_URL`; it is excluded from default test runs.

## Commands used to validate

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m ruff format app tests
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

## Test results

The latest completed default test run reported:

```text
65 passed, 1 deselected in 0.84s
```

Ruff formatting reported 38 files already formatted. Ruff lint reported all
checks passed. An explicit `pytest -m integration` run selected the optional
network test, which skipped because no public test URL was supplied:

```text
1 skipped, 65 deselected in 0.07s
```

Black 26.5.1 was invoked twice but did not complete in this environment and was
interrupted after producing no output. Ruff's Black-compatible formatter was
used successfully instead. No Black success is claimed.

## Known limitations

- Real inspection depends on network access, YouTube availability, and yt-dlp
  compatibility.
- No cookies or authenticated inspection is supported.
- Display names cover a small set of common languages.
- Caption availability can change between inspection and a future download.
- No subtitle, media, audio, or thumbnail files are downloaded.

## Recommendations for Phase 3

- Add a dedicated subtitle-download boundary that accepts the selected stable
  track model rather than leaking yt-dlp mappings.
- Validate subtitle MIME/extension and content before persistence.
- Use configuration-owned temporary and output paths with atomic writes.
- Preserve manual/automatic provenance through acquisition.
- Add fixture-based tests for empty, malformed, right-to-left, and long Arabic
  caption content before adding export or transcription.
