# Phase 7 — Reliability and Testing

## Implemented changes

- Added a per-user JSON configuration file with `config show`, `config set`, and
  `config reset`. `CAPTIONFORGE_` environment variables take precedence.
- Invalid persisted fields recover independently to defaults. CLI updates are
  validated before an atomic write.
- Added configurable selective retries (`retry_count`, `retry_delay_seconds`)
  for temporary YouTube metadata, caption, audio, and Whisper model failures.
  Invalid input and permanent availability failures are not retried.
- Centralized expected, cancellation, and unexpected CLI error handling.
  Normal terminal output does not expose technical exception details.
- Output is flushed, synced, and atomically replaced. Failed multi-format
  exports remove newly created files and temporary `.partial` files.
- Added output/temp write validation, configurable free-space checks, corrupt
  and empty artifact checks, and best-effort cleanup.
- Expanded job state with current/failure stage, start/finish timestamps, and
  processing duration.
- Kept technical diagnostics in rotating 10 MB logs with 14-day retention and
  added job, method, video, retry, model/device, failure, and duration context.
- Added `pytest-cov` and `mypy` to development dependencies.

## Tests

The Phase 7 offline tests cover retry success, retry exhaustion boundaries,
non-retryable errors, configuration persistence, environment precedence,
invalid-value recovery, atomic replacement, partial-file absence, job state
transitions, config CLI commands, stable exit codes, and friendly unexpected
error output. Existing mocked tests continue to cover the complete caption-first
and Whisper-fallback workflows, cancellation, temporary cleanup, permissions,
disk space, exporters, adapters, services, and parsing.

Completion run:

```text
pytest: 124 passed, 2 optional integration tests deselected
coverage: 88% overall
ruff check: passed
ruff format --check: passed
mypy app: passed
```

Coverage and mypy are included in the updated `dev` extra:

```bash
python -m pip install -e ".[dev]"
python -m pytest --cov=app --cov-report=term-missing
python -m mypy app
```

## Optional integration tests

Real network/model tests remain marked `integration` and are offline by default.
Set `CAPTIONFORGE_INTEGRATION_VIDEO_URL` and/or
`CAPTIONFORGE_INTEGRATION_AUDIO`, then run `pytest -m integration`.

## Known limitations

- Cleanup cannot guarantee removal when the operating system denies deletion;
  the technical cause is logged.
- Existing files are protected by default. A system failure between atomic
  replacements in an explicit multi-format overwrite can leave a mixture of
  old and new complete files, but never a partially written individual file.
- Retry classification depends on errors translated by the local adapters;
  upstream services can introduce new message variants.
- Live integration, coverage, and type-check results depend on optional tools,
  network access, credentials/content availability, and local model resources.

## Phase 8 recommendations

Keep future work outside the Phase 7 core: packaging and release automation,
structured machine-readable job history, resumable downloads, and GUI-specific
progress/cancellation can be evaluated independently. Add platform CI for
Linux, macOS, and Windows with coverage thresholds and a scheduled opt-in live
smoke test.
