# Phase 6 — Arabic Post-Processing

## Scope

Phase 6 adds one conservative post-processing pipeline for downloaded YouTube
captions and local Whisper results. Every exporter receives the same cleaned
`SubtitleSegment` sequence. Translation, spelling correction, grammar rewriting,
speaker diarization, GUI work, and cloud processing remain out of scope.

## Processing pipeline

`PostProcessingService` performs deterministic operations in this order:

1. Decode caption markup, remove empty cue-only segments, collapse whitespace,
   and normalize safe punctuation spacing.
2. Remove exact consecutive duplicates, collapse adjacent repeated Whisper
   phrases, and remove repeated phrase overlap between neighboring segments.
3. Remove a silence cue only when Whisper reports a very high no-speech
   probability and the text is a known non-speech label.
4. Clamp negative times, repair invalid durations, and remove overlaps.
5. Merge short neighboring fragments when the gap, sentence boundary, and
   configured text limit make the merge safe.
6. Split long text near sentence or word boundaries and divide its timestamps
   proportionally.
7. Enforce configured duration limits and wrap output to at most two lines.

The pipeline never guesses corrected words. Arabic dialect, mixed Arabic-English
content, names, religious expressions, diacritics, and Arabic-Indic digits are
preserved by default.

## Configuration

Safe defaults are 42 characters per line, two lines, durations from 0.8 to 7
seconds, a 1-second merge threshold, and a 0.9 duplicate similarity threshold.
The following potentially destructive transforms are independently configurable
and disabled by default:

- Arabic diacritic removal
- Arabic letter normalization
- Arabic-Indic digit conversion

All settings can be supplied through the `CAPTIONFORGE_` environment prefix; see
`.env.example`.

## Integration and CLI

`SubtitleService` sends parsed SRT, VTT, and YouTube JSON3 candidates through the
processor. `TranscriptionService` converts Whisper output to `SubtitleSegment`,
including confidence and no-speech metadata, then runs the same processor.

Both `captionforge extract` and `captionforge transcribe` accept
`--no-postprocess`. The new `captionforge clean INPUT_FILE` command reads SRT or
VTT and writes the same format with a `.cleaned` filename by default.

## Verification

Offline tests cover Arabic and Latin punctuation, mixed text, diacritics,
Quranic text and names, whitespace, duplicates, Whisper repetition and silence
cues, empty input, timing repair, merging, splitting, line limits, disabled
processing, source workflows, and SRT/VTT cleanup. The Phase 6 completion run
passes pytest plus Ruff linting and formatting checks.
