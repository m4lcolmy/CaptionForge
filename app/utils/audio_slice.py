"""Exact tail-slicing of prepared PCM audio, used to resume a failed run.

The audio handed to Whisper is already mono 16 kHz PCM WAV, so a cut at time T
is a byte offset rather than a re-encode: no FFmpeg process, no resampling, and
no drift between the slice and the timestamps that get shifted to match it.
"""

from __future__ import annotations

import contextlib
import wave
from pathlib import Path

from app.core.exceptions import AudioConversionError


def wav_duration_seconds(source: Path) -> float | None:
    """Return the duration of a PCM WAV file, or None if it cannot be read."""
    try:
        with contextlib.closing(wave.open(str(source), "rb")) as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, EOFError, wave.Error):
        return None


def slice_wav_tail(source: Path, destination: Path, start_seconds: float) -> float:
    """Write the part of ``source`` at or after ``start_seconds`` to ``destination``.

    Returns the exact start time of the written audio, which is the requested
    time rounded down to a whole frame. Callers must add that returned value -
    not the requested one - to the timestamps the engine reports, or resumed
    segments land fractionally early.
    """
    if start_seconds <= 0:
        raise ValueError("start_seconds must be positive to slice a tail")
    try:
        with contextlib.closing(wave.open(str(source), "rb")) as reader:
            rate = reader.getframerate()
            total_frames = reader.getnframes()
            if not rate:
                raise AudioConversionError(
                    "The prepared audio reports no sample rate, so it cannot "
                    "be resumed from a partial transcription."
                )
            offset_frames = min(int(start_seconds * rate), total_frames)
            reader.setpos(offset_frames)
            frames = reader.readframes(total_frames - offset_frames)
            parameters = reader.getparams()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(wave.open(str(destination), "wb")) as writer:
            writer.setnchannels(parameters.nchannels)
            writer.setsampwidth(parameters.sampwidth)
            writer.setframerate(rate)
            writer.writeframes(frames)
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioConversionError(
            "The prepared audio could not be sliced to resume transcription.",
            details=str(exc),
        ) from exc
    return offset_frames / rate


__all__ = ["slice_wav_tail", "wav_duration_seconds"]
