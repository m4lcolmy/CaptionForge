"""Lazy faster-whisper adapter with device selection and error translation."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    AudioTranscriptionError,
    CudaUnavailableError,
    EmptyTranscriptionError,
    GpuMemoryError,
    InvalidComputeTypeError,
    ModelLoadError,
    TranscriptionCancelledError,
    UnsupportedModelError,
    WhisperNotInstalledError,
)
from app.models.transcription import TranscriptionResult, TranscriptionSegment

ProgressCallback = Callable[[str, float | None], None]
CancelCallback = Callable[[], bool]


class WhisperAdapter:
    """Load, run, and release faster-whisper without leaking engine objects."""

    def __init__(
        self,
        *,
        model_factory: Callable[..., Any] | None = None,
        cuda_detector: Callable[[], bool] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._cuda_detector = cuda_detector

    def cuda_available(self) -> bool:
        """Return whether CTranslate2 reports at least one CUDA device."""
        if self._cuda_detector is not None:
            return self._cuda_detector()
        try:
            ctranslate2 = importlib.import_module("ctranslate2")
            return bool(ctranslate2.get_cuda_device_count())
        except (ImportError, AttributeError, RuntimeError):
            return False

    def select_device(self, requested: str) -> str:
        """Resolve auto/cpu/cuda, rejecting unavailable explicit CUDA."""
        normalized = requested.lower()
        if normalized == "auto":
            return "cuda" if self.cuda_available() else "cpu"
        if normalized == "cuda" and not self.cuda_available():
            raise CudaUnavailableError(
                "CUDA was requested, but no compatible NVIDIA GPU was detected."
            )
        if normalized not in {"cpu", "cuda"}:
            raise CudaUnavailableError(
                f"Unsupported transcription device '{requested}'."
            )
        return normalized

    @staticmethod
    def select_compute_type(requested: str, device: str) -> str:
        """Choose conservative CPU and performant CUDA defaults."""
        normalized = requested.lower()
        if normalized == "auto":
            return "float16" if device == "cuda" else "int8"
        allowed = {
            "default",
            "int8",
            "int8_float16",
            "int8_float32",
            "int16",
            "float16",
            "float32",
            "bfloat16",
        }
        if normalized not in allowed:
            raise InvalidComputeTypeError(
                f"Unsupported Whisper compute type '{requested}'."
            )
        return normalized

    def transcribe(
        self,
        audio_path: Path,
        *,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
        beam_size: int = 5,
        vad_enabled: bool = True,
        min_silence_duration_ms: int = 500,
        download_root: Path | None = None,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file and convert all output to local models."""
        notify = progress or (lambda _message, _percent: None)
        is_cancelled = cancelled or (lambda: False)
        selected_device = self.select_device(device)
        selected_compute = self.select_compute_type(compute_type, selected_device)
        model: Any = None
        raw_segments: Any = None
        try:
            if is_cancelled():
                raise TranscriptionCancelledError("Transcription was cancelled.")
            notify("Loading model", 30.0)
            factory = self._model_factory or self._import_model_factory()
            kwargs: dict[str, Any] = {
                "device": selected_device,
                "compute_type": selected_compute,
            }
            if download_root is not None:
                kwargs["download_root"] = str(download_root)
            model = factory(model_name, **kwargs)
            notify("Transcribing", 40.0)
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=vad_enabled,
                vad_parameters={"min_silence_duration_ms": min_silence_duration_ms}
                if vad_enabled
                else None,
            )
            detected_language = str(getattr(info, "language", language or "unknown"))
            probability = _optional_float(getattr(info, "language_probability", None))
            duration = _optional_float(
                getattr(info, "duration", None)
                or getattr(info, "duration_after_vad", None)
            )
            converted: list[TranscriptionSegment] = []
            for fallback_index, segment in enumerate(raw_segments, start=1):
                if is_cancelled():
                    raise TranscriptionCancelledError("Transcription was cancelled.")
                text = str(getattr(segment, "text", "")).strip()
                if not text:
                    continue
                start = float(segment.start)
                end = float(segment.end)
                converted.append(
                    TranscriptionSegment(
                        index=len(converted) + 1,
                        start_seconds=max(0.0, start),
                        end_seconds=max(end, start + 0.001),
                        text=text,
                        language=detected_language,
                        confidence=_confidence(segment),
                        no_speech_probability=_optional_float(
                            getattr(segment, "no_speech_prob", None)
                        ),
                    )
                )
                if duration and duration > 0:
                    percent = 40.0 + min(45.0, max(0.0, end / duration * 45.0))
                else:
                    percent = min(84.0, 40.0 + fallback_index)
                notify("Transcribing", percent)
            if not converted:
                raise EmptyTranscriptionError(
                    "Whisper found no usable speech in the prepared audio."
                )
            return TranscriptionResult(
                segments=tuple(converted),
                detected_language=detected_language,
                language_probability=probability,
                duration_seconds=duration,
                model_name=model_name,
                device=selected_device,
                compute_type=selected_compute,
            )
        except (
            TranscriptionCancelledError,
            EmptyTranscriptionError,
            CudaUnavailableError,
            InvalidComputeTypeError,
            WhisperNotInstalledError,
        ):
            raise
        except Exception as exc:
            self._translate_error(exc, loading=model is None)
            raise AssertionError("unreachable") from exc
        finally:
            close = getattr(raw_segments, "close", None)
            if callable(close):
                close()
            raw_segments = None
            model = None
            gc.collect()

    @staticmethod
    def _import_model_factory() -> Callable[..., Any]:
        try:
            module = importlib.import_module("faster_whisper")
            return module.WhisperModel
        except (ImportError, AttributeError) as exc:
            raise WhisperNotInstalledError(
                "faster-whisper is not installed. Install CaptionForge's "
                "transcription dependencies before using this command."
            ) from exc

    @staticmethod
    def _translate_error(exc: Exception, *, loading: bool) -> None:
        message = str(exc)
        lowered = message.lower()
        if "out of memory" in lowered or "cuda_error_out_of_memory" in lowered:
            raise GpuMemoryError(
                "The GPU ran out of memory. Try a smaller model, int8 compute, or CPU.",
                details=message,
            ) from exc
        if "compute type" in lowered or "quantization" in lowered:
            raise InvalidComputeTypeError(
                "The selected compute type is not supported on this device.",
                details=message,
            ) from exc
        if loading and any(
            marker in lowered
            for marker in ("invalid model", "model not found", "repository not found")
        ):
            raise UnsupportedModelError(
                "The requested Whisper model name or path is not supported.",
                details=message,
            ) from exc
        if loading:
            raise ModelLoadError(
                "The Whisper model could not be loaded or downloaded. Check the "
                "model name, internet connection, disk space, and permissions.",
                details=message,
            ) from exc
        raise AudioTranscriptionError(
            "Whisper could not read or transcribe the prepared audio.",
            details=message,
        ) from exc


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _confidence(segment: Any) -> float | None:
    avg_logprob = _optional_float(getattr(segment, "avg_logprob", None))
    if avg_logprob is None:
        return None
    # A bounded, useful score; it is intentionally not presented as calibrated.
    import math

    return max(0.0, min(1.0, math.exp(avg_logprob)))
