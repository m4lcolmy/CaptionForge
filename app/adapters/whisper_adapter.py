"""Lazy faster-whisper adapter with device selection and error translation."""

from __future__ import annotations

import ctypes
import gc
import importlib
from collections.abc import Callable
from contextlib import suppress
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
from app.models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)

ProgressCallback = Callable[[str, float | None], None]
CancelCallback = Callable[[], bool]
# Invoked for each accepted segment as it streams out of the engine, so a
# caller can checkpoint finished work before a later failure discards it.
SegmentCallback = Callable[[TranscriptionSegment], None]

# The families CTranslate2 dlopens by bare soname, in dependency order:
# libcublas needs libcublasLt, so that one has to be resident first.
_BUNDLED_CUDA_LIBRARIES = (
    "libcublasLt.so.*",
    "libcublas.so.*",
    "libcudnn.so.*",
)
_cuda_preload_done = False


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
        """Return whether CUDA can actually run, not merely that a GPU exists.

        A driver can report devices while the CUDA math libraries CTranslate2
        links against are absent. Counting devices alone sends ``auto`` to a
        device that fails at inference, so both must hold.
        """
        if self._cuda_detector is not None:
            return self._cuda_detector()
        return self.cuda_device_present() and not self.missing_cuda_libraries()

    @staticmethod
    def cuda_device_present() -> bool:
        """Return whether CTranslate2 reports at least one CUDA device."""
        try:
            ctranslate2 = importlib.import_module("ctranslate2")
            return bool(ctranslate2.get_cuda_device_count())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def missing_cuda_libraries() -> tuple[str, ...]:
        """Return the CUDA library families that cannot be loaded."""
        _preload_bundled_cuda_libraries()
        families = {
            "cuBLAS": ("libcublas.so.12", "libcublas.so.11", "libcublas.so"),
            "cuDNN": ("libcudnn.so.9", "libcudnn.so.8", "libcudnn.so"),
        }
        missing = []
        for name, candidates in families.items():
            if not any(_library_loads(candidate) for candidate in candidates):
                missing.append(name)
        return tuple(missing)

    def select_device(self, requested: str) -> str:
        """Resolve auto/cpu/cuda, rejecting unavailable explicit CUDA."""
        normalized = requested.lower()
        if normalized == "auto":
            return "cuda" if self.cuda_available() else "cpu"
        if normalized == "cuda" and not self.cuda_available():
            missing = self.missing_cuda_libraries()
            if self._cuda_detector is None and self.cuda_device_present() and missing:
                raise CudaUnavailableError(
                    f"A CUDA GPU was detected but {' and '.join(missing)} could "
                    "not be loaded. Install the matching NVIDIA runtime "
                    "libraries, or run with --device cpu."
                )
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
        vad_threshold: float = 0.5,
        vad_speech_pad_ms: int = 400,
        condition_on_previous_text: bool = False,
        initial_prompt: str | None = None,
        word_timestamps: bool = True,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        hallucination_silence_threshold: float | None = None,
        download_root: Path | None = None,
        time_offset_seconds: float = 0.0,
        first_segment_index: int = 1,
        total_duration_seconds: float | None = None,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
        on_segment: SegmentCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file and convert all output to local models.

        ``time_offset_seconds`` and ``first_segment_index`` place the result
        inside a longer recording, so a tail sliced off a partially
        transcribed file continues the original numbering and timeline
        rather than restarting at zero.
        """
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
            # hallucination_silence_threshold is only honoured alongside word
            # timestamps, so requesting one implies the other.
            aligned_words = word_timestamps or (
                hallucination_silence_threshold is not None
            )
            raw_segments, info = model.transcribe(
                str(audio_path),
                language=language,
                beam_size=beam_size,
                vad_filter=vad_enabled,
                vad_parameters=(
                    {
                        "min_silence_duration_ms": min_silence_duration_ms,
                        "threshold": vad_threshold,
                        "speech_pad_ms": vad_speech_pad_ms,
                    }
                    if vad_enabled
                    else None
                ),
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                word_timestamps=aligned_words,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                hallucination_silence_threshold=hallucination_silence_threshold,
            )
            detected_language = str(getattr(info, "language", language or "unknown"))
            probability = _optional_float(getattr(info, "language_probability", None))
            duration = _optional_float(
                getattr(info, "duration", None)
                or getattr(info, "duration_after_vad", None)
            )
            # A resumed run only sees its own slice, so progress is measured
            # against the whole recording when the caller knows that length.
            span = total_duration_seconds or (
                duration + time_offset_seconds if duration else None
            )
            converted: list[TranscriptionSegment] = []
            for fallback_index, segment in enumerate(raw_segments, start=1):
                if is_cancelled():
                    raise TranscriptionCancelledError("Transcription was cancelled.")
                text = str(getattr(segment, "text", "")).strip()
                if not text:
                    continue
                start = float(segment.start) + time_offset_seconds
                end = float(segment.end) + time_offset_seconds
                accepted = TranscriptionSegment(
                    index=first_segment_index + len(converted),
                    start_seconds=max(0.0, start),
                    end_seconds=max(end, start + 0.001),
                    text=text,
                    language=detected_language,
                    confidence=_confidence(segment),
                    no_speech_probability=_optional_float(
                        getattr(segment, "no_speech_prob", None)
                    ),
                    words=_word_timings(segment, time_offset_seconds),
                )
                converted.append(accepted)
                # Publish before the next decode step can fail, so whatever
                # reaches here survives an out-of-memory abort further on.
                if on_segment is not None:
                    on_segment(accepted)
                if span and span > 0:
                    percent = 40.0 + min(45.0, max(0.0, end / span * 45.0))
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
                duration_seconds=span if span else duration,
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
        if "is not found or cannot be loaded" in lowered or any(
            library in lowered for library in ("libcublas", "libcudnn", "libcuda")
        ):
            raise CudaUnavailableError(
                "The GPU could not be used because a required NVIDIA runtime "
                "library is missing. Install the matching CUDA libraries, or "
                "run with --device cpu.",
                details=message,
            ) from exc
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


def _word_timings(segment: Any, offset_seconds: float = 0.0) -> tuple[WordTiming, ...]:
    """Convert engine word alignments, skipping anything malformed."""
    words = getattr(segment, "words", None) or ()
    converted: list[WordTiming] = []
    for word in words:
        text = str(getattr(word, "word", "")).strip()
        if not text:
            continue
        try:
            start = max(0.0, float(word.start) + offset_seconds)
            end = max(start, float(word.end) + offset_seconds)
        except (TypeError, ValueError):
            continue
        converted.append(
            WordTiming(
                start_seconds=start,
                end_seconds=end,
                text=text,
                probability=_optional_float(getattr(word, "probability", None)),
            )
        )
    return tuple(converted)


def _preload_bundled_cuda_libraries() -> None:
    """Make pip's nvidia-*-cu12 wheels reachable without LD_LIBRARY_PATH.

    Those wheels install their shared objects under ``site-packages/nvidia/*/lib``,
    a directory the dynamic loader never searches, so CTranslate2's lazy dlopen
    by bare soname fails even though the libraries are installed. Loading each
    one here by absolute path puts its soname in the process, and the later
    dlopen resolves to the copy already resident. Each wheel sets ``RPATH
    $ORIGIN``, so the sibling libraries it needs resolve from the same folder.

    Runs at most once per process, and does nothing when the wheels are absent.
    """
    global _cuda_preload_done
    if _cuda_preload_done:
        return
    _cuda_preload_done = True
    try:
        nvidia = importlib.import_module("nvidia")
    except ImportError:
        return
    for root in getattr(nvidia, "__path__", []):
        for pattern in _BUNDLED_CUDA_LIBRARIES:
            for library in sorted(Path(root).glob(f"*/lib/{pattern}")):
                with suppress(OSError):
                    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _library_loads(soname: str) -> bool:
    """Return whether a shared library can be resolved and loaded.

    CTranslate2 dlopens the CUDA math libraries lazily by bare soname, so this
    resolves them exactly the way it does. Probing by full path instead would
    report libraries as present that CTranslate2 still cannot open — for
    example pip's nvidia-*-cu12 wheels when LD_LIBRARY_PATH is unset.
    """
    try:
        ctypes.CDLL(soname)
    except OSError:
        return False
    return True


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _confidence(segment: Any) -> float | None:
    avg_logprob = _optional_float(getattr(segment, "avg_logprob", None))
    if avg_logprob is None:
        return None
    # A bounded, useful score; it is intentionally not presented as calibrated.
    import math

    return max(0.0, min(1.0, math.exp(avg_logprob)))
