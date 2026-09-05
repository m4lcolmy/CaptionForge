"""Free-VRAM probing and Whisper memory estimation used to plan before loading.

The point of this module is to answer "will this fit?" *before* CTranslate2
allocates anything, so the common out-of-memory case becomes a quiet downgrade
instead of a failed job. The estimate is deliberately approximate; it only has
to be good enough to pick a starting plan, because the degradation ladder in
:mod:`app.services.transcription_plan` still recovers when it guesses wrong.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

# NVML is queried instead of CUDA because ``cuMemGetInfo`` needs a CUDA context,
# and creating one costs a few hundred megabytes of the very resource being
# measured. NVML reads the driver directly and allocates no device memory.
_NVML_LIBRARIES = ("libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll")
_NVML_SUCCESS = 0

MEGABYTE = 1024 * 1024


class _NvmlMemory(ctypes.Structure):
    """Mirror of ``nvmlMemory_t`` from the NVML headers."""

    _fields_ = (
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


@dataclass(frozen=True)
class VramSnapshot:
    """What the driver reports for one device at one instant."""

    total_bytes: int
    free_bytes: int

    @property
    def used_bytes(self) -> int:
        return max(0, self.total_bytes - self.free_bytes)


@dataclass(frozen=True)
class _ModelProfile:
    """The three dimensions of a Whisper checkpoint that drive its footprint."""

    parameters: int
    decoder_layers: int
    model_dimension: int


# Published Whisper architecture sizes. Aliases resolve onto these below.
_MODEL_PROFILES: dict[str, _ModelProfile] = {
    "tiny": _ModelProfile(39_000_000, 4, 384),
    "base": _ModelProfile(74_000_000, 6, 512),
    "small": _ModelProfile(244_000_000, 12, 768),
    "medium": _ModelProfile(769_000_000, 24, 1024),
    "large": _ModelProfile(1_550_000_000, 32, 1280),
    # turbo keeps the large encoder but ships four decoder layers instead of 32.
    "turbo": _ModelProfile(809_000_000, 4, 1280),
}

_BYTES_PER_PARAMETER: dict[str, float] = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int16": 2.0,
    # Mixed modes keep a minority of tensors at higher precision.
    "int8_float32": 1.4,
    "int8_float16": 1.2,
    "int8": 1.0,
    "default": 2.0,
}

# The CUDA context plus the cuBLAS and cuDNN kernel workspaces CTranslate2
# faults in on first use. Independent of model and audio length.
_RUNTIME_OVERHEAD_BYTES = 512 * MEGABYTE

# Whisper decodes at most 448 tokens per 30-second window, and CTranslate2
# holds a self- and cross-attention key/value pair per decoder layer per beam.
_MAX_DECODE_TOKENS = 448
_KV_TENSORS_PER_LAYER = 4  # self key, self value, cross key, cross value
_KV_BYTES_PER_ELEMENT = 2

# Cross-attention weights retained for the dynamic-time-warping alignment that
# produces word timings, as a fraction of the encoder activation footprint.
_WORD_TIMESTAMP_FACTOR = 0.45


def resolve_profile(model_name: str) -> _ModelProfile:
    """Map any checkpoint name or local path onto the closest known profile.

    Unknown names fall back to ``large`` so an unrecognised checkpoint is
    budgeted pessimistically rather than optimistically.
    """
    stem = os.path.basename(model_name.rstrip("/\\")).lower()
    if "turbo" in stem:
        return _MODEL_PROFILES["turbo"]
    for size in ("large", "medium", "small", "base", "tiny"):
        if size in stem:
            profile = _MODEL_PROFILES[size]
            # Distilled checkpoints keep the layer shape but drop decoder depth.
            if "distil" in stem and size in {"large", "medium"}:
                return _ModelProfile(
                    int(profile.parameters * 0.49),
                    max(2, profile.decoder_layers // 8),
                    profile.model_dimension,
                )
            return profile
    return _MODEL_PROFILES["large"]


def estimate_vram_bytes(
    model_name: str,
    *,
    compute_type: str,
    beam_size: int,
    word_timestamps: bool,
) -> int:
    """Estimate peak device memory for one 30-second Whisper window.

    Nothing here scales with the length of the audio: Whisper is a fixed
    30-second-window model, so an eight-hour file and a one-minute file reach
    the same peak. The tunable terms are the weights, the per-beam key/value
    cache, and the word-alignment buffers.
    """
    profile = resolve_profile(model_name)
    per_parameter = _BYTES_PER_PARAMETER.get(compute_type.lower(), 2.0)
    weights = int(profile.parameters * per_parameter)

    kv_per_beam = (
        _KV_TENSORS_PER_LAYER
        * profile.decoder_layers
        * profile.model_dimension
        * _MAX_DECODE_TOKENS
        * _KV_BYTES_PER_ELEMENT
    )
    decoder = kv_per_beam * max(1, beam_size)

    # Encoder activations for the 1500-frame mel window, all layers live.
    encoder = profile.model_dimension * 1500 * profile.decoder_layers * 4
    alignment = int(encoder * _WORD_TIMESTAMP_FACTOR) if word_timestamps else 0

    return weights + decoder + encoder + alignment + _RUNTIME_OVERHEAD_BYTES


def read_vram(device_index: int = 0) -> VramSnapshot | None:
    """Return the driver's memory report, or None when NVML cannot answer.

    A None result means "unknown", never "empty": callers must treat it as a
    reason to skip the pre-flight check, not as a reason to refuse the GPU.
    """
    library = _load_nvml()
    if library is None:
        return None
    if library.nvmlInit_v2() != _NVML_SUCCESS:
        return None
    try:
        handle = ctypes.c_void_p()
        status = library.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(device_index), ctypes.byref(handle)
        )
        if status != _NVML_SUCCESS:
            return None
        memory = _NvmlMemory()
        if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)):
            return None
        return VramSnapshot(total_bytes=int(memory.total), free_bytes=int(memory.free))
    except (OSError, AttributeError):
        return None
    finally:
        with _suppressed():
            library.nvmlShutdown()


def _load_nvml() -> ctypes.CDLL | None:
    """Open the NVIDIA management library shipped with the driver."""
    for name in _NVML_LIBRARIES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class _suppressed:
    """Tiny context manager so NVML teardown never masks a real result."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exception: object) -> bool:
        return True


def format_bytes(value: int) -> str:
    """Render a byte count the way the CLI and logs present it."""
    if value >= 1024 * MEGABYTE:
        return f"{value / (1024 * MEGABYTE):.2f} GiB"
    return f"{value / MEGABYTE:.0f} MiB"
