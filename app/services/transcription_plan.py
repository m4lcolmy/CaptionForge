"""Ordered degradation of Whisper settings so a GPU job downgrades, never fails.

Two mechanisms live here. :func:`select_plan` is the pre-flight: given the
memory the driver reports free, it picks the richest plan that is predicted to
fit, so most jobs never touch an out-of-memory path at all. :func:`build_ladder`
is the recovery: free memory is a moving target and fragmentation means "free"
is not "allocatable", so when a plan does fail the caller walks to the next
rung. Every ladder ends on CPU, which cannot exhaust device memory, and that
terminal rung is what makes the job finish rather than fail.

Rungs are ordered by quality lost per byte saved:

1. ``beam_size`` down to 1 - frees the largest tunable block, the per-beam
   key/value cache, for a small accuracy cost.
2. ``float16`` to ``int8_float16`` to ``int8`` - roughly halves the weights.
3. word timestamps off - frees the alignment buffers; subtitle split points get
   less precise, but the transcript text is untouched.
4. a smaller checkpoint - the first step that changes what is actually heard.
5. CPU - unbounded memory, no accuracy cost, an order of magnitude slower.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.adapters.gpu_memory import estimate_vram_bytes, format_bytes

# Ordered largest to smallest; a downgrade moves one position right. Every
# large-* revision shares one rung: they are the same size, so swapping between
# them frees nothing and costs a second multi-gigabyte download.
_MODEL_SIZES = ("large-v3", "medium", "small", "base", "tiny")

# Quantisation steps, each roughly halving the weight footprint.
_COMPUTE_STEPS = ("float32", "float16", "int8_float16", "int8")

# A slice of device memory left unclaimed. Absorbs allocator fragmentation and
# whatever the desktop compositor takes while the job runs.
DEFAULT_HEADROOM_BYTES = 384 * 1024 * 1024


@dataclass(frozen=True)
class TranscriptionPlan:
    """One complete, runnable set of engine settings."""

    model_name: str
    device: str
    compute_type: str
    beam_size: int
    word_timestamps: bool

    @property
    def estimated_vram_bytes(self) -> int:
        """Predicted peak device memory, or zero for a CPU plan."""
        if self.device != "cuda":
            return 0
        return estimate_vram_bytes(
            self.model_name,
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
        )

    def fits_within(self, free_bytes: int, headroom: int) -> bool:
        """Whether this plan is predicted to fit in the memory reported free."""
        if self.device != "cuda":
            return True
        return self.estimated_vram_bytes + headroom <= free_bytes

    def describe(self) -> str:
        """A one-line rendering for logs, progress messages, and the CLI."""
        words = "word timings" if self.word_timestamps else "no word timings"
        summary = (
            f"{self.model_name} on {self.device} "
            f"({self.compute_type}, beam {self.beam_size}, {words})"
        )
        if self.device == "cuda":
            return f"{summary}, needs about {format_bytes(self.estimated_vram_bytes)}"
        return summary


def build_ladder(
    initial: TranscriptionPlan,
    *,
    allow_word_timestamp_loss: bool = True,
    allow_model_downgrade: bool = True,
    allow_cpu_fallback: bool = True,
    minimum_model: str = "small",
) -> tuple[TranscriptionPlan, ...]:
    """Return ``initial`` followed by every cheaper plan, in order of preference.

    Each rung applies one more reduction than the rung before it, so the walk
    gives up exactly as much quality as the hardware forces and no more. A CPU
    plan is appended last and keeps the originally requested checkpoint: by that
    point every faster option has been tried, and finishing slowly at full
    accuracy beats failing.
    """
    if initial.device != "cuda":
        return (initial,)

    rungs = [initial]
    current = initial

    for beam in (2, 1):
        if current.beam_size > beam:
            current = replace(current, beam_size=beam)
            rungs.append(current)

    for compute in _quantisation_steps_below(current.compute_type):
        current = replace(current, compute_type=compute)
        rungs.append(current)

    if allow_word_timestamp_loss and current.word_timestamps:
        current = replace(current, word_timestamps=False)
        rungs.append(current)

    if allow_model_downgrade:
        for smaller in _smaller_models(current.model_name, minimum_model):
            current = replace(current, model_name=smaller)
            rungs.append(current)

    if allow_cpu_fallback:
        rungs.append(
            TranscriptionPlan(
                model_name=initial.model_name,
                device="cpu",
                compute_type="int8",
                beam_size=initial.beam_size,
                word_timestamps=initial.word_timestamps,
            )
        )

    return tuple(_deduplicated(rungs))


def select_plan(
    ladder: tuple[TranscriptionPlan, ...],
    free_bytes: int | None,
    *,
    headroom: int = DEFAULT_HEADROOM_BYTES,
) -> TranscriptionPlan:
    """Pick the first rung predicted to fit in the memory reported free.

    ``free_bytes`` of None means the driver could not be queried. That is not
    evidence of scarcity, so the requested plan is used unchanged and the ladder
    handles any failure after the fact.
    """
    if free_bytes is None:
        return ladder[0]
    for plan in ladder:
        if plan.fits_within(free_bytes, headroom):
            return plan
    return ladder[-1]


def remaining_after(
    ladder: tuple[TranscriptionPlan, ...], plan: TranscriptionPlan
) -> tuple[TranscriptionPlan, ...]:
    """Return the rungs below ``plan``, so a caller resumes where it left off."""
    for position, candidate in enumerate(ladder):
        if candidate == plan:
            return ladder[position + 1 :]
    return ()


def _quantisation_steps_below(compute_type: str) -> tuple[str, ...]:
    """Return the quantisation steps cheaper than the current one."""
    normalized = compute_type.lower()
    if normalized in {"default", "bfloat16", "int16"}:
        # Not on the ladder proper; enter it at the first strictly cheaper step.
        return ("int8_float16", "int8")
    if normalized not in _COMPUTE_STEPS:
        return ()
    return _COMPUTE_STEPS[_COMPUTE_STEPS.index(normalized) + 1 :]


def _smaller_models(model_name: str, minimum: str) -> tuple[str, ...]:
    """Return the standard checkpoints below ``model_name``, down to ``minimum``.

    A checkpoint that is not one of the published sizes - a local path or a
    fine-tune - has no meaningful "one size smaller", so nothing is offered and
    the ladder moves straight to the CPU rung.
    """
    position = _size_position(model_name)
    if position is None:
        return ()
    floor = _size_position(minimum)
    stop = len(_MODEL_SIZES) if floor is None else floor + 1
    return _MODEL_SIZES[position + 1 : stop]


def _size_position(model_name: str) -> int | None:
    """Locate a checkpoint on the size ladder, tolerating version suffixes."""
    normalized = model_name.lower()
    for index, size in enumerate(_MODEL_SIZES):
        if normalized == size:
            return index
    # Every large revision, and the turbo distillation of it, sits at the top.
    if normalized.startswith("large"):
        return 0
    for index, size in enumerate(_MODEL_SIZES):
        if normalized.startswith(size.split("-")[0]):
            return index
    return None


def _deduplicated(
    plans: list[TranscriptionPlan],
) -> list[TranscriptionPlan]:
    """Drop rungs that repeat an earlier one, preserving order."""
    seen: set[TranscriptionPlan] = set()
    unique: list[TranscriptionPlan] = []
    for plan in plans:
        if plan not in seen:
            seen.add(plan)
            unique.append(plan)
    return unique


__all__ = [
    "DEFAULT_HEADROOM_BYTES",
    "TranscriptionPlan",
    "build_ladder",
    "remaining_after",
    "select_plan",
]
