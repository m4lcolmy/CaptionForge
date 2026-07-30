"""Opt-in real-model smoke test; excluded from the default suite."""

import os
from pathlib import Path

import pytest

from app.adapters.whisper_adapter import WhisperAdapter


@pytest.mark.integration
def test_real_whisper_model() -> None:
    audio_value = os.getenv("CAPTIONFORGE_INTEGRATION_AUDIO")
    if not audio_value:
        pytest.skip("Set CAPTIONFORGE_INTEGRATION_AUDIO to a short local audio file")
    result = WhisperAdapter().transcribe(
        Path(audio_value),
        model_name=os.getenv("CAPTIONFORGE_INTEGRATION_MODEL", "tiny"),
        device="cpu",
        compute_type="int8",
    )
    assert result.segments
