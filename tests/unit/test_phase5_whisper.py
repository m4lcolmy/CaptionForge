"""Offline Phase 5 adapter and workflow coverage."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.adapters.whisper_adapter import WhisperAdapter
from app.core.config import Config
from app.core.exceptions import (
    CudaUnavailableError,
    EmptyTranscriptionError,
    GpuMemoryError,
    InvalidComputeTypeError,
    ModelLoadError,
    TranscriptionCancelledError,
    UnsupportedModelError,
)
from app.models.subtitle import (
    RawSubtitle,
    SubtitleDiscoveryResult,
)
from app.models.transcription import TranscriptionResult, TranscriptionSegment
from app.services.export_service import ExportService
from app.services.subtitle_service import SubtitleService
from app.services.transcription_service import TranscriptionService
from tests.conftest import VIDEO_URL, make_track


class FakeModel:
    """Minimal faster-whisper model with inspectable calls."""

    calls: list[dict[str, Any]] = []
    segments: list[Any] = [
        SimpleNamespace(
            id=0,
            start=0.0,
            end=1.5,
            text=" مرحبا بالعالم ",
            avg_logprob=-0.1,
            no_speech_prob=0.02,
        )
    ]

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.calls.append({"load_name": name, **kwargs})

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append({"audio": audio, **kwargs})
        return iter(self.segments), SimpleNamespace(
            language="ar", language_probability=0.97, duration=3.0
        )


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    FakeModel.calls = []
    FakeModel.segments = [
        SimpleNamespace(
            id=0,
            start=0.0,
            end=1.5,
            text=" مرحبا بالعالم ",
            avg_logprob=-0.1,
            no_speech_prob=0.02,
        )
    ]


def adapter(cuda: bool = False) -> WhisperAdapter:
    return WhisperAdapter(model_factory=FakeModel, cuda_detector=lambda: cuda)


def test_cpu_and_cuda_auto_selection() -> None:
    assert adapter(False).select_device("auto") == "cpu"
    assert adapter(True).select_device("auto") == "cuda"
    with pytest.raises(CudaUnavailableError):
        adapter(False).select_device("cuda")


def test_compute_type_selection_and_validation() -> None:
    assert WhisperAdapter.select_compute_type("auto", "cpu") == "int8"
    assert WhisperAdapter.select_compute_type("auto", "cuda") == "float16"
    assert WhisperAdapter.select_compute_type("float32", "cpu") == "float32"
    with pytest.raises(InvalidComputeTypeError):
        WhisperAdapter.select_compute_type("nope", "cpu")


def test_model_loading_arabic_vad_and_segment_conversion(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    progress: list[tuple[str, float | None]] = []
    result = adapter().transcribe(
        audio,
        model_name="small",
        language="ar",
        beam_size=7,
        vad_enabled=True,
        min_silence_duration_ms=650,
        download_root=tmp_path / "models",
        progress=lambda message, percent: progress.append((message, percent)),
    )

    assert FakeModel.calls[0] == {
        "load_name": "small",
        "device": "cpu",
        "compute_type": "int8",
        "download_root": str(tmp_path / "models"),
    }
    call = FakeModel.calls[1]
    assert call["language"] == "ar"
    assert call["beam_size"] == 7
    assert call["vad_filter"] is True
    assert call["vad_parameters"]["min_silence_duration_ms"] == 650
    assert result.detected_language == "ar"
    assert result.language_probability == 0.97
    assert result.segments[0].index == 1
    assert result.segments[0].text == "مرحبا بالعالم"
    assert result.segments[0].confidence == pytest.approx(0.9048, rel=0.001)
    assert [item[0] for item in progress][:2] == ["Loading model", "Transcribing"]


def test_vad_can_be_disabled(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()
    adapter().transcribe(audio, model_name="tiny", vad_enabled=False)
    assert FakeModel.calls[1]["vad_filter"] is False
    assert FakeModel.calls[1]["vad_parameters"] is None


def test_empty_segments_and_cancellation(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.touch()
    FakeModel.segments = []
    with pytest.raises(EmptyTranscriptionError):
        adapter().transcribe(audio, model_name="small")
    with pytest.raises(TranscriptionCancelledError):
        adapter().transcribe(audio, model_name="small", cancelled=lambda: True)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("network connection reset", ModelLoadError),
        ("invalid model repository not found", UnsupportedModelError),
        ("CUDA out of memory", GpuMemoryError),
        ("unsupported compute type", InvalidComputeTypeError),
    ],
)
def test_model_loading_errors_are_translated(
    message: str, expected: type[Exception], tmp_path: Path
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(message)

    with pytest.raises(expected):
        WhisperAdapter(
            model_factory=fail, cuda_detector=lambda: "CUDA" in message
        ).transcribe(tmp_path / "audio.wav", model_name="bad", device="auto")


class WorkflowVideoService:
    def __init__(self, discovery: SubtitleDiscoveryResult) -> None:
        self.discovery = discovery

    def inspect(
        self, _url: str, _language: str, *, allow_translated: bool = False
    ) -> SubtitleDiscoveryResult:
        return self.discovery


class WorkflowYtDlp:
    def download_subtitle(self, _video_id: str, _track: Any) -> RawSubtitle:
        return RawSubtitle(
            format="vtt",
            content="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nقديم\n",
        )


class WorkflowAudio:
    calls = 0

    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(self, *_args: Any, **_kwargs: Any) -> Path:
        self.calls += 1
        job = self.root / "captionforge-test"
        job.mkdir()
        audio = job / "prepared.wav"
        audio.write_bytes(b"RIFF")
        return audio


class WorkflowWhisper:
    calls = 0

    def select_device(self, requested: str) -> str:
        return "cpu" if requested == "auto" else requested

    @staticmethod
    def select_compute_type(requested: str, device: str) -> str:
        return WhisperAdapter.select_compute_type(requested, device)

    def transcribe(self, _audio: Path, **_kwargs: Any) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(
            segments=(
                TranscriptionSegment(
                    index=1,
                    start_seconds=0,
                    end_seconds=1,
                    text="جديد",
                    language="ar",
                ),
            ),
            detected_language="ar",
            language_probability=0.99,
            model_name=str(_kwargs["model_name"]),
            device="cpu",
            compute_type="int8",
        )


def make_workflow(
    tmp_path: Path, video_metadata: Any, *, captions: bool
) -> tuple[TranscriptionService, WorkflowAudio, WorkflowWhisper]:
    track = make_track("ar") if captions else None
    discovery = SubtitleDiscoveryResult(
        video=video_metadata,
        selected_track=track,
        preferred_language="ar",
    )
    audio = WorkflowAudio(tmp_path)
    whisper = WorkflowWhisper()
    ytdlp = WorkflowYtDlp()
    service = TranscriptionService(
        WorkflowVideoService(discovery),  # type: ignore[arg-type]
        ytdlp,  # type: ignore[arg-type]
        SubtitleService(),
        audio,  # type: ignore[arg-type]
        whisper,  # type: ignore[arg-type]
        ExportService(),
        Config(default_output_folder=tmp_path),
    )
    return service, audio, whisper


def test_caption_first_and_forced_transcription(
    tmp_path: Path, video_metadata: Any
) -> None:
    service, audio, whisper = make_workflow(tmp_path, video_metadata, captions=True)
    caption_result = service.process(
        VIDEO_URL, formats=("txt",), output_directory=tmp_path
    )
    assert caption_result.used_existing_captions is True
    assert audio.calls == whisper.calls == 0
    assert caption_result.paths[0].read_text(encoding="utf-8").strip() == "قديم"

    forced = service.process(
        VIDEO_URL,
        force=True,
        formats=("srt", "vtt", "txt", "json"),
        output_directory=tmp_path,
        overwrite=True,
    )
    assert forced.used_existing_captions is False
    assert audio.calls == whisper.calls == 1
    assert {path.suffix for path in forced.paths} == {".srt", ".vtt", ".txt", ".json"}
    assert "جديد" in forced.paths[0].read_text(encoding="utf-8")
    assert not (tmp_path / "captionforge-test").exists()


def test_workflow_progress_and_keep_audio(tmp_path: Path, video_metadata: Any) -> None:
    service, _, _ = make_workflow(tmp_path, video_metadata, captions=False)
    messages: list[str] = []
    result = service.process(
        VIDEO_URL,
        formats=("txt",),
        keep_audio=True,
        progress=lambda message, _percent: messages.append(message),
    )
    assert result.prepared_audio_path is not None
    assert result.prepared_audio_path.exists()
    assert messages[0] == "Inspecting video"
    assert messages[-1] == "Completed"


def test_workflow_cancellation(tmp_path: Path, video_metadata: Any) -> None:
    service, audio, _ = make_workflow(tmp_path, video_metadata, captions=False)
    with pytest.raises(TranscriptionCancelledError):
        service.process(VIDEO_URL, cancelled=lambda: True)
    assert audio.calls == 0


def test_cuda_needs_loadable_libraries_not_just_a_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A driver reporting a GPU is not proof CUDA can run."""
    monkeypatch.setattr(
        WhisperAdapter, "cuda_device_present", staticmethod(lambda: True)
    )
    monkeypatch.setattr(
        WhisperAdapter, "missing_cuda_libraries", staticmethod(lambda: ("cuBLAS",))
    )
    adapter = WhisperAdapter()

    assert adapter.cuda_available() is False
    # auto must fall back rather than pick a device that dies at inference
    assert adapter.select_device("auto") == "cpu"


def test_explicit_cuda_names_the_missing_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message must say what to install instead of denying the GPU."""
    monkeypatch.setattr(
        WhisperAdapter, "cuda_device_present", staticmethod(lambda: True)
    )
    monkeypatch.setattr(
        WhisperAdapter, "missing_cuda_libraries", staticmethod(lambda: ("cuBLAS",))
    )

    with pytest.raises(CudaUnavailableError) as caught:
        WhisperAdapter().select_device("cuda")

    assert "cuBLAS" in caught.value.message
    assert "--device cpu" in caught.value.message


def test_missing_cuda_library_is_not_reported_as_an_audio_fault() -> None:
    """The old message blamed the audio for a device problem."""
    with pytest.raises(CudaUnavailableError) as caught:
        WhisperAdapter._translate_error(
            RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"),
            loading=False,
        )

    assert "--device cpu" in caught.value.message


def test_preload_is_a_no_op_without_the_nvidia_wheels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine with no pip CUDA wheels must not error during detection."""
    from app.adapters import whisper_adapter

    monkeypatch.setattr(whisper_adapter, "_cuda_preload_done", False)

    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(whisper_adapter.importlib, "import_module", missing)
    whisper_adapter._preload_bundled_cuda_libraries()


def test_preload_loads_bundled_libraries_by_absolute_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """pip installs these where the loader cannot see them, so use full paths."""
    from app.adapters import whisper_adapter

    lib = tmp_path / "cublas" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcublas.so.12").write_bytes(b"")
    (lib / "libcublasLt.so.12").write_bytes(b"")

    class FakeNvidia:
        __path__ = [str(tmp_path)]

    loaded: list[str] = []
    monkeypatch.setattr(whisper_adapter, "_cuda_preload_done", False)
    monkeypatch.setattr(
        whisper_adapter.importlib, "import_module", lambda name: FakeNvidia
    )
    monkeypatch.setattr(
        whisper_adapter.ctypes, "CDLL", lambda path, mode=0: loaded.append(str(path))
    )
    whisper_adapter._preload_bundled_cuda_libraries()

    assert [Path(entry).name for entry in loaded] == [
        "libcublasLt.so.12",
        "libcublas.so.12",
    ]
    assert all(Path(entry).is_absolute() for entry in loaded)


def test_preload_runs_only_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.adapters import whisper_adapter

    calls: list[str] = []
    monkeypatch.setattr(whisper_adapter, "_cuda_preload_done", False)
    monkeypatch.setattr(
        whisper_adapter.importlib,
        "import_module",
        lambda name: calls.append(name) or _raise_import(name),
    )
    whisper_adapter._preload_bundled_cuda_libraries()
    whisper_adapter._preload_bundled_cuda_libraries()
    assert calls == ["nvidia"]


def _raise_import(name: str) -> object:
    raise ImportError(name)
