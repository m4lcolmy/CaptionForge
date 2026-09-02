"""Offline tests for the local web interface."""

import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Config
from app.core.exceptions import InvalidYouTubeUrlError, WhisperNotInstalledError
from app.interfaces.web import server as web_server
from app.interfaces.web.jobs import JobRequest
from app.models.subtitle import SubtitleDiscoveryResult
from app.models.video import VideoMetadata
from tests.conftest import VIDEO_URL, make_track

TOKEN = "test-token-value"
PORT = 8765
HEADERS = {"X-CaptionForge-Token": TOKEN, "Host": f"127.0.0.1:{PORT}"}


class StubWorkflowResult:
    """Minimal stand-in for TranscriptionWorkflowResult."""

    def __init__(self, paths: tuple[Path, ...], video: VideoMetadata) -> None:
        self.paths = paths
        self.video = video
        self.used_existing_captions = True
        self.transcription = None
        self.prepared_audio_path = None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """A test client wired to a server with a temporary output folder."""
    config = Config(default_output_folder=tmp_path / "output")
    application = web_server.create_app(
        config,
        token=TOKEN,
        port=PORT,
        preferences_file=tmp_path / "web-preferences.json",
    )
    with TestClient(application, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        yield test_client


def stub_inspection(
    monkeypatch: pytest.MonkeyPatch, video: VideoMetadata
) -> SubtitleDiscoveryResult:
    """Replace the inspection service graph with a fixed result."""
    track = make_track("ar")
    result = SubtitleDiscoveryResult(
        video=video,
        manual_tracks=(track,),
        selected_track=track,
        preferred_language="ar",
        selection_reason="exact manual match",
    )

    class StubVideoService:
        def inspect(self, url: str, language: str, **_: Any) -> SubtitleDiscoveryResult:
            return result

    monkeypatch.setattr(
        web_server, "create_video_service", lambda config: StubVideoService()
    )
    return result


def stub_workflow(monkeypatch: pytest.MonkeyPatch, process: Callable[..., Any]) -> None:
    """Replace the transcription service graph used by the job worker."""
    from app.interfaces.web import jobs as jobs_module

    class StubService:
        def process(self, url: str, **kwargs: Any) -> Any:
            return process(url, **kwargs)

    monkeypatch.setattr(
        jobs_module, "create_transcription_service", lambda config: StubService()
    )


def wait_for_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    """Poll a job until it reaches a terminal state."""
    for _ in range(200):
        response = client.get(f"/api/jobs/{job_id}", headers=HEADERS)
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
    raise AssertionError("job did not finish")


def test_health_reports_defaults(client: TestClient) -> None:
    response = client.get("/api/health", headers=HEADERS)
    assert response.status_code == 200
    payload = response.json()
    assert payload["app"] == "CaptionForge"
    assert "srt" in payload["supported_formats"]
    assert payload["default_device"] == "auto"
    assert isinstance(payload["cuda_available"], bool)


def test_device_radio_values_match_what_the_adapter_accepts() -> None:
    """The page must not offer a device the adapter would reject."""
    markup = (web_server.STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8")
    offered = set(re.findall(r'name="device" value="([a-z]+)"', markup))
    assert offered == {"auto", "cpu", "cuda"}
    for device in offered:
        Config(whisper_device=device)


def test_model_choices_are_rendered_from_a_single_list() -> None:
    """Model radios come from the script, including the escape hatch."""
    script = (web_server.STATIC_DIRECTORY / "app.js").read_text(encoding="utf-8")
    for size in ("tiny", "base", "small", "medium", "large-v3"):
        assert f'value: "{size}"' in script
    assert "Something else" in script


def test_api_requires_the_token(client: TestClient) -> None:
    response = client.get("/api/health", headers={"Host": f"127.0.0.1:{PORT}"})
    assert response.status_code == 401
    assert response.json()["code"] == "TokenRequired"


def test_rebound_hostnames_are_rejected(client: TestClient) -> None:
    response = client.get(
        "/api/health", headers={**HEADERS, "Host": "attacker.example.com"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "HostNotAllowed"


def test_page_sets_no_referrer(client: TestClient) -> None:
    response = client.get("/", headers={"Host": f"127.0.0.1:{PORT}"})
    assert response.status_code == 200
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "CaptionForge" in response.text


def test_inspect_returns_tracks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, video_metadata: VideoMetadata
) -> None:
    stub_inspection(monkeypatch, video_metadata)
    response = client.post(
        "/api/inspect", json={"url": VIDEO_URL, "language": "ar"}, headers=HEADERS
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["video"]["title"] == "Example video"
    assert payload["selected_track"]["language_code"] == "ar"


def test_inspect_maps_invalid_url_to_bad_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingService:
        def inspect(self, *args: Any, **kwargs: Any) -> Any:
            raise InvalidYouTubeUrlError("That is not a YouTube video link.")

    monkeypatch.setattr(
        web_server, "create_video_service", lambda config: FailingService()
    )
    response = client.post("/api/inspect", json={"url": "nope"}, headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error"] == "That is not a YouTube video link."


def test_unknown_format_is_rejected_before_work_starts(client: TestClient) -> None:
    response = client.post(
        "/api/jobs", json={"url": VIDEO_URL, "formats": ["mp3"]}, headers=HEADERS
    )
    assert response.status_code == 422


def test_job_runs_and_serves_its_files(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    video_metadata: VideoMetadata,
) -> None:
    produced = tmp_path / "Example video.srt"
    produced.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    stub_workflow(
        monkeypatch,
        lambda url, **kwargs: StubWorkflowResult((produced,), video_metadata),
    )
    created = client.post(
        "/api/jobs", json={"url": VIDEO_URL, "formats": ["srt"]}, headers=HEADERS
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    finished = wait_for_terminal(client, job_id)
    assert finished["status"] == "completed"
    assert finished["percent"] == 100.0
    assert finished["used_existing_captions"] is True
    assert [file["name"] for file in finished["files"]] == ["Example video.srt"]

    download = client.get(
        f"/api/jobs/{job_id}/files/Example video.srt", headers=HEADERS
    )
    assert download.status_code == 200
    assert "hello" in download.text


def test_job_reports_progress_from_the_workflow_callbacks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, video_metadata: VideoMetadata
) -> None:
    def process(url: str, **kwargs: Any) -> Any:
        kwargs["progress"]("Preparing audio", 10.0)
        kwargs["progress"]("Transcribing", 60.0)
        return StubWorkflowResult((), video_metadata)

    stub_workflow(monkeypatch, process)
    created = client.post("/api/jobs", json={"url": VIDEO_URL}, headers=HEADERS)
    job_id = created.json()["id"]
    finished = wait_for_terminal(client, job_id)
    assert finished["status"] == "completed"
    assert finished["stage"] == "Completed"


def test_failed_job_reports_the_short_message_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def process(url: str, **kwargs: Any) -> Any:
        raise WhisperNotInstalledError(
            "Install faster-whisper to transcribe locally.",
            details="ModuleNotFoundError: faster_whisper",
        )

    stub_workflow(monkeypatch, process)
    created = client.post("/api/jobs", json={"url": VIDEO_URL}, headers=HEADERS)
    finished = wait_for_terminal(client, created.json()["id"])
    assert finished["status"] == "failed"
    assert finished["error"] == "Install faster-whisper to transcribe locally."
    assert "ModuleNotFoundError" not in str(finished)


def test_cancel_stops_the_workflow(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.exceptions import TranscriptionCancelledError

    def process(url: str, **kwargs: Any) -> Any:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if kwargs["cancelled"]():
                raise TranscriptionCancelledError("Transcription was cancelled.")
            time.sleep(0.01)
        raise AssertionError("cancellation was never observed")

    stub_workflow(monkeypatch, process)
    created = client.post("/api/jobs", json={"url": VIDEO_URL}, headers=HEADERS)
    job_id = created.json()["id"]
    client.post(f"/api/jobs/{job_id}/cancel", headers=HEADERS)
    finished = wait_for_terminal(client, job_id)
    assert finished["status"] == "cancelled"


def test_download_rejects_paths_the_job_did_not_produce(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, video_metadata: VideoMetadata
) -> None:
    stub_workflow(
        monkeypatch, lambda url, **kwargs: StubWorkflowResult((), video_metadata)
    )
    created = client.post("/api/jobs", json={"url": VIDEO_URL}, headers=HEADERS)
    job_id = created.json()["id"]
    wait_for_terminal(client, job_id)
    response = client.get(
        f"/api/jobs/{job_id}/files/..%2F..%2Fetc%2Fpasswd", headers=HEADERS
    )
    assert response.status_code == 404


def test_unknown_job_is_reported_cleanly(client: TestClient) -> None:
    response = client.get("/api/jobs/does-not-exist", headers=HEADERS)
    assert response.status_code == 404
    assert response.json()["code"] == "JobNotFound"


def test_job_request_carries_every_option(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, video_metadata: VideoMetadata
) -> None:
    seen: dict[str, Any] = {}

    def process(url: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["url"] = url
        return StubWorkflowResult((), video_metadata)

    stub_workflow(monkeypatch, process)
    created = client.post(
        "/api/jobs",
        json={
            "url": VIDEO_URL,
            "language": "ar",
            "formats": ["srt", "docx"],
            "model": "medium",
            "device": "cpu",
            "prompt": "CaptionForge",
            "force": True,
            "timestamped_txt": True,
            "postprocess": False,
        },
        headers=HEADERS,
    )
    wait_for_terminal(client, created.json()["id"])
    assert seen["url"] == VIDEO_URL
    assert seen["language"] == "ar"
    assert seen["formats"] == ("srt", "docx")
    assert seen["model_name"] == "medium"
    assert seen["device"] == "cpu"
    assert seen["initial_prompt"] == "CaptionForge"
    assert seen["force"] is True
    assert seen["timestamped_txt"] is True
    assert seen["postprocess"] is False


def test_job_request_defaults_are_conservative() -> None:
    request = JobRequest(url=VIDEO_URL)
    assert request.postprocess is True
    assert request.force is False
    assert request.overwrite is False


def test_static_assets_ship_with_the_package(client: TestClient) -> None:
    for asset in ("app.css", "app.js"):
        assert (web_server.STATIC_DIRECTORY / asset).is_file()
        response = client.get(f"/static/{asset}", headers=HEADERS)
        assert response.status_code == 200
        assert response.content


def test_page_hides_toggled_sections_by_default() -> None:
    """Author display rules must not outrank the hidden attribute."""
    css = (web_server.STATIC_DIRECTORY / "app.css").read_text(encoding="utf-8")
    assert "[hidden] { display: none !important; }" in css


def test_page_loads_only_local_assets() -> None:
    """The interface must work with no internet connection."""
    markup = (web_server.STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8")
    assert "http://" not in markup
    assert "https://" not in markup


def test_health_separates_a_missing_gpu_from_missing_libraries(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detected card with unloadable libraries must not read as "no GPU"."""
    from app.adapters.whisper_adapter import WhisperAdapter

    monkeypatch.setattr(
        WhisperAdapter, "cuda_device_present", staticmethod(lambda: True)
    )
    monkeypatch.setattr(
        WhisperAdapter,
        "missing_cuda_libraries",
        staticmethod(lambda: ("cuBLAS", "cuDNN")),
    )
    payload = client.get("/api/health", headers=HEADERS).json()
    assert payload["cuda_available"] is False
    assert payload["cuda_device_present"] is True
    assert payload["cuda_missing_libraries"] == ["cuBLAS", "cuDNN"]


def test_health_reports_no_gpu_when_none_is_present(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.adapters.whisper_adapter import WhisperAdapter

    monkeypatch.setattr(
        WhisperAdapter, "cuda_device_present", staticmethod(lambda: False)
    )
    monkeypatch.setattr(WhisperAdapter, "missing_cuda_libraries", staticmethod(tuple))
    payload = client.get("/api/health", headers=HEADERS).json()
    assert payload["cuda_available"] is False
    assert payload["cuda_device_present"] is False
    assert payload["cuda_missing_libraries"] == []


def test_page_starts_with_one_format_checked(client: TestClient) -> None:
    """The page pre-selects a single format; the CLI keeps every configured one."""
    payload = client.get("/api/health", headers=HEADERS).json()
    assert payload["default_formats"] == ["srt", "vtt"]
    assert payload["preferences"]["formats"] == ["srt"]


def test_initial_format_follows_configured_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Someone who configures docx first sees docx checked, not srt."""
    config = Config(
        default_output_folder=tmp_path / "output",
        default_output_formats=("docx", "txt"),
    )
    application = web_server.create_app(
        config,
        token=TOKEN,
        port=PORT,
        preferences_file=tmp_path / "web-preferences.json",
    )
    with TestClient(application, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        payload = test_client.get("/api/health", headers=HEADERS).json()
    assert payload["preferences"]["formats"] == ["docx"]


def test_script_never_starts_with_an_empty_selection() -> None:
    """new Set(undefined) is silently empty, so the script must fall back."""
    script = (web_server.STATIC_DIRECTORY / "app.js").read_text(encoding="utf-8")
    assert "function initialFormats()" in script
    assert "new Set(initialFormats())" in script
    assert '["srt"]' in script
