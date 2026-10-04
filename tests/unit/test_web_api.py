"""Offline tests for the local web interface."""

import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.adapters.ffmpeg_adapter import FFmpegAdapter, MediaProbe
from app.core.config import Config
from app.core.constants import (
    LOCAL_MEDIA_EXTENSIONS,
    MEDIA_AUDIO_KEY,
    MEDIA_VIDEO_HEIGHTS,
)
from app.core.exceptions import (
    InvalidYouTubeUrlError,
    MediaDownloadCancelledError,
    MediaFormatUnavailableError,
    TranscriptionCancelledError,
    WhisperNotInstalledError,
)
from app.interfaces.web import server as web_server
from app.interfaces.web.jobs import JobRequest
from app.interfaces.web.schemas import MediaJobRequestBody
from app.interfaces.web.uploads import STALE_AFTER_SECONDS, UPLOAD_FOLDER, safe_name
from app.models.media import MediaDownloadResult, MediaKind, MediaOptions, MediaVariant
from app.models.subtitle import SubtitleDiscoveryResult
from app.models.video import VideoMetadata
from app.services.video_service import VideoInspection
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
    """A test client wired to a server with temporary output and temp folders."""
    config = Config(
        default_output_folder=tmp_path / "output", temp_directory=tmp_path / "temp"
    )
    application = web_server.create_app(
        config,
        token=TOKEN,
        port=PORT,
        preferences_file=tmp_path / "web-preferences.json",
    )
    with TestClient(application, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        yield test_client


MEDIA = MediaOptions(
    variants=(
        MediaVariant(
            key="720",
            kind=MediaKind.VIDEO,
            label="720p",
            extension="mp4",
            height=720,
            estimated_bytes=50_000_000,
        ),
        MediaVariant(
            key="audio",
            kind=MediaKind.AUDIO,
            label="MP3",
            extension="mp3",
            estimated_bytes=5_000_000,
        ),
    )
)


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

        def inspect_all(self, url: str, language: str, **_: Any) -> VideoInspection:
            return VideoInspection(discovery=result, media=MEDIA)

    monkeypatch.setattr(
        web_server, "create_video_service", lambda config: StubVideoService()
    )
    return result


def stub_media_service(
    monkeypatch: pytest.MonkeyPatch, download: Callable[..., Any]
) -> None:
    """Replace the download service graph used by the media worker."""
    from app.interfaces.web import jobs as jobs_module

    class StubService:
        def download(self, url: str, quality: str, **kwargs: Any) -> Any:
            return download(url, quality, **kwargs)

    monkeypatch.setattr(
        jobs_module, "create_media_service", lambda config: StubService()
    )


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
        def inspect_all(self, *args: Any, **kwargs: Any) -> Any:
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


def test_inspect_carries_the_download_qualities(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, video_metadata: VideoMetadata
) -> None:
    """One lookup answers both questions the page asks."""
    stub_inspection(monkeypatch, video_metadata)
    response = client.post("/api/inspect", json={"url": VIDEO_URL}, headers=HEADERS)
    assert response.status_code == 200
    variants = response.json()["media"]["variants"]
    assert [item["key"] for item in variants] == ["720", "audio"]
    assert variants[1]["label"] == "MP3"


def test_media_job_downloads_and_serves_the_file(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    video_metadata: VideoMetadata,
) -> None:
    produced = tmp_path / "Example video [720p].mp4"
    produced.write_bytes(b"not really an mp4")
    variant = MEDIA.variants[0]
    stub_media_service(
        monkeypatch,
        lambda url, quality, **kwargs: MediaDownloadResult(
            path=produced, variant=variant, video=video_metadata
        ),
    )
    created = client.post(
        "/api/media", json={"url": VIDEO_URL, "quality": "720"}, headers=HEADERS
    )
    assert created.status_code == 202
    assert created.json()["kind"] == "media"

    finished = wait_for_terminal(client, created.json()["id"])
    assert finished["status"] == "completed"
    assert finished["media"]["label"] == "720p"
    assert [file["name"] for file in finished["files"]] == ["Example video [720p].mp4"]

    download = client.get(
        f"/api/jobs/{created.json()['id']}/files/Example video [720p].mp4",
        headers=HEADERS,
    )
    assert download.status_code == 200
    assert download.content == b"not really an mp4"


def test_media_job_reports_the_short_message_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(url: str, quality: str, **kwargs: Any) -> Any:
        raise MediaFormatUnavailableError(
            "This video does not publish a 4320p stream.",
            details="yt_dlp.utils.DownloadError: requested format is not available",
        )

    stub_media_service(monkeypatch, download)
    created = client.post(
        "/api/media", json={"url": VIDEO_URL, "quality": "4320"}, headers=HEADERS
    )
    finished = wait_for_terminal(client, created.json()["id"])
    assert finished["status"] == "failed"
    assert finished["error"] == "This video does not publish a 4320p stream."
    assert "DownloadError" not in str(finished)


def test_media_job_can_be_cancelled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(url: str, quality: str, **kwargs: Any) -> Any:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if kwargs["cancelled"]():
                raise MediaDownloadCancelledError("The download was cancelled.")
            time.sleep(0.01)
        raise AssertionError("cancellation was never observed")

    stub_media_service(monkeypatch, download)
    created = client.post(
        "/api/media", json={"url": VIDEO_URL, "quality": "audio"}, headers=HEADERS
    )
    job_id = created.json()["id"]
    client.post(f"/api/jobs/{job_id}/cancel", headers=HEADERS)
    assert wait_for_terminal(client, job_id)["status"] == "cancelled"


def test_a_download_does_not_queue_behind_a_transcription(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, video_metadata: VideoMetadata
) -> None:
    """The two lanes are separate, so one click stays responsive."""
    running = threading.Event()

    def process(url: str, **kwargs: Any) -> Any:
        running.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not kwargs["cancelled"]():
            time.sleep(0.01)
        raise TranscriptionCancelledError("Transcription was cancelled.")

    produced = Path(str(client.app.state.config.default_output_folder))
    produced.mkdir(parents=True, exist_ok=True)
    file = produced / "Example video.mp3"
    file.write_bytes(b"audio")
    stub_workflow(monkeypatch, process)
    stub_media_service(
        monkeypatch,
        lambda url, quality, **kwargs: MediaDownloadResult(
            path=file, variant=MEDIA.variants[1], video=video_metadata
        ),
    )
    captions = client.post("/api/jobs", json={"url": VIDEO_URL}, headers=HEADERS)
    assert running.wait(timeout=5.0)

    media = client.post(
        "/api/media", json={"url": VIDEO_URL, "quality": "audio"}, headers=HEADERS
    )
    finished = wait_for_terminal(client, media.json()["id"])
    assert finished["status"] == "completed"
    client.post(f"/api/jobs/{captions.json()['id']}/cancel", headers=HEADERS)


def test_the_page_offers_no_quality_the_api_cannot_resolve() -> None:
    """Every key the script can send is a key the request model accepts."""
    script = (web_server.STATIC_DIRECTORY / "app.js").read_text(encoding="utf-8")
    assert "quality: variant.key" in script
    for height in MEDIA_VIDEO_HEIGHTS:
        MediaJobRequestBody(url=VIDEO_URL, quality=str(height))
    MediaJobRequestBody(url=VIDEO_URL, quality=MEDIA_AUDIO_KEY)


# ---------- files from this computer ----------


def upload(client: TestClient, name: str, content: bytes = b"sound") -> dict[str, Any]:
    """Hand the server a file the way the page's XMLHttpRequest does."""
    response = client.put(
        "/api/uploads",
        content=content,
        headers={**HEADERS, "X-CaptionForge-Filename": quote(name)},
    )
    assert response.status_code == 201, response.text
    payload: dict[str, Any] = response.json()
    return payload


def stub_probe(monkeypatch: pytest.MonkeyPatch, *, has_audio: bool = True) -> None:
    """Let FFprobe report a 90-second file without running it."""
    monkeypatch.setattr(
        FFmpegAdapter,
        "probe",
        lambda self, source: MediaProbe(duration_seconds=90.0, has_audio=has_audio),
    )


def test_health_lists_what_the_file_chooser_accepts(client: TestClient) -> None:
    payload = client.get("/api/health", headers=HEADERS).json()
    assert payload["local_media_extensions"] == list(LOCAL_MEDIA_EXTENSIONS)


def test_a_chosen_file_is_copied_and_then_looked_up_by_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub_probe(monkeypatch)

    stored = upload(client, "محاضرة 3.MP4", b"x" * 4096)

    path = Path(stored["path"])
    assert path.is_absolute()
    assert path.is_relative_to(tmp_path / "temp" / UPLOAD_FOLDER)
    assert path.name == "محاضرة 3.mp4"
    assert path.read_bytes() == b"x" * 4096
    assert stored["size_bytes"] == 4096

    looked_up = client.post(
        "/api/inspect", json={"url": stored["path"]}, headers=HEADERS
    )
    assert looked_up.status_code == 200
    payload = looked_up.json()
    assert payload["video"]["title"] == "محاضرة 3"
    assert payload["video"]["local_path"] == stored["path"]
    assert payload["video"]["duration_seconds"] == 90
    assert payload["selected_track"] is None
    assert payload["media"]["variants"] == []


@pytest.mark.parametrize(
    ("sent", "kept"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\fakepath\\talk.MKV", "talk.mkv"),
        ("CON.mp4", "upload.mp4"),
        ("notes.tar.gz", "notes.tar.gz"),
        ("", "upload"),
    ],
)
def test_an_uploaded_name_stays_inside_its_folder(sent: str, kept: str) -> None:
    assert safe_name(sent) == kept


def test_an_empty_file_is_refused_and_leaves_nothing(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.put(
        "/api/uploads",
        content=b"",
        headers={**HEADERS, "X-CaptionForge-Filename": "talk.mp3"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "That file is empty."
    assert not list((tmp_path / "temp" / UPLOAD_FOLDER).iterdir())


def test_a_new_file_replaces_the_previous_copy(client: TestClient) -> None:
    first = Path(upload(client, "one.mp3")["path"])
    second = Path(upload(client, "two.mp3")["path"])
    assert not first.parent.exists()
    assert second.is_file()


def test_a_copy_an_unfinished_job_still_reads_is_kept(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    running = threading.Event()

    def process(url: str, **kwargs: Any) -> Any:
        running.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not kwargs["cancelled"]():
            time.sleep(0.01)
        raise TranscriptionCancelledError("Transcription was cancelled.")

    stub_workflow(monkeypatch, process)
    first = Path(upload(client, "one.mp3")["path"])
    job = client.post("/api/jobs", json={"url": str(first)}, headers=HEADERS)
    assert running.wait(timeout=5.0)

    upload(client, "two.mp3")

    assert first.is_file()
    client.post(f"/api/jobs/{job.json()['id']}/cancel", headers=HEADERS)
    wait_for_terminal(client, job.json()["id"])


def test_copies_go_when_the_server_stops(tmp_path: Path) -> None:
    config = Config(
        default_output_folder=tmp_path / "output", temp_directory=tmp_path / "temp"
    )
    application = web_server.create_app(
        config, token=TOKEN, port=PORT, preferences_file=tmp_path / "prefs.json"
    )
    with TestClient(application, base_url=f"http://127.0.0.1:{PORT}") as test_client:
        stored = Path(upload(test_client, "talk.mp3")["path"])
        assert stored.is_file()
    assert not stored.parent.exists()


def test_copies_a_crashed_session_left_are_swept_at_start(tmp_path: Path) -> None:
    uploads = tmp_path / "temp" / UPLOAD_FOLDER
    stale, fresh = uploads / "stale", uploads / "fresh"
    for folder in (stale, fresh):
        folder.mkdir(parents=True)
        (folder / "talk.mp3").write_bytes(b"x")
    old = time.time() - STALE_AFTER_SECONDS - 60
    os.utime(stale, (old, old))
    config = Config(
        default_output_folder=tmp_path / "output", temp_directory=tmp_path / "temp"
    )
    application = web_server.create_app(
        config, token=TOKEN, port=PORT, preferences_file=tmp_path / "prefs.json"
    )
    with TestClient(application, base_url=f"http://127.0.0.1:{PORT}"):
        pass
    # Another server may still be using a recent copy.
    assert not stale.exists() and fresh.exists()


def test_a_file_without_sound_is_a_bad_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_probe(monkeypatch, has_audio=False)
    stored = upload(client, "silent.mp4")
    response = client.post(
        "/api/inspect", json={"url": stored["path"]}, headers=HEADERS
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": "That file has no sound to transcribe.",
        "code": "NoAudioStreamError",
        "retryable": False,
        "update_may_help": False,
    }


def test_a_missing_path_is_a_bad_request(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/inspect", json={"url": str(tmp_path / "gone.mp4")}, headers=HEADERS
    )
    assert response.status_code == 400
    assert response.json()["code"] == "LocalFileNotFoundError"


def test_uploads_need_the_token(client: TestClient) -> None:
    response = client.put(
        "/api/uploads",
        content=b"x",
        headers={"Host": f"127.0.0.1:{PORT}", "X-CaptionForge-Filename": "a.mp3"},
    )
    assert response.status_code == 401


def test_the_field_takes_a_path_and_the_page_takes_a_file() -> None:
    """A url-typed field would refuse a path before the script saw it."""
    markup = (web_server.STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8")
    field = re.search(r'<input\s+id="url"[^>]*>', markup)
    assert field is not None
    assert 'type="text"' in field.group(0)
    assert 'id="choose-btn"' in markup
    assert re.search(r'<input type="file" id="file-input"[^>]*hidden>', markup)
