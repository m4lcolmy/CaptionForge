"""Offline tests for whole-file MP4 and MP3 downloads."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest
from yt_dlp.utils import DownloadError

from app.adapters.ytdlp_adapter import YtDlpAdapter, map_media_options
from app.core.config import Config
from app.core.exceptions import (
    MediaDownloadCancelledError,
    MediaDownloadError,
    MediaFormatUnavailableError,
)
from app.models.media import MediaKind, MediaOptions, MediaVariant
from app.models.video import VideoMetadata
from app.services.media_service import MediaService
from app.services.video_service import VideoInspection
from tests.conftest import VIDEO_ID, VIDEO_URL

AUDIO = MediaVariant(key="audio", kind=MediaKind.AUDIO, label="MP3", extension="mp3")
VIDEO_720 = MediaVariant(
    key="720", kind=MediaKind.VIDEO, label="720p", extension="mp4", height=720
)


def formats_response(**extra: Any) -> dict[str, Any]:
    """A metadata response carrying a realistic spread of streams."""
    return {
        "id": VIDEO_ID,
        "title": "Example video",
        "duration": 600,
        "formats": [
            {"format_id": "251", "vcodec": "none", "acodec": "opus", "abr": 128},
            {
                "format_id": "140",
                "vcodec": "none",
                "acodec": "mp4a",
                "abr": 128,
                "filesize": 9_600_000,
            },
            {
                "format_id": "137",
                "vcodec": "avc1",
                "acodec": "none",
                "height": 1080,
                "ext": "mp4",
                "filesize": 120_000_000,
            },
            {
                "format_id": "136",
                "vcodec": "avc1",
                "acodec": "none",
                "height": 720,
                "ext": "mp4",
                "filesize": 60_000_000,
            },
            {
                "format_id": "18",
                "vcodec": "avc1",
                "acodec": "mp4a",
                "height": 360,
                "ext": "mp4",
                "filesize": 20_000_000,
            },
            # A height nobody should be offered, between the ladder's rungs.
            {
                "format_id": "160",
                "vcodec": "avc1",
                "acodec": "none",
                "height": 144,
                "ext": "mp4",
                "filesize": 3_000_000,
            },
        ],
        **extra,
    }


class RecordingExtractor(AbstractContextManager["RecordingExtractor"]):
    """An extractor that records its options and writes a file like yt-dlp would."""

    def __init__(
        self,
        options: dict[str, Any],
        *,
        error: Exception | None = None,
        produces: str = "media.mp4",
        hook_payloads: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self.options = options
        self.error = error
        self.produces = produces
        self.hook_payloads = hook_payloads

    def __enter__(self) -> "RecordingExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def extract_info(self, url: str, *, download: bool) -> Mapping[str, Any]:
        for payload in self.hook_payloads:
            for hook in self.options.get("progress_hooks", []):
                hook(payload)
        if self.error:
            raise self.error
        template = Path(self.options["outtmpl"])
        target = template.parent / self.produces
        target.write_bytes(b"x" * 1024)
        return {"id": VIDEO_ID}


def test_only_ladder_heights_are_offered_highest_first() -> None:
    options = map_media_options(formats_response())
    assert [item.key for item in options.variants] == ["1080", "720", "360", "audio"]
    assert options.variants[0].label == "1080p"
    assert options.audio_variant() is not None


def test_a_separate_video_stream_is_sized_with_its_audio() -> None:
    """A 1080p MP4 is two streams, and the person downloads both of them."""
    options = map_media_options(formats_response())
    by_key = {item.key: item for item in options.variants}
    assert by_key["1080"].estimated_bytes == 120_000_000 + 9_600_000
    # 360p is progressive: its own size already includes the sound.
    assert by_key["360"].estimated_bytes == 20_000_000


def test_an_mp3_is_sized_from_its_bitrate_not_from_the_source() -> None:
    options = map_media_options(formats_response())
    audio = options.audio_variant()
    assert audio is not None
    # 192 kbps over ten minutes, not the 128 kbps source stream.
    assert audio.estimated_bytes == int(192 * 1000 / 8 * 600)


def test_a_video_without_a_format_list_offers_nothing() -> None:
    assert map_media_options({"id": VIDEO_ID}).variants == ()


def test_sizes_fall_back_to_the_bitrate_when_no_filesize_is_published() -> None:
    raw = {
        "id": VIDEO_ID,
        "duration": 100,
        "formats": [
            {"vcodec": "avc1", "acodec": "none", "height": 720, "tbr": 1000},
            {"vcodec": "none", "acodec": "mp4a", "abr": 128},
        ],
    }
    options = map_media_options(raw)
    video = options.video_variants()[0]
    assert video.estimated_bytes == int(1000 * 1000 / 8 * 100) + int(
        128 * 1000 / 8 * 100
    )


def test_a_missing_duration_leaves_the_size_unknown_rather_than_wrong() -> None:
    raw = {
        "id": VIDEO_ID,
        "formats": [{"vcodec": "avc1", "acodec": "none", "height": 720, "tbr": 1000}],
    }
    assert map_media_options(raw).video_variants()[0].estimated_bytes is None


def test_resolve_steps_down_to_a_height_the_video_publishes() -> None:
    options = map_media_options(formats_response())
    assert options.resolve("1080").key == "1080"
    assert options.resolve("900").key == "720"
    assert options.resolve("2160").key == "1080"
    # Below everything on offer, the smallest file is the honest answer.
    assert options.resolve("240").key == "360"
    assert options.resolve("best").key == "1080"
    assert options.resolve("mp3").key == "audio"
    assert options.resolve("banana") is None


def test_the_format_selector_never_asks_for_more_than_the_chosen_height(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def factory(options: dict[str, Any]) -> Any:
        captured.update(options)
        return RecordingExtractor(options)

    YtDlpAdapter(factory).download_media(VIDEO_ID, tmp_path, VIDEO_720)
    assert "height<=720" in captured["format"]
    assert captured["merge_output_format"] == "mp4"
    assert captured["skip_download"] is False
    assert "postprocessors" not in captured


def test_an_audio_download_asks_ffmpeg_for_an_mp3(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def factory(options: dict[str, Any]) -> Any:
        captured.update(options)
        return RecordingExtractor(options, produces="media.mp3")

    path = YtDlpAdapter(factory).download_media(VIDEO_ID, tmp_path, AUDIO)
    assert path.name == "media.mp3"
    assert captured["postprocessors"][0]["preferredcodec"] == "mp3"
    assert "merge_output_format" not in captured


def test_a_configured_ffmpeg_reaches_yt_dlp(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def factory(options: dict[str, Any]) -> Any:
        captured.update(options)
        return RecordingExtractor(options)

    adapter = YtDlpAdapter(factory, ffmpeg_location="/opt/ffmpeg/bin/ffmpeg")
    adapter.download_media(VIDEO_ID, tmp_path, VIDEO_720)
    assert captured["ffmpeg_location"] == "/opt/ffmpeg/bin/ffmpeg"


def test_an_unavailable_format_is_not_reported_as_retryable(tmp_path: Path) -> None:
    error = DownloadError("ERROR: Requested format is not available")

    def factory(options: dict[str, Any]) -> Any:
        return RecordingExtractor(options, error=error)

    with pytest.raises(MediaFormatUnavailableError) as failure:
        YtDlpAdapter(factory).download_media(VIDEO_ID, tmp_path, VIDEO_720)
    assert failure.value.retryable is False


def test_a_transient_failure_is_reported_as_retryable(tmp_path: Path) -> None:
    def factory(options: dict[str, Any]) -> Any:
        return RecordingExtractor(options, error=DownloadError("ERROR: timed out"))

    with pytest.raises(MediaDownloadError) as failure:
        YtDlpAdapter(factory).download_media(VIDEO_ID, tmp_path, VIDEO_720)
    assert failure.value.retryable is True


def test_cancelling_stops_the_download_between_chunks(tmp_path: Path) -> None:
    """yt-dlp has no cancel handle, so the progress hook is the only way out."""

    def factory(options: dict[str, Any]) -> Any:
        return RecordingExtractor(
            options,
            hook_payloads=({"status": "downloading", "downloaded_bytes": 1},),
        )

    with pytest.raises(MediaDownloadCancelledError):
        YtDlpAdapter(factory).download_media(
            VIDEO_ID, tmp_path, VIDEO_720, cancelled=lambda: True
        )
    assert not (tmp_path / "media.mp4").exists()


def test_progress_is_reported_while_bytes_arrive(tmp_path: Path) -> None:
    seen: list[tuple[str, float | None]] = []

    def factory(options: dict[str, Any]) -> Any:
        return RecordingExtractor(
            options,
            hook_payloads=(
                {
                    "status": "downloading",
                    "downloaded_bytes": 50,
                    "total_bytes": 200,
                },
                {"status": "finished"},
            ),
        )

    YtDlpAdapter(factory).download_media(
        VIDEO_ID,
        tmp_path,
        VIDEO_720,
        progress_callback=lambda message, percent: seen.append((message, percent)),
    )
    assert seen == [("Downloading 720p", 25.0)]


class StubVideoService:
    """Answers inspection with fixed metadata and a fixed set of qualities."""

    def __init__(self, video: VideoMetadata, options: MediaOptions) -> None:
        self.video = video
        self.options = options

    def inspect_all(self, url: str, language: str, **_: Any) -> VideoInspection:
        from app.models.subtitle import SubtitleDiscoveryResult

        return VideoInspection(
            discovery=SubtitleDiscoveryResult(
                video=self.video, preferred_language=language
            ),
            media=self.options,
        )


class StubDownloader:
    """Writes a file into the workspace the way the real adapter would."""

    def __init__(self, name: str = "media.mp4", size: int = 2048) -> None:
        self.name = name
        self.size = size
        self.calls: list[MediaVariant] = []

    def download_media(
        self, video_id: str, destination: Path, variant: MediaVariant, **_: Any
    ) -> Path:
        self.calls.append(variant)
        path = destination / self.name
        path.write_bytes(b"x" * self.size)
        return path


def make_service(
    tmp_path: Path,
    video: VideoMetadata,
    downloader: StubDownloader | None = None,
    options: MediaOptions | None = None,
) -> tuple[MediaService, Config, StubDownloader]:
    """Build the service against temporary output and temporary workspace roots."""
    config = Config(
        default_output_folder=tmp_path / "output", temp_directory=tmp_path / "temp"
    )
    adapter = downloader or StubDownloader()
    resolved = options if options is not None else map_media_options(formats_response())
    service = MediaService(StubVideoService(video, resolved), adapter, config)
    return service, config, adapter


def test_a_download_is_named_after_the_video_and_its_quality(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, config, _ = make_service(tmp_path, video_metadata)
    result = service.download(VIDEO_URL, "720")
    assert result.path == config.default_output_folder / "Example video [720p].mp4"
    assert result.path.read_bytes() == b"x" * 2048


def test_an_mp3_is_named_without_a_quality_suffix(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    """There is only one audio quality, so a label would say nothing."""
    service, config, _ = make_service(
        tmp_path, video_metadata, StubDownloader(name="media.mp3")
    )
    result = service.download(VIDEO_URL, "mp3")
    assert result.path == config.default_output_folder / "Example video.mp3"


def test_a_second_download_never_replaces_the_first(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, config, _ = make_service(tmp_path, video_metadata)
    first = service.download(VIDEO_URL, "720")
    second = service.download(VIDEO_URL, "720")
    assert first.path.name == "Example video [720p].mp4"
    assert second.path.name == "Example video [720p] (2).mp4"
    assert len(list(config.default_output_folder.iterdir())) == 2


def test_overwrite_reuses_the_same_name(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, config, _ = make_service(tmp_path, video_metadata)
    first = service.download(VIDEO_URL, "720")
    second = service.download(VIDEO_URL, "720", overwrite=True)
    assert first.path == second.path
    assert len(list(config.default_output_folder.iterdir())) == 1


def test_the_extension_follows_what_ffmpeg_actually_produced(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    """A video with no MP4 pair comes back in the container yt-dlp had."""
    service, _, _ = make_service(
        tmp_path, video_metadata, StubDownloader(name="media.webm")
    )
    assert service.download(VIDEO_URL, "720").path.suffix == ".webm"


def test_a_quality_the_video_does_not_offer_is_refused_by_name(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, _, adapter = make_service(tmp_path, video_metadata)
    with pytest.raises(MediaFormatUnavailableError) as failure:
        service.download(VIDEO_URL, "banana")
    assert "1080" in failure.value.message
    assert adapter.calls == []


def test_the_workspace_is_cleaned_up_after_success_and_after_failure(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, config, _ = make_service(tmp_path, video_metadata)
    service.download(VIDEO_URL, "720")
    assert list(config.temp_directory.iterdir()) == []

    class FailingDownloader(StubDownloader):
        def download_media(self, *args: Any, **kwargs: Any) -> Path:
            raise MediaDownloadError("YouTube hung up.")

    failing, config, _ = make_service(tmp_path, video_metadata, FailingDownloader())
    with pytest.raises(MediaDownloadError):
        failing.download(VIDEO_URL, "720")
    assert list(config.temp_directory.iterdir()) == []


def test_a_cancellation_before_the_download_starts_writes_nothing(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, config, adapter = make_service(tmp_path, video_metadata)
    with pytest.raises(MediaDownloadCancelledError):
        service.download(VIDEO_URL, "720", cancelled=lambda: True)
    assert adapter.calls == []
    assert not config.default_output_folder.exists() or not list(
        config.default_output_folder.iterdir()
    )


def test_options_reports_what_the_inspection_found(
    tmp_path: Path, video_metadata: VideoMetadata
) -> None:
    service, _, _ = make_service(tmp_path, video_metadata)
    video, options = service.options(VIDEO_URL)
    assert video.video_id == VIDEO_ID
    assert [item.key for item in options.variants] == ["1080", "720", "360", "audio"]
