"""yt-dlp adapter for metadata-only YouTube inspection."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import yt_dlp
from yt_dlp.utils import DownloadError

from app.core.exceptions import (
    AudioDownloadError,
    AudioFormatUnavailableError,
    CaptionForgeError,
    MetadataRetrievalError,
    PrivateVideoError,
    SubtitleDownloadError,
    VideoUnavailableError,
)
from app.core.logging_config import get_logger
from app.models.subtitle import RawSubtitle, SubtitleSourceType, SubtitleTrack
from app.models.video import VideoMetadata
from app.utils.language_utils import language_name, normalize_language_code
from app.utils.url_utils import canonical_youtube_url


class Extractor(Protocol):
    """Minimal interface used from a yt-dlp extractor."""

    def extract_info(self, url: str, *, download: bool) -> Mapping[str, Any]:
        """Extract metadata for one URL."""


ExtractorFactory = Callable[[dict[str, Any]], AbstractContextManager[Extractor]]


class YtDlpInspection:
    """Clean adapter result before application-level selection."""

    def __init__(
        self,
        video: VideoMetadata,
        manual_tracks: tuple[SubtitleTrack, ...],
        automatic_tracks: tuple[SubtitleTrack, ...],
    ) -> None:
        self.video = video
        self.manual_tracks = manual_tracks
        self.automatic_tracks = automatic_tracks


class YtDlpAdapter:
    """Retrieve and map YouTube metadata without downloading content."""

    OPTIONS = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writethumbnail": False,
    }

    def __init__(self, extractor_factory: ExtractorFactory | None = None) -> None:
        self._extractor_factory = extractor_factory or yt_dlp.YoutubeDL

    def inspect(self, video_id: str, original_url: str) -> YtDlpInspection:
        """Retrieve and map metadata and caption availability for one video."""
        url = canonical_youtube_url(video_id)
        log = get_logger()
        log.info("Starting YouTube metadata retrieval for video_id={}", video_id)
        try:
            with self._extractor_factory(dict(self.OPTIONS)) as extractor:
                raw = extractor.extract_info(url, download=False)
        except DownloadError as exc:
            log.exception("yt-dlp metadata retrieval failed for video_id={}", video_id)
            raise self._translate_download_error(exc) from exc
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.exception("Metadata retrieval failed for video_id={}", video_id)
            raise MetadataRetrievalError(
                "YouTube metadata could not be retrieved. "
                "Check your internet connection and try again.",
                details=str(exc),
            ) from exc
        except Exception as exc:
            log.exception("Unexpected extractor failure for video_id={}", video_id)
            raise MetadataRetrievalError(
                "YouTube metadata could not be retrieved. "
                "Check your internet connection and try again.",
                details=str(exc),
            ) from exc

        if not isinstance(raw, Mapping):
            raise MetadataRetrievalError(
                "YouTube returned an invalid metadata response."
            )
        try:
            video = self._map_video(raw, original_url)
            manual = self._map_tracks(raw.get("subtitles"), SubtitleSourceType.MANUAL)
            automatic = self._map_tracks(
                raw.get("automatic_captions"), SubtitleSourceType.AUTOMATIC
            )
        except (ValueError, TypeError) as exc:
            log.exception("Unable to map metadata for video_id={}", video_id)
            raise MetadataRetrievalError(
                "YouTube returned metadata that CaptionForge could not interpret.",
                details=str(exc),
            ) from exc
        log.info(
            "Metadata retrieval succeeded for video_id={}; manual_tracks={}; "
            "automatic_tracks={}",
            video.video_id,
            len(manual),
            len(automatic),
        )
        return YtDlpInspection(video, manual, automatic)

    def download_subtitle(self, video_id: str, track: SubtitleTrack) -> RawSubtitle:
        """Download only the selected caption track into a temporary directory."""
        url = canonical_youtube_url(video_id)
        with TemporaryDirectory(prefix="captionforge-") as temporary:
            output_template = str(Path(temporary) / "%(id)s.%(ext)s")
            options = {
                **self.OPTIONS,
                "outtmpl": output_template,
                "writesubtitles": not track.is_automatic,
                "writeautomaticsub": track.is_automatic,
                "subtitleslangs": [track.language_code],
                "subtitlesformat": "vtt/best",
            }
            try:
                with self._extractor_factory(options) as extractor:
                    extractor.extract_info(url, download=True)
            except DownloadError as exc:
                raise SubtitleDownloadError(
                    "The selected YouTube caption track could not be downloaded.",
                    details=str(exc),
                ) from exc
            except Exception as exc:
                raise SubtitleDownloadError(
                    "The selected YouTube caption track could not be downloaded.",
                    details=str(exc),
                ) from exc

            candidates = sorted(
                path
                for path in Path(temporary).iterdir()
                if path.is_file()
                and path.suffix.lower().lstrip(".") in {"vtt", "srt", "json3"}
            )
            if not candidates:
                raise SubtitleDownloadError(
                    "YouTube did not provide data for the selected caption track."
                )
            path = candidates[0]
            try:
                return RawSubtitle(
                    content=path.read_text(encoding="utf-8-sig"),
                    format=path.suffix.lower().lstrip("."),
                )
            except (OSError, UnicodeError) as exc:
                raise SubtitleDownloadError(
                    "The downloaded caption track could not be read as UTF-8.",
                    details=str(exc),
                ) from exc

    def download_audio(
        self,
        video_id: str,
        destination: Path,
        progress_callback: Callable[[str, float | None], None] | None = None,
    ) -> Path:
        """Download the best audio stream only, never the video stream."""
        destination.mkdir(parents=True, exist_ok=True)
        template = str(destination / "source.%(ext)s")

        def hook(data: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes")
            percent = (
                float(downloaded) / float(total) * 100
                if total and downloaded is not None
                else None
            )
            progress_callback("Downloading audio", percent)

        options = {
            **self.OPTIONS,
            "skip_download": False,
            "format": "bestaudio",
            "outtmpl": template,
            "progress_hooks": [hook],
            "noplaylist": True,
        }
        try:
            with self._extractor_factory(options) as extractor:
                info = extractor.extract_info(
                    canonical_youtube_url(video_id), download=True
                )
        except DownloadError as exc:
            if "requested format is not available" in str(exc).lower():
                raise AudioFormatUnavailableError(
                    "No downloadable audio stream is available for this video."
                ) from exc
            raise AudioDownloadError(
                "The audio could not be downloaded. Please try again later.",
                details=str(exc),
            ) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise AudioDownloadError(
                "The audio download could not be completed.", details=str(exc)
            ) from exc
        except Exception as exc:
            raise AudioDownloadError(
                "The audio download could not be completed.", details=str(exc)
            ) from exc
        candidates = sorted(path for path in destination.iterdir() if path.is_file())
        if not candidates:
            requested = (
                info.get("requested_downloads") if isinstance(info, Mapping) else None
            )
            raise AudioFormatUnavailableError(
                "YouTube did not provide a usable audio file for this video.",
                details=str(requested) if requested else None,
            )
        return candidates[0]

    @staticmethod
    def _map_video(raw: Mapping[str, Any], original_url: str) -> VideoMetadata:
        video_id = str(raw.get("id") or "")
        webpage_url = str(raw.get("webpage_url") or canonical_youtube_url(video_id))
        duration = raw.get("duration")
        return VideoMetadata(
            video_id=video_id,
            title=str(raw.get("title") or "Untitled video"),
            channel_name=_optional_string(raw.get("channel") or raw.get("uploader")),
            channel_id=_optional_string(
                raw.get("channel_id") or raw.get("uploader_id")
            ),
            duration_seconds=int(duration) if duration is not None else None,
            webpage_url=webpage_url,
            original_url=original_url,
            thumbnail_url=_optional_string(raw.get("thumbnail")),
            upload_date=_parse_upload_date(raw.get("upload_date")),
            is_live=bool(raw.get("is_live", False)),
            live_status=_optional_string(raw.get("live_status")),
            availability=_optional_string(raw.get("availability")),
            age_limit=_optional_int(raw.get("age_limit")),
            description=_optional_string(raw.get("description")),
        )

    @staticmethod
    def _map_tracks(
        raw_tracks: Any, source: SubtitleSourceType
    ) -> tuple[SubtitleTrack, ...]:
        if not isinstance(raw_tracks, Mapping):
            return ()
        tracks: list[SubtitleTrack] = []
        for code, entries in raw_tracks.items():
            if not isinstance(code, str):
                continue
            normalized = normalize_language_code(code)
            if normalized is None:
                continue
            candidate_entries: Sequence[Any] = (
                entries
                if isinstance(entries, Sequence)
                and not isinstance(entries, (str, bytes))
                else ()
            )
            valid_entries = [
                entry for entry in candidate_entries if isinstance(entry, Mapping)
            ]
            formats = sorted(
                {
                    entry["ext"].lower()
                    for entry in valid_entries
                    if isinstance(entry.get("ext"), str) and entry["ext"].strip()
                }
            )
            tracks.append(
                SubtitleTrack(
                    language_code=code,
                    normalized_language_code=normalized,
                    language_name=language_name(normalized),
                    source_type=source,
                    is_automatic=source is SubtitleSourceType.AUTOMATIC,
                    available_formats=tuple(formats),
                    track_count=len(valid_entries),
                )
            )
        return tuple(sorted(tracks, key=lambda item: item.normalized_language_code))

    @staticmethod
    def _translate_download_error(exc: DownloadError) -> CaptionForgeError:
        message = str(exc).lower()
        if "private video" in message or "video is private" in message:
            return PrivateVideoError("This video is private and cannot be inspected.")
        if any(
            marker in message
            for marker in (
                "video unavailable",
                "removed",
                "deleted",
                "not available",
                "members-only",
                "age-restricted",
                "region",
            )
        ):
            return VideoUnavailableError(
                "The video could not be accessed. It may be private, removed, "
                "age-restricted, or region-restricted."
            )
        return MetadataRetrievalError(
            "YouTube metadata could not be retrieved. "
            "Check your internet connection and try again.",
            details=str(exc),
        )


def _parse_upload_date(value: Any) -> date | None:
    """Safely parse yt-dlp's compact upload date."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
