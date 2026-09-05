"""yt-dlp adapter for metadata-only YouTube inspection."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

import yt_dlp
from yt_dlp.utils import DownloadError

from app.core.constants import (
    MEDIA_AUDIO_BITRATE_KBPS,
    MEDIA_AUDIO_EXTENSION,
    MEDIA_AUDIO_KEY,
    MEDIA_VIDEO_EXTENSION,
    MEDIA_VIDEO_HEIGHTS,
)
from app.core.exceptions import (
    AudioDownloadError,
    AudioFormatUnavailableError,
    AudioStreamForbiddenError,
    CaptionForgeError,
    MediaDownloadCancelledError,
    MediaDownloadError,
    MediaFormatUnavailableError,
    MetadataRetrievalError,
    PrivateVideoError,
    SubtitleDownloadError,
    SubtitleStreamForbiddenError,
    VideoUnavailableError,
)
from app.core.logging_config import get_logger
from app.models.media import MediaKind, MediaOptions, MediaVariant
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
        media: MediaOptions | None = None,
    ) -> None:
        self.video = video
        self.manual_tracks = manual_tracks
        self.automatic_tracks = automatic_tracks
        # The same metadata response already lists every stream, so the
        # downloadable qualities cost nothing extra to carry along.
        self.media = media or MediaOptions()


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

    def __init__(
        self,
        extractor_factory: ExtractorFactory | None = None,
        ffmpeg_location: str | None = None,
    ) -> None:
        self._extractor_factory = extractor_factory or yt_dlp.YoutubeDL
        # Merging an MP4 and encoding an MP3 both shell out to FFmpeg, so a
        # configured executable has to reach yt-dlp as well as our own adapter.
        self._ffmpeg_location = ffmpeg_location

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
            media = map_media_options(raw)
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
        return YtDlpInspection(video, manual, automatic, media)

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
                # json3 first: YouTube's VTT auto-captions are rolling/karaoke
                # cues that repeat each line, which json3 does not do.
                "subtitlesformat": "json3/vtt/best",
            }
            try:
                with self._extractor_factory(options) as extractor:
                    extractor.extract_info(url, download=True)
            except DownloadError as exc:
                if _is_extractor_stale(str(exc).lower()):
                    raise SubtitleStreamForbiddenError(
                        "YouTube refused the caption track. This normally means "
                        "the installed yt-dlp is too old for YouTube's current "
                        "site: upgrade it with 'pip install --upgrade yt-dlp'.",
                        details=str(exc),
                    ) from exc
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
            message = str(exc).lower()
            if "requested format is not available" in message:
                raise AudioFormatUnavailableError(
                    "No downloadable audio stream is available for this video."
                ) from exc
            if _is_extractor_stale(message):
                raise AudioStreamForbiddenError(
                    "YouTube refused the audio stream. This normally means the "
                    "installed yt-dlp is too old for YouTube's current site: "
                    "upgrade it with 'pip install --upgrade yt-dlp' and retry.",
                    details=str(exc),
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

    def download_media(
        self,
        video_id: str,
        destination: Path,
        variant: MediaVariant,
        progress_callback: Callable[[str, float | None], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Path:
        """Download one whole file: a merged MP4, or the audio encoded as MP3."""
        destination.mkdir(parents=True, exist_ok=True)
        stop = cancelled or (lambda: False)

        def hook(data: dict[str, Any]) -> None:
            if stop():
                # yt-dlp has no cancel handle; raising out of the hook is how a
                # download in flight is stopped between chunks.
                raise MediaDownloadCancelledError("The download was cancelled.")
            if progress_callback is None or data.get("status") != "downloading":
                return
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            done = data.get("downloaded_bytes")
            percent = (
                float(done) / float(total) * 100 if total and done is not None else None
            )
            progress_callback(f"Downloading {variant.label}", percent)

        announced = False

        def after(data: dict[str, Any]) -> None:
            nonlocal announced
            if stop():
                raise MediaDownloadCancelledError("The download was cancelled.")
            if progress_callback is None or data.get("status") != "started":
                return
            # Merging an MP4 and encoding an MP3 both take real time on a long
            # video, and both happen after the byte counter has reached 100%.
            # Several postprocessors can run; the person only needs telling once.
            if not announced:
                announced = True
                progress_callback("Preparing the file", None)

        options: dict[str, Any] = {
            **self.OPTIONS,
            "skip_download": False,
            "format": _format_selector(variant),
            "outtmpl": str(destination / "media.%(ext)s"),
            "progress_hooks": [hook],
            "postprocessor_hooks": [after],
            "noplaylist": True,
            "overwrites": True,
            # CaptionForge reports its own progress; yt-dlp's bar would print
            # over the CLI's lines and into the server's terminal for nothing.
            "noprogress": True,
        }
        if variant.is_audio:
            options["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": MEDIA_AUDIO_EXTENSION,
                    "preferredquality": str(MEDIA_AUDIO_BITRATE_KBPS),
                }
            ]
        else:
            options["merge_output_format"] = MEDIA_VIDEO_EXTENSION
        if self._ffmpeg_location:
            options["ffmpeg_location"] = self._ffmpeg_location
        try:
            with self._extractor_factory(options) as extractor:
                extractor.extract_info(canonical_youtube_url(video_id), download=True)
        except MediaDownloadCancelledError:
            raise
        except DownloadError as exc:
            message = str(exc).lower()
            if stop():
                raise MediaDownloadCancelledError(
                    "The download was cancelled."
                ) from exc
            if "requested format is not available" in message:
                raise MediaFormatUnavailableError(
                    f"This video does not publish a {variant.label} stream. "
                    "Look the video up again to refresh what it offers."
                ) from exc
            if _is_extractor_stale(message):
                raise MediaFormatUnavailableError(
                    "YouTube refused the media stream. This normally means the "
                    "installed yt-dlp is too old for YouTube's current site: "
                    "upgrade it with 'pip install --upgrade yt-dlp' and retry.",
                    details=str(exc),
                ) from exc
            raise MediaDownloadError(
                "The download could not be completed. Please try again later.",
                details=str(exc),
            ) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise MediaDownloadError(
                "The download could not be completed.", details=str(exc)
            ) from exc
        except Exception as exc:
            raise MediaDownloadError(
                "The download could not be completed.", details=str(exc)
            ) from exc
        return _produced_file(destination, variant)

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
            # YouTube offers machine translations of the ASR track for ~150
            # languages; they carry tlang= in the caption URL. Their quality is
            # far below both a real caption and local transcription.
            urls = [
                entry["url"]
                for entry in valid_entries
                if isinstance(entry.get("url"), str) and entry["url"].strip()
            ]
            tracks.append(
                SubtitleTrack(
                    language_code=code,
                    normalized_language_code=normalized,
                    language_name=language_name(normalized),
                    source_type=source,
                    is_automatic=source is SubtitleSourceType.AUTOMATIC,
                    is_translated=any("tlang=" in url for url in urls),
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


def _is_extractor_stale(message: str) -> bool:
    """Recognize failures that an updated extractor, not a retry, resolves."""
    return any(
        marker in message
        for marker in (
            "403",
            "forbidden",
            "needs to be reloaded",
            "confirm you're not a bot",
            "please sign in",
            "nsig extraction failed",
            "unable to extract",
        )
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


def _format_selector(variant: MediaVariant) -> str:
    """Ask yt-dlp for the best streams that satisfy one offered variant."""
    if variant.is_audio:
        return "bestaudio/best"
    height = variant.height
    if height is None:
        return (
            f"bestvideo[ext={MEDIA_VIDEO_EXTENSION}]+bestaudio/"
            "bestvideo+bestaudio/best"
        )
    # H.264 first, deliberately. YouTube also publishes VP9 inside an MP4 at
    # most heights, and yt-dlp rates it higher, but a .mp4 that QuickTime and
    # ordinary video editors refuse to open is not what "download as MP4"
    # promises. After that: any MP4 pair, any pair, then a progressive stream.
    return (
        f"bestvideo[height<={height}][vcodec^=avc1]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={height}][ext={MEDIA_VIDEO_EXTENSION}]+bestaudio/"
        f"bestvideo[height<={height}]+bestaudio/"
        f"best[height<={height}]/best"
    )


def _produced_file(destination: Path, variant: MediaVariant) -> Path:
    """Identify the finished file, ignoring anything left beside it."""
    files = [path for path in destination.iterdir() if path.is_file()]
    if not files:
        raise MediaFormatUnavailableError(
            "YouTube did not provide a usable file for this video."
        )
    wanted = f".{variant.extension.lower()}"
    matching = [path for path in files if path.suffix.lower() == wanted]
    # A merge or an MP3 encode can leave its source behind when FFmpeg is
    # unavailable; the largest file is still the one worth keeping.
    return max(matching or files, key=lambda path: path.stat().st_size)


def map_media_options(raw: Mapping[str, Any]) -> MediaOptions:
    """Turn yt-dlp's format list into the qualities a person can be offered."""
    formats = raw.get("formats")
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        return MediaOptions()
    entries = [entry for entry in formats if isinstance(entry, Mapping)]
    duration = _optional_int(raw.get("duration"))
    audio_only = [
        entry
        for entry in entries
        if entry.get("vcodec") in {None, "none"}
        and entry.get("acodec") not in {None, "none"}
    ]
    best_audio = max(audio_only, key=_ranking(duration), default=None)
    audio_bytes = _stream_bytes(best_audio, duration) if best_audio else None

    variants: list[MediaVariant] = []
    for height in MEDIA_VIDEO_HEIGHTS:
        candidates = [
            entry
            for entry in entries
            if entry.get("vcodec") not in {None, "none"}
            and _optional_int(entry.get("height")) == height
        ]
        if not candidates:
            continue
        best = max(candidates, key=_ranking(duration))
        video_bytes = _stream_bytes(best, duration)
        carries_audio = best.get("acodec") not in {None, "none"}
        estimated = _total_bytes(video_bytes, None if carries_audio else audio_bytes)
        variants.append(
            MediaVariant(
                key=str(height),
                kind=MediaKind.VIDEO,
                label=f"{height}p",
                extension=MEDIA_VIDEO_EXTENSION,
                height=height,
                estimated_bytes=estimated,
            )
        )
    if best_audio is not None:
        variants.append(
            MediaVariant(
                key=MEDIA_AUDIO_KEY,
                kind=MediaKind.AUDIO,
                label="MP3",
                extension=MEDIA_AUDIO_EXTENSION,
                # An MP3 is re-encoded at a fixed bitrate, so the source
                # stream's size says nothing useful about the result.
                estimated_bytes=_bitrate_bytes(MEDIA_AUDIO_BITRATE_KBPS, duration),
            )
        )
    return MediaOptions(variants=tuple(variants))


def _ranking(
    duration: int | None,
) -> Callable[[Mapping[str, Any]], tuple[int, float]]:
    """Rank the streams at one height the way the size estimate should read.

    A published ``filesize`` comes first, ahead of a higher bitrate without
    one. YouTube advertises inflated ``tbr`` values on the adaptive entries
    that carry no size, and estimating from those overstated a 19 MB download
    as 30 MB. The formats that publish a size are also the ones yt-dlp
    normally settles on, so trusting them keeps the chip honest.
    """

    def rank(entry: Mapping[str, Any]) -> tuple[int, float]:
        published = any(entry.get(key) for key in ("filesize", "filesize_approx"))
        return (1 if published else 0, float(_stream_bytes(entry, duration) or 0))

    return rank


def _stream_bytes(entry: Mapping[str, Any] | None, duration: int | None) -> int | None:
    """Best available size for one stream, estimated from its bitrate if need be."""
    if entry is None:
        return None
    for key in ("filesize", "filesize_approx"):
        value = _optional_int(entry.get(key))
        if value:
            return value
    rate = entry.get("tbr") or entry.get("vbr") or entry.get("abr")
    try:
        return _bitrate_bytes(float(rate), duration) if rate else None
    except (TypeError, ValueError):
        return None


def _bitrate_bytes(kilobits_per_second: float, duration: int | None) -> int | None:
    """Convert a bitrate and a duration into an approximate byte count."""
    if not duration or kilobits_per_second <= 0:
        return None
    return int(kilobits_per_second * 1000 / 8 * duration)


def _total_bytes(*parts: int | None) -> int | None:
    """Sum the sizes that are known, or report nothing when none of them are."""
    known = [part for part in parts if part]
    return sum(known) if known else None
