"""Shared construction of the CaptionForge service graph."""

from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.adapters.whisper_adapter import WhisperAdapter
from app.adapters.ytdlp_adapter import YtDlpAdapter
from app.core.config import Config
from app.services.audio_service import AudioService
from app.services.export_service import ExportService
from app.services.subtitle_service import SubtitleService
from app.services.transcription_service import TranscriptionService
from app.services.video_service import VideoService


def create_video_service(config: Config | None = None) -> VideoService:
    """Construct the inspection service graph."""
    return VideoService(YtDlpAdapter(), SubtitleService(config), config)


def create_transcription_service(config: Config) -> TranscriptionService:
    """Construct the caption-first transcription workflow graph."""
    adapter = YtDlpAdapter()
    subtitles = SubtitleService(config)
    video_service = VideoService(adapter, subtitles, config)
    return TranscriptionService(
        video_service,
        adapter,
        subtitles,
        AudioService(
            video_service,
            adapter,
            FFmpegAdapter(config.ffmpeg_executable),
            config,
        ),
        WhisperAdapter(),
        ExportService(),
        config,
    )
