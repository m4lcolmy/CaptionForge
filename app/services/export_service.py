"""Coordinate safe multi-format subtitle exports."""

from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from app.core.constants import SUPPORTED_OUTPUT_FORMATS
from app.core.exceptions import ExportError
from app.exporters.json_exporter import render_json
from app.exporters.srt_exporter import render_srt
from app.exporters.txt_exporter import render_txt
from app.exporters.vtt_exporter import render_vtt
from app.models.subtitle import SubtitleSegment, SubtitleTrack
from app.models.video import VideoMetadata
from app.utils.file_utils import (
    atomic_write_text,
    ensure_disk_space,
    ensure_output_directory,
    sanitize_filename,
)


class ExportService:
    """Validate destinations and write requested subtitle formats as UTF-8."""

    def export(
        self,
        video: VideoMetadata,
        track: SubtitleTrack,
        segments: Sequence[SubtitleSegment],
        formats: Sequence[str],
        output_directory: Path,
        *,
        timestamped_txt: bool = False,
        overwrite: bool = False,
    ) -> tuple[Path, ...]:
        """Write all requested formats, refusing accidental overwrites."""
        normalized = tuple(dict.fromkeys(item.lower() for item in formats))
        unknown = set(normalized) - SUPPORTED_OUTPUT_FORMATS
        if not normalized:
            raise ExportError("At least one output format must be selected.")
        if unknown:
            raise ExportError(
                f"Unsupported output format(s): {', '.join(sorted(unknown))}"
            )
        directory = ensure_output_directory(output_directory)
        stem = sanitize_filename(video.title, fallback=video.video_id)
        paths = tuple(directory / f"{stem}.{extension}" for extension in normalized)
        existing = [path for path in paths if path.exists()]
        if existing and not overwrite:
            raise ExportError(
                f"Output file already exists: {existing[0]}. "
                "Use --overwrite to replace it."
            )
        rendered = {
            "srt": lambda: render_srt(segments),
            "vtt": lambda: render_vtt(segments),
            "txt": lambda: render_txt(segments, timestamped=timestamped_txt),
            "json": lambda: render_json(video, track, segments),
        }
        contents = tuple(rendered[extension]() for extension in normalized)
        required_bytes = sum(len(item.encode("utf-8")) for item in contents)
        ensure_disk_space(directory, required_bytes)
        created: list[Path] = []
        try:
            for path, content in zip(paths, contents, strict=True):
                was_present = path.exists()
                atomic_write_text(path, content, overwrite=overwrite)
                if not was_present:
                    created.append(path)
        except ExportError:
            for path in created:
                with suppress(OSError):
                    path.unlink(missing_ok=True)
            raise
        return paths
