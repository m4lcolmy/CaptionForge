"""Offline coverage for Deepgram: the adapter, the key, and the workflow branch.

Nothing here reaches Deepgram. The adapter is pointed at stand-in openers, or
at a real HTTP server on 127.0.0.1 when the streamed upload itself is tested.
"""

from __future__ import annotations

import io
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from app.adapters.deepgram_adapter import (
    DeepgramAdapter,
    check_and_save_key,
    keyterms_from,
    parse_response,
    supports_keyterms,
)
from app.adapters.ffmpeg_adapter import FFmpegAdapter
from app.core.config import Config
from app.core.deepgram_key import (
    DeepgramKey,
    describe_key,
    forget_key,
    load_key,
    normalize_key,
    save_key,
)
from app.core.exceptions import (
    DeepgramCreditError,
    DeepgramKeyMissingError,
    DeepgramKeyRejectedError,
    DeepgramRequestError,
    DeepgramTimeoutError,
    DeepgramUnavailableError,
    EmptyTranscriptionError,
    TranscriptionCancelledError,
    ValidationError,
)
from app.models.subtitle import SubtitleDiscoveryResult
from app.models.transcription import TranscriptionResult
from app.services.export_service import ExportService
from app.services.subtitle_service import SubtitleService
from app.services.transcription_service import TranscriptionService
from tests.conftest import VIDEO_URL, make_track
from tests.unit.test_phase5_whisper import (
    WorkflowAudio,
    WorkflowVideoService,
    WorkflowWhisper,
    WorkflowYtDlp,
)

KEY = "0123456789abcdef0123456789abcdef01234567"


def reply(
    words: list[tuple[str, float, float]],
    *,
    utterances: bool = True,
    language: str | None = None,
) -> dict[str, Any]:
    """A trimmed Deepgram answer holding these punctuated words."""
    raw = [
        {
            "word": text.strip(".,").lower(),
            "punctuated_word": text,
            "start": start,
            "end": end,
            "confidence": 0.9,
        }
        for text, start, end in words
    ]
    channel: dict[str, Any] = {
        "alternatives": [{"transcript": " ".join(w[0] for w in words), "words": raw}]
    }
    if language:
        channel["detected_language"] = language
        channel["language_confidence"] = 0.97
    results: dict[str, Any] = {"channels": [channel]}
    if utterances and raw:
        results["utterances"] = [
            {
                "start": raw[0]["start"],
                "end": raw[-1]["end"],
                "confidence": 0.93,
                "transcript": " ".join(w[0] for w in words),
                "words": raw,
            }
        ]
    return {"metadata": {"duration": 12.5}, "results": results}


class Answer:
    """What a stand-in opener hands back: a body to read, in a with block."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> Answer:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return self._body


def opener_returning(
    payload: dict[str, Any], seen: list[Any] | None = None, pause: float = 0.0
) -> Callable[[Any, float], Answer]:
    """An opener that reads the whole upload, as http.client would, then answers."""

    def open_(request: Any, _timeout: float) -> Answer:
        body = b""
        if request.data is not None:
            while block := request.data.read(8192):
                body += block
                time.sleep(pause)
        if seen is not None:
            seen.append((request, body))
        return Answer(json.dumps(payload).encode())

    return open_


def opener_failing(code: int, body: dict[str, Any] | None = None) -> Any:
    def open_(request: Any, _timeout: float) -> Answer:
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "error",
            {},  # type: ignore[arg-type]
            io.BytesIO(json.dumps(body or {}).encode()),
        )

    return open_


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "prepared.ogg"
    path.write_bytes(b"O" * 50_000)
    return path


# ---------- request building ----------


def test_keyterms_come_from_the_names_field() -> None:
    assert keyterms_from("Ahmad Ali, Kayseri،  Kayseri , ,Nova") == (
        "Ahmad Ali",
        "Kayseri",
        "Nova",
    )
    assert keyterms_from(None) == ()
    # Deepgram refuses the whole request past its token cap, so stop short.
    many = ", ".join(f"word{index}" for index in range(500))
    assert len(keyterms_from(many)) == 300


def test_query_names_the_language_or_asks_for_detection() -> None:
    adapter = DeepgramAdapter()
    named = dict(adapter.build_query(model="nova-3", language="ar"))
    assert named["language"] == "ar" and "detect_language" not in named
    assert named["smart_format"] == named["utterances"] == "true"
    detected = dict(adapter.build_query(model="nova-3", language=None))
    assert detected["detect_language"] == "true" and "language" not in detected


def test_terms_use_the_parameter_the_model_understands() -> None:
    adapter = DeepgramAdapter()
    nova3 = adapter.build_query(model="nova-3", language="tr", keyterms=("A", "B"))
    assert [pair for pair in nova3 if pair[0] == "keyterm"] == [
        ("keyterm", "A"),
        ("keyterm", "B"),
    ]
    nova2 = adapter.build_query(model="nova-2", language="tr", keyterms=("A",))
    assert ("keywords", "A") in nova2
    assert supports_keyterms("nova-3-medical") and not supports_keyterms("nova-2")


# ---------- reading the answer ----------


def test_utterances_become_segments_with_word_timings() -> None:
    result = parse_response(
        reply([("Merhaba,", 0.5, 0.9), ("dünya.", 1.0, 1.6)], language="tr"),
        model="nova-3",
        language=None,
    )
    assert result.engine == "deepgram"
    assert result.model_name == "nova-3"
    assert result.detected_language == "tr"
    assert result.language_probability == pytest.approx(0.97)
    assert result.duration_seconds == 12.5
    (segment,) = result.segments
    # Built from the punctuated words, so the reflow can cut at their timings.
    assert segment.text == "Merhaba, dünya."
    assert len(segment.words) == len(segment.text.split())
    assert (segment.start_seconds, segment.end_seconds) == (0.5, 1.6)


def test_a_reply_without_utterances_is_split_at_pauses() -> None:
    words = [("One", 0.0, 0.4), ("two.", 0.5, 0.9), ("Three", 3.0, 3.4)]
    result = parse_response(
        reply(words, utterances=False), model="nova-3", language="en"
    )
    assert [segment.text for segment in result.segments] == ["One two.", "Three"]
    assert result.detected_language == "en"


def test_a_reply_without_speech_says_so() -> None:
    with pytest.raises(EmptyTranscriptionError):
        parse_response({"results": {"channels": [{}]}}, model="nova-3", language="ar")


# ---------- the upload ----------


def test_the_audio_is_streamed_with_the_key_and_type(audio: Path) -> None:
    seen: list[Any] = []
    progress: list[tuple[str, float | None]] = []
    adapter = DeepgramAdapter(
        opener=opener_returning(reply([("Hi.", 0.0, 0.5)]), seen),
        base_url="https://deepgram.test/v1",
    )

    result = adapter.transcribe(
        audio,
        api_key=KEY,
        model="nova-3",
        language="ar",
        keyterms=("Ahmad",),
        progress=lambda message, percent: progress.append((message, percent)),
    )

    request, body = seen[0]
    assert body == audio.read_bytes()
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Token {KEY}"
    assert request.get_header("Content-type") == "audio/ogg"
    assert request.get_header("Content-length") == str(audio.stat().st_size)
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
    assert query["model"] == ["nova-3"] and query["keyterm"] == ["Ahmad"]
    assert result.segments[0].text == "Hi."
    uploads = [
        percent
        for message, percent in progress
        if "Uploading" in message and percent is not None
    ]
    assert uploads == sorted(uploads) and uploads[-1] == pytest.approx(75.0)


class _Recorder(BaseHTTPRequestHandler):
    received: list[tuple[str, dict[str, str], bytes]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server's own name
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.received.append((self.path, dict(self.headers), body))
        payload = json.dumps(reply([("Hello.", 0.0, 0.4)])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: object) -> None:
        return None


@pytest.fixture
def local_server() -> Iterator[str]:
    _Recorder.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_a_real_http_upload_arrives_whole(local_server: str, tmp_path: Path) -> None:
    """urllib really does stream the file-like body, with its length."""
    source = tmp_path / "prepared.ogg"
    source.write_bytes(os.urandom(300_000))
    result = DeepgramAdapter(base_url=local_server).transcribe(
        source, api_key=KEY, model="nova-3", language="en"
    )
    path, headers, body = _Recorder.received[0]
    assert path.startswith("/v1/listen?")
    assert body == source.read_bytes()
    assert headers["Authorization"] == f"Token {KEY}"
    assert result.segments[0].text == "Hello."


def test_cancel_stops_the_upload(audio: Path) -> None:
    calls = {"count": 0}

    def cancelled() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    adapter = DeepgramAdapter(opener=opener_returning(reply([]), pause=0.2))
    started = time.monotonic()
    with pytest.raises(TranscriptionCancelledError):
        adapter.transcribe(audio, api_key=KEY, model="nova-3", cancelled=cancelled)
    # Seven 8 KB blocks at 0.2 s each would take 1.4 s; cancel answers sooner.
    assert time.monotonic() - started < 1.2


def test_cancel_does_not_wait_for_deepgram_to_answer(audio: Path) -> None:
    def slow(request: Any, _timeout: float) -> Answer:
        while request.data.read(8192):
            pass
        time.sleep(3)
        return Answer(b"{}")

    done = threading.Event()
    adapter = DeepgramAdapter(opener=slow)
    started = time.monotonic()
    threading.Timer(0.3, done.set).start()
    with pytest.raises(TranscriptionCancelledError):
        adapter.transcribe(audio, api_key=KEY, model="nova-3", cancelled=done.is_set)
    assert time.monotonic() - started < 1.5


# ---------- what goes wrong ----------


@pytest.mark.parametrize(
    ("code", "error", "retryable"),
    [
        (401, DeepgramKeyRejectedError, False),
        (403, DeepgramKeyRejectedError, False),
        (402, DeepgramCreditError, False),
        (400, DeepgramRequestError, False),
        (504, DeepgramTimeoutError, False),
        (503, DeepgramUnavailableError, True),
        (429, DeepgramUnavailableError, True),
    ],
)
def test_each_refusal_says_what_to_do(
    audio: Path, code: int, error: type[Exception], retryable: bool
) -> None:
    adapter = DeepgramAdapter(opener=opener_failing(code))
    with pytest.raises(error) as raised:
        adapter.transcribe(audio, api_key=KEY, model="nova-3")
    assert raised.value.retryable is retryable  # type: ignore[attr-defined]


def test_a_bad_request_carries_deepgrams_reason(audio: Path) -> None:
    adapter = DeepgramAdapter(
        opener=opener_failing(
            400, {"err_msg": "No such model/language/tier combination found."}
        )
    )
    with pytest.raises(DeepgramRequestError) as raised:
        adapter.transcribe(audio, api_key=KEY, model="nova-2", language="ar")
    assert "No such model/language" in raised.value.message


def test_no_connection_is_retryable(audio: Path) -> None:
    def offline(_request: Any, _timeout: float) -> Answer:
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    with pytest.raises(DeepgramUnavailableError) as raised:
        DeepgramAdapter(opener=offline).transcribe(audio, api_key=KEY, model="nova-3")
    assert raised.value.retryable
    assert "could not be reached" in raised.value.message


# ---------- the key ----------


@pytest.mark.parametrize(
    ("opener", "verdict"),
    [
        (opener_returning({"projects": []}), True),
        (opener_failing(401), False),
        # Too narrow to list projects is still a real key.
        (opener_failing(403), True),
        (opener_failing(503), None),
    ],
)
def test_a_key_is_checked_with_deepgram(opener: Any, verdict: bool | None) -> None:
    assert DeepgramAdapter(opener=opener).verify_key(KEY) is verdict


def test_an_unreachable_deepgram_cannot_judge_a_key() -> None:
    def offline(_request: Any, _timeout: float) -> Answer:
        raise urllib.error.URLError("no route")

    assert DeepgramAdapter(opener=offline).verify_key(KEY) is None


def test_a_saved_key_is_readable_only_by_its_owner(tmp_path: Path) -> None:
    path = tmp_path / "captionforge" / "deepgram.key"
    saved = save_key(f"  Token {KEY}\n", path)
    assert saved == DeepgramKey(KEY, "file")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_key(path, environ={}, env_file=tmp_path / "none") == saved


def test_the_environment_wins_over_the_saved_file(tmp_path: Path) -> None:
    path = tmp_path / "deepgram.key"
    save_key(KEY, path)
    other = "f" * 40
    found = load_key(path, environ={"DEEPGRAM_API_KEY": other}, env_file=tmp_path)
    assert found == DeepgramKey(other, "environment")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"CAPTIONFORGE_DEEPGRAM_API_KEY={other}\n", encoding="utf-8")
    assert load_key(path, environ={}, env_file=dotenv) == found


def test_only_the_last_four_characters_ever_show(tmp_path: Path) -> None:
    key = DeepgramKey(KEY, "file")
    assert KEY not in repr(key) and KEY not in str(describe_key(key))
    assert describe_key(key) == {"saved": True, "source": "file", "hint": "…4567"}
    assert describe_key(None) == {"saved": False, "source": None, "hint": None}


def test_something_that_is_not_a_key_is_refused(tmp_path: Path) -> None:
    for wrong in ("", "short", "has a space in the middle of it really", "ключ" * 10):
        with pytest.raises(ValidationError):
            normalize_key(wrong)
    path = tmp_path / "deepgram.key"
    path.write_text("not a key\n", encoding="utf-8")
    assert load_key(path, environ={}, env_file=tmp_path / "none") is None


def test_forgetting_removes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "deepgram.key"
    save_key(KEY, path)
    assert forget_key(path) is True
    assert not path.exists()
    assert forget_key(path) is False


def test_a_refused_key_is_never_saved(tmp_path: Path) -> None:
    path = tmp_path / "deepgram.key"
    with pytest.raises(DeepgramKeyRejectedError):
        check_and_save_key(KEY, DeepgramAdapter(opener=opener_failing(401)), path)
    assert not path.exists()


def test_offline_the_key_is_saved_unchecked(tmp_path: Path) -> None:
    path = tmp_path / "deepgram.key"
    saved, verdict = check_and_save_key(
        KEY, DeepgramAdapter(opener=opener_failing(503)), path
    )
    assert verdict is None and saved.value == KEY and path.is_file()


# ---------- the workflow ----------


class WorkflowDeepgram:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Path, **kwargs: Any) -> TranscriptionResult:
        self.calls.append({"audio": audio, **kwargs})
        kwargs["progress"]("Uploading audio to Deepgram", 50.0)
        return parse_response(
            reply([("Merhaba.", 0.0, 0.8)], language="tr"),
            model=kwargs["model"],
            language=kwargs["language"],
        )


class RecordingAudio(WorkflowAudio):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.options: dict[str, Any] = {}

    def prepare(self, *args: Any, **kwargs: Any) -> Path:
        self.options = kwargs
        return super().prepare(*args, **kwargs)


def deepgram_workflow(
    tmp_path: Path,
    video_metadata: Any,
    *,
    key: DeepgramKey | None,
    captions: bool = False,
) -> tuple[TranscriptionService, RecordingAudio, WorkflowWhisper, WorkflowDeepgram]:
    discovery = SubtitleDiscoveryResult(
        video=video_metadata,
        selected_track=make_track("ar") if captions else None,
        preferred_language="ar",
    )
    audio = RecordingAudio(tmp_path)
    whisper = WorkflowWhisper()
    deepgram = WorkflowDeepgram()
    service = TranscriptionService(
        WorkflowVideoService(discovery),  # type: ignore[arg-type]
        WorkflowYtDlp(),  # type: ignore[arg-type]
        SubtitleService(),
        audio,  # type: ignore[arg-type]
        whisper,  # type: ignore[arg-type]
        ExportService(),
        Config(default_output_folder=tmp_path),
        deepgram=deepgram,  # type: ignore[arg-type]
        deepgram_key=lambda: key,
    )
    return service, audio, whisper, deepgram


def test_deepgram_is_sent_compressed_audio_and_the_names(
    tmp_path: Path, video_metadata: Any
) -> None:
    service, audio, whisper, deepgram = deepgram_workflow(
        tmp_path, video_metadata, key=DeepgramKey(KEY, "file")
    )
    result = service.process(
        VIDEO_URL,
        engine="deepgram",
        language="tr",
        initial_prompt="Kayseri, Ahmad",
        formats=("txt",),
    )
    assert whisper.calls == 0
    (call,) = deepgram.calls
    assert call["api_key"] == KEY
    assert call["model"] == "nova-3"
    assert call["language"] == "tr"
    assert call["keyterms"] == ("Kayseri", "Ahmad")
    assert audio.options["audio_format"] == "ogg"
    assert audio.options["bitrate_kbps"] == 48
    assert result.transcription is not None
    assert result.transcription.engine == "deepgram"
    assert "Merhaba." in result.paths[0].read_text(encoding="utf-8")


def test_a_missing_key_fails_before_any_audio_is_fetched(
    tmp_path: Path, video_metadata: Any
) -> None:
    service, audio, _, deepgram = deepgram_workflow(tmp_path, video_metadata, key=None)
    with pytest.raises(DeepgramKeyMissingError):
        service.process(VIDEO_URL, engine="deepgram", formats=("txt",))
    assert audio.calls == 0 and deepgram.calls == []


def test_captions_still_come_first_without_any_key(
    tmp_path: Path, video_metadata: Any
) -> None:
    service, audio, _, deepgram = deepgram_workflow(
        tmp_path, video_metadata, key=None, captions=True
    )
    result = service.process(VIDEO_URL, engine="deepgram", formats=("txt",))
    assert result.used_existing_captions
    assert audio.calls == 0 and deepgram.calls == []


def test_whisper_stays_the_default_and_keeps_wav(
    tmp_path: Path, video_metadata: Any
) -> None:
    service, audio, whisper, deepgram = deepgram_workflow(
        tmp_path, video_metadata, key=DeepgramKey(KEY, "file")
    )
    service.process(VIDEO_URL, formats=("txt",))
    assert whisper.calls == 1 and deepgram.calls == []
    assert audio.options["audio_format"] is None


def test_an_unknown_engine_is_refused(tmp_path: Path, video_metadata: Any) -> None:
    service, *_ = deepgram_workflow(tmp_path, video_metadata, key=None)
    with pytest.raises(ValidationError):
        service.process(VIDEO_URL, engine="siri")


def test_the_engine_setting_is_validated() -> None:
    assert Config(transcription_engine=" Deepgram ").transcription_engine == "deepgram"
    with pytest.raises(ValueError):
        Config(transcription_engine="siri")


def test_upload_audio_is_encoded_as_opus() -> None:
    command = FFmpegAdapter().build_conversion_command(
        Path("in.webm"), Path("out.ogg"), audio_format="ogg", bitrate_kbps=48
    )
    assert command[command.index("-acodec") + 1] == "libopus"
    assert command[command.index("-b:a") + 1] == "48k"
    # The size the interfaces show holds only if the rate is held to.
    assert command[command.index("-vbr") + 1] == "constrained"
    wav = FFmpegAdapter().build_conversion_command(Path("in.webm"), Path("out.wav"))
    assert wav[wav.index("-acodec") + 1] == "pcm_s16le" and "-b:a" not in wav
