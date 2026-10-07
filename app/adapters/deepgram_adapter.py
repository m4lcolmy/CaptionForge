"""Deepgram's pre-recorded transcription API, reached with the standard library.

This is the only part of CaptionForge that sends a recording off this
computer, and it does so only when someone picks Deepgram. It needs no SDK:
one streamed POST carries the audio, and the reply is converted into the same
:class:`TranscriptionResult` Whisper produces, so post-processing and export
cannot tell the engines apart.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from app.core.constants import DEEPGRAM_API_URL
from app.core.deepgram_key import DeepgramKey, normalize_key, save_key
from app.core.exceptions import (
    DeepgramCreditError,
    DeepgramError,
    DeepgramKeyRejectedError,
    DeepgramRequestError,
    DeepgramTimeoutError,
    DeepgramUnavailableError,
    EmptyTranscriptionError,
    TranscriptionCancelledError,
)
from app.models.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)

ProgressCallback = Callable[[str, float | None], None]
CancelCallback = Callable[[], bool]
Opener = Callable[[urllib.request.Request, float], Any]

# Deepgram answers once the whole recording is transcribed and stops trying
# after ten minutes, so the read timeout has to outlast that.
RESPONSE_TIMEOUT_SECONDS = 660.0
VERIFY_TIMEOUT_SECONDS = 15.0
# How often the waiting thread looks at the cancel flag.
CANCEL_POLL_SECONDS = 0.25
# Key terms are capped at 500 tokens a request, and Deepgram rejects the whole
# request beyond that, so stay well under it.
MAXIMUM_KEYTERM_WORDS = 300
# When a reply carries no utterances, words are grouped at pauses this long.
FALLBACK_PAUSE_SECONDS = 0.8
FALLBACK_MAXIMUM_SECONDS = 12.0

_CONTENT_TYPES = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}

# Upload progress spans this stretch of the transcription's share of the bar.
_UPLOAD_START = 40.0
_UPLOAD_END = 75.0
_WAITING = 78.0
_PARSED = 85.0


def supports_keyterms(model: str) -> bool:
    """Nova-3 and Flux take ``keyterm``; earlier models take ``keywords``."""
    lowered = model.lower()
    return lowered.startswith(("nova-3", "flux"))


def keyterms_from(prompt: str | None) -> tuple[str, ...]:
    """Split "Names and spellings" at commas into terms Deepgram can take."""
    if not prompt:
        return ()
    terms: list[str] = []
    words = 0
    for raw in prompt.replace("،", ",").split(","):
        term = " ".join(raw.split())
        if not term or term in terms:
            continue
        words += len(term.split())
        if words > MAXIMUM_KEYTERM_WORDS:
            break
        terms.append(term)
    return tuple(terms)


class _CancellableUpload:
    """A file read in blocks by http.client, reporting and stopping as it goes."""

    def __init__(
        self,
        handle: BinaryIO,
        total: int,
        stop: threading.Event,
        report: Callable[[float], None],
    ) -> None:
        self._handle = handle
        self._total = max(1, total)
        self._sent = 0
        self._stop = stop
        self._report = report

    def read(self, size: int = -1) -> bytes:
        if self._stop.is_set():
            # Raised inside http.client's send loop; not an OSError, so
            # urllib passes it straight up instead of wrapping it.
            raise TranscriptionCancelledError("Transcription was cancelled.")
        block = self._handle.read(size)
        self._sent += len(block)
        self._report(min(1.0, self._sent / self._total))
        return block


class DeepgramAdapter:
    """Send one prepared recording to Deepgram and read the transcript back."""

    def __init__(
        self,
        *,
        opener: Opener | None = None,
        base_url: str = DEEPGRAM_API_URL,
    ) -> None:
        self._opener = opener or (
            lambda request, timeout: urllib.request.urlopen(request, timeout=timeout)
        )
        self._base_url = base_url.rstrip("/")

    # ---------- the key ----------

    def verify_key(self, key: str) -> bool | None:
        """True when Deepgram accepts the key, False when it refuses, None offline.

        A key scoped too narrowly to list projects still answers 403 rather
        than 401, and may well transcribe, so only 401 counts as a refusal.
        """
        request = urllib.request.Request(
            f"{self._base_url}/projects",
            headers=_headers(key),
            method="GET",
        )
        try:
            with self._opener(request, VERIFY_TIMEOUT_SECONDS) as reply:
                reply.read(1)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return False
            if exc.code == 403:
                return True
            return None
        except (urllib.error.URLError, OSError):
            return None

    # ---------- transcription ----------

    def build_query(
        self,
        *,
        model: str,
        language: str | None,
        keyterms: Sequence[str] = (),
    ) -> list[tuple[str, str]]:
        """The query string, as ordered pairs because terms repeat a name."""
        query = [
            ("model", model),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("utterances", "true"),
        ]
        if language:
            query.append(("language", language))
        else:
            # Detection only knows about 35 languages, so a named language
            # is always sent as such rather than left to guesswork.
            query.append(("detect_language", "true"))
        name = "keyterm" if supports_keyterms(model) else "keywords"
        query.extend((name, term) for term in keyterms)
        return query

    def transcribe(
        self,
        audio_path: Path,
        *,
        api_key: str,
        model: str,
        language: str | None = None,
        keyterms: Sequence[str] = (),
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> TranscriptionResult:
        """Upload the recording, wait for Deepgram, and convert its answer."""
        notify = progress or (lambda _message, _percent: None)
        is_cancelled = cancelled or (lambda: False)
        if is_cancelled():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        query = urllib.parse.urlencode(
            self.build_query(model=model, language=language, keyterms=keyterms)
        )
        size = audio_path.stat().st_size
        stop = threading.Event()
        outcome: dict[str, Any] = {}
        last_percent = [-1]

        def report(fraction: float) -> None:
            # http.client reads 8 KB at a time; one update per percent is plenty.
            percent = int(fraction * 100)
            if percent == last_percent[0]:
                return
            last_percent[0] = percent
            notify(
                "Uploading audio to Deepgram",
                _UPLOAD_START + fraction * (_UPLOAD_END - _UPLOAD_START),
            )

        def send() -> None:
            try:
                with audio_path.open("rb") as handle:
                    request = urllib.request.Request(
                        f"{self._base_url}/listen?{query}",
                        data=_CancellableUpload(handle, size, stop, report),
                        headers={
                            **_headers(api_key),
                            "Content-Type": _content_type(audio_path),
                            "Content-Length": str(size),
                        },
                        method="POST",
                    )
                    with self._opener(request, RESPONSE_TIMEOUT_SECONDS) as reply:
                        outcome["body"] = reply.read()
            except BaseException as exc:  # noqa: BLE001 - handed to the caller
                outcome["error"] = exc

        notify("Uploading audio to Deepgram", _UPLOAD_START)
        worker = threading.Thread(
            target=send, name="captionforge-deepgram", daemon=True
        )
        worker.start()
        waiting_reported = False
        while worker.is_alive():
            worker.join(CANCEL_POLL_SECONDS)
            if is_cancelled():
                # The upload stops at its next block. Once the audio is all
                # there Deepgram finishes regardless; its answer is dropped.
                stop.set()
                raise TranscriptionCancelledError("Transcription was cancelled.")
            if not waiting_reported and last_percent[0] >= 100:
                waiting_reported = True
                notify("Deepgram is transcribing", _WAITING)

        error = outcome.get("error")
        if error is not None:
            raise _translate(error) from error
        payload = _decode(outcome.get("body", b""))
        result = parse_response(payload, model=model, language=language)
        notify("Reading the transcript", _PARSED)
        return result


def check_and_save_key(
    value: str,
    adapter: DeepgramAdapter | None = None,
    path: Path | None = None,
) -> tuple[DeepgramKey, bool | None]:
    """Tidy a pasted key, ask Deepgram about it, and save it unless refused.

    Returns the saved key and Deepgram's verdict. Offline the verdict is None
    and the key is saved anyway: the person may well be about to reconnect,
    and the first transcription checks it again.
    """
    key = normalize_key(value)
    verdict = (adapter or DeepgramAdapter()).verify_key(key)
    if verdict is False:
        raise DeepgramKeyRejectedError(
            "Deepgram did not accept that key. Nothing was saved."
        )
    return save_key(key, path), verdict


def parse_response(
    payload: dict[str, Any], *, model: str, language: str | None
) -> TranscriptionResult:
    """Turn Deepgram's JSON into segments with word timings."""
    results = payload.get("results") or {}
    channels = results.get("channels") or [{}]
    channel = channels[0] if channels else {}
    detected = str(channel.get("detected_language") or language or "unknown").lower()
    probability = _probability(channel.get("language_confidence"))
    utterances = results.get("utterances") or []
    if utterances:
        pieces = [
            (
                _words(item.get("words") or ()),
                str(item.get("transcript") or ""),
                item.get("start"),
                item.get("end"),
                item.get("confidence"),
            )
            for item in utterances
        ]
    else:
        alternatives = channel.get("alternatives") or [{}]
        pieces = [
            (group, "", None, None, None)
            for group in _group_at_pauses(_words(alternatives[0].get("words") or ()))
        ]
    segments: list[TranscriptionSegment] = []
    for words, transcript, start, end, confidence in pieces:
        # Built from the words when there are any, so the word count matches
        # the text and the reflow can cut lines at the real word timings.
        text = " ".join(word.text for word in words) if words else transcript.strip()
        if not text:
            continue
        begin = _seconds(start, words[0].start_seconds if words else None)
        finish = _seconds(end, words[-1].end_seconds if words else None)
        if begin is None or finish is None:
            continue
        segments.append(
            TranscriptionSegment(
                index=len(segments) + 1,
                start_seconds=max(0.0, begin),
                end_seconds=max(finish, begin + 0.001),
                text=text,
                language=detected,
                confidence=_probability(confidence),
                words=words,
            )
        )
    if not segments:
        raise EmptyTranscriptionError("Deepgram found no speech in the audio.")
    metadata = payload.get("metadata") or {}
    duration = metadata.get("duration")
    return TranscriptionResult(
        segments=tuple(segments),
        detected_language=detected,
        language_probability=probability,
        duration_seconds=float(duration) if isinstance(duration, int | float) else None,
        model_name=model,
        device="deepgram",
        compute_type="cloud",
        engine="deepgram",
    )


def _words(raw: Sequence[dict[str, Any]]) -> tuple[WordTiming, ...]:
    converted: list[WordTiming] = []
    for item in raw:
        text = str(item.get("punctuated_word") or item.get("word") or "").strip()
        if not text:
            continue
        try:
            start = max(0.0, float(item["start"]))
            end = max(start, float(item["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        converted.append(
            WordTiming(
                start_seconds=start,
                end_seconds=end,
                text=text,
                probability=_probability(item.get("confidence")),
            )
        )
    return tuple(converted)


def _group_at_pauses(words: tuple[WordTiming, ...]) -> list[tuple[WordTiming, ...]]:
    """Split one long word list where the speaker pauses, or every 12 seconds."""
    groups: list[list[WordTiming]] = []
    for word in words:
        current = groups[-1] if groups else None
        if (
            current is None
            or word.start_seconds - current[-1].end_seconds >= FALLBACK_PAUSE_SECONDS
            or word.end_seconds - current[0].start_seconds > FALLBACK_MAXIMUM_SECONDS
        ):
            groups.append([word])
        else:
            current.append(word)
    return [tuple(group) for group in groups]


def _seconds(value: Any, fallback: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Token {key}", "Accept": "application/json"}


def _content_type(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def _decode(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepgramUnavailableError(
            "Deepgram sent back an answer CaptionForge could not read.",
            details=str(exc),
        ) from exc
    if not isinstance(payload, dict):
        raise DeepgramUnavailableError(
            "Deepgram sent back an answer CaptionForge could not read."
        )
    return payload


def _reason(error: urllib.error.HTTPError) -> str | None:
    """Deepgram's own explanation, from ``err_msg`` in its JSON error body."""
    try:
        body = json.loads(error.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable body just has no reason
        return None
    if not isinstance(body, dict):
        return None
    reason = body.get("err_msg") or body.get("message") or body.get("reason")
    return str(reason)[:200] if reason else None


def _translate(error: BaseException) -> BaseException:
    """Map what went wrong to an error a person can act on."""
    if isinstance(error, TranscriptionCancelledError | DeepgramError):
        return error
    if isinstance(error, urllib.error.HTTPError):
        reason = _reason(error)
        details = f"HTTP {error.code}: {reason or error.reason}"
        if error.code == 401:
            return DeepgramKeyRejectedError(
                "Deepgram did not accept the API key. Check it, or save a new one.",
                details=details,
            )
        if error.code == 403:
            return DeepgramKeyRejectedError(
                "This Deepgram key is not allowed to transcribe. Use a key "
                "with the Member role or higher.",
                details=details,
            )
        if error.code == 402:
            return DeepgramCreditError(
                "The Deepgram account has run out of credit.", details=details
            )
        if error.code == 504:
            return DeepgramTimeoutError(
                "Deepgram gave up on this recording because it took too long. "
                "Whisper on this computer has no such limit.",
                details=details,
            )
        if error.code == 429 or error.code >= 500:
            return DeepgramUnavailableError(
                "Deepgram is not answering right now. Try again shortly.",
                details=details,
            )
        message = "Deepgram could not transcribe this audio"
        return DeepgramRequestError(
            f"{message}: {reason}" if reason else f"{message}.", details=details
        )
    if isinstance(error, urllib.error.URLError) and isinstance(
        error.reason, TimeoutError
    ):
        error = error.reason
    if isinstance(error, socket.timeout | TimeoutError):
        return DeepgramUnavailableError(
            "Deepgram took too long to answer. Try again shortly.",
            details=str(error),
        )
    if isinstance(error, urllib.error.URLError | OSError):
        return DeepgramUnavailableError(
            "Deepgram could not be reached. Check the internet connection.",
            details=str(getattr(error, "reason", error)),
        )
    return DeepgramUnavailableError(
        "Deepgram could not be reached. Check the internet connection.",
        details=f"{type(error).__name__}: {error}",
    )


__all__ = [
    "DeepgramAdapter",
    "check_and_save_key",
    "keyterms_from",
    "parse_response",
    "supports_keyterms",
]
