"""Guards that keep the local server reachable only from this machine's browser."""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

TOKEN_HEADER = "X-CaptionForge-Token"
TOKEN_QUERY = "t"


def generate_token() -> str:
    """Create a per-process token that authorizes API calls."""
    return secrets.token_urlsafe(32)


def allowed_hosts(port: int) -> frozenset[str]:
    """Host header values a browser on this machine can legitimately send."""
    return frozenset(
        {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
        }
    )


class LocalOnlyMiddleware(BaseHTTPMiddleware):
    """Reject rebound hostnames and unauthorized API calls."""

    def __init__(self, app: ASGIApp, *, token: str, port: int) -> None:
        super().__init__(app)
        self._token = token
        self._hosts = allowed_hosts(port)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Apply the host and token checks, then add hardening headers."""
        host = request.headers.get("host", "")
        if host not in self._hosts:
            return JSONResponse(
                {
                    "error": (
                        "CaptionForge only answers requests addressed to "
                        "localhost on this machine."
                    ),
                    "code": "HostNotAllowed",
                    "retryable": False,
                },
                status_code=403,
            )
        if request.url.path.startswith("/api/") and not self._authorized(request):
            return JSONResponse(
                {
                    "error": (
                        "This page is missing its access token. Open the link "
                        "CaptionForge printed in your terminal."
                    ),
                    "code": "TokenRequired",
                    "retryable": False,
                },
                status_code=401,
            )
        response = await call_next(request)
        self._harden(response)
        return response

    def _authorized(self, request: Request) -> bool:
        """Compare the supplied token in constant time."""
        supplied = request.headers.get(TOKEN_HEADER) or request.query_params.get(
            TOKEN_QUERY, ""
        )
        return secrets.compare_digest(supplied, self._token)

    @staticmethod
    def _harden(response: Response) -> None:
        """Stop the token in the address bar from leaking to third parties."""
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
