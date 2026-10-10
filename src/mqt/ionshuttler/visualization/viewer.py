# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Open compilation results in a browser viewer served from this Python process.

The viewer is one page for all results. It can show results sent from Python
and saved result files that you open in the page. Python replays each result;
the page only draws it. The viewer listens only on this computer and stops when
the Python process ends.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, unquote, urlsplit

from ._player.page import viewer_page

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

_SERVER: _ViewerServer | None = None
_SERVER_LOCK = threading.Lock()
_MAX_UPLOAD_BYTES = 512 * 1024 * 1024


class _ViewerServer:
    """Serve views and saved results to the local viewer page."""

    def __init__(
        self,
        page: Callable[[str], str],
        open_result: Callable[[object, str], Mapping[str, object]],
    ) -> None:
        """Start serving in a background thread.

        Args:
            page: Function that returns the viewer page for an access token.
            open_result: Function that turns parsed result JSON and a title into
                a view. It raises :class:`ValueError` for unsupported results.
        """
        self.token = secrets.token_urlsafe(24)
        self._page = page
        self._open_result = open_result
        self._views: list[tuple[str, Mapping[str, object]]] = []
        self._lock = threading.Lock()
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _handler(self))
        self._thread = threading.Thread(target=self._http.serve_forever, name="ionshuttler-viewer", daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """Address of the viewer page, including the access token."""
        host, port = self._http.server_address[:2]
        return f"http://{host!s}:{port}/?token={self.token}"

    def add_view(self, view: Mapping[str, object]) -> int:
        """Store a view for the viewer page.

        Returns:
            The view number.
        """
        with self._lock:
            self._views.append((str(view["title"]), view))
            return len(self._views) - 1

    def view_url(self, view_id: int) -> str:
        """Return the address that opens the viewer at one view.

        Returns:
            The address.
        """
        return f"{self.url}#view={view_id}"

    def close(self) -> None:
        """Stop serving and wait for the server thread."""
        self._http.shutdown()
        self._http.server_close()
        self._thread.join()

    def answer(self, method: str, path: str, query: str, body: bytes, file_name: str) -> tuple[int, str, bytes]:
        """Answer one request.

        Returns:
            The status code, content type, and content.
        """
        token = parse_qs(query).get("token", [""])[0]
        if not secrets.compare_digest(token, self.token):
            return _json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong access token"})
        if method == "GET" and path == "/":
            return HTTPStatus.OK, "text/html; charset=utf-8", self._page(self.token).encode()
        if method == "GET" and path == "/views":
            with self._lock:
                listing = [{"id": index, "title": title} for index, (title, _view) in enumerate(self._views)]
            return _json(HTTPStatus.OK, listing)
        if method == "GET" and path.startswith("/views/"):
            with self._lock:
                views = list(self._views)
            number = path.removeprefix("/views/")
            if not number.isdigit() or int(number) >= len(views):
                return _json(HTTPStatus.NOT_FOUND, {"error": "unknown view"})
            return _json(HTTPStatus.OK, views[int(number)][1])
        if method == "POST" and path == "/results":
            return self._receive_result(body, file_name)
        return _json(HTTPStatus.NOT_FOUND, {"error": "unknown address"})

    def _receive_result(self, body: bytes, file_name: str) -> tuple[int, str, bytes]:
        """Turn an uploaded result file into a new view.

        Returns:
            The response with the new view number, or an error message.
        """
        try:
            data = json.loads(body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return _json(HTTPStatus.BAD_REQUEST, {"error": f"the file is not valid JSON ({error})"})
        title = file_name.removesuffix(".json") or "result"
        try:
            view = self._open_result(data, title)
        except (TypeError, ValueError, KeyError) as error:
            return _json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return _json(HTTPStatus.OK, {"id": self.add_view(view)})


def open_viewer(*, wait: bool | None = None) -> str:
    """Open the result viewer in the browser.

    Use **Open result…** in the page, or drop a file on it, to show a saved
    compilation result (``result.save(...)``).

    Args:
        wait: Whether to block until Ctrl+C so the viewer keeps running.
            ``None`` waits in plain scripts and returns at once in notebooks
            and interactive shells, where the viewer runs as long as the
            session.

    Returns:
        The viewer address. Open it yourself if no browser window appears.
    """
    server = _server()
    _open_browser(server.url)
    _wait_if_needed(server.url, wait=wait)
    return server.url


def show_in_viewer(view: Mapping[str, object], *, wait: bool | None = None) -> str:
    """Add a view to the result viewer and open it in the browser.

    Visualizers call this function; most users call their ``open`` method.

    Args:
        view: The view, as created by the visualizer.
        wait: Whether to block until Ctrl+C; see :func:`open_viewer`.

    Returns:
        The viewer address of the view.
    """
    server = _server()
    url = server.view_url(server.add_view(view))
    _open_browser(url)
    _wait_if_needed(url, wait=wait)
    return url


def _server() -> _ViewerServer:
    """Return the viewer server of this process, and start it if needed.

    Returns:
        The running server.
    """
    global _SERVER  # ruff: ignore[global-statement] - One viewer server serves the whole process.
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = _ViewerServer(_page, _open_result)
            logger.info("IonShuttler viewer running at %s", _SERVER.url)
        return _SERVER


def _page(token: str) -> str:
    """Return the viewer page with every drawing script.

    Returns:
        The HTML document.
    """
    return viewer_page(token, {"GridDrawing": files("mqt.ionshuttler.visualization.grid").joinpath("draw.js")})


def _open_result(data: object, title: str) -> Mapping[str, object]:
    """Replay a saved compilation result and return its view.

    Returns:
        The view with the default display settings.

    Raises:
        ValueError: If the data is no compilation result the viewer can show.
    """
    from mqt.ionshuttler.core.result import RESULT_SCHEMA  # ruff: ignore[import-outside-top-level]
    from mqt.ionshuttler.grid.architecture import GRID_ARCHITECTURE_SCHEMA  # ruff: ignore[import-outside-top-level]

    if not isinstance(data, dict) or data.get("schema") != RESULT_SCHEMA:
        msg = "the file is not an IonShuttler compilation result"
        raise ValueError(msg)
    architecture = data.get("architecture")
    if not isinstance(architecture, dict) or architecture.get("schema") != GRID_ARCHITECTURE_SCHEMA:
        msg = "the viewer can show Grid results only; Linear results are not supported yet"
        raise ValueError(msg)

    from mqt.ionshuttler.grid.result import result_from_dict  # ruff: ignore[import-outside-top-level]
    from mqt.ionshuttler.visualization.grid import GridVisualizer  # ruff: ignore[import-outside-top-level]

    view = GridVisualizer().visualize(result_from_dict(data)).player_data()
    view["title"] = title
    return view


def _handler(server: _ViewerServer) -> type[BaseHTTPRequestHandler]:
    """Create the request handler class for one viewer server.

    Returns:
        The handler class.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._respond(b"")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > _MAX_UPLOAD_BYTES:
                self._send(*_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "the file is too large"}))
                return
            self._respond(self.rfile.read(length))

        # http.server writes every request to stderr; send it to the debug log instead.
        # The method and parameter names must match the base class.
        # ruff: ignore[no-self-use, builtin-argument-shadowing]
        def log_message(self, format: str, *args: object) -> None:
            logger.debug(format, *args)

        def _respond(self, body: bytes) -> None:
            address = urlsplit(self.path)
            file_name = unquote(self.headers.get("X-File-Name", "result.json"))
            self._send(*server.answer(self.command, address.path, address.query, body, file_name))

        def _send(self, status: int, content_type: str, content: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def _json(status: int, value: object) -> tuple[int, str, bytes]:
    """Encode a JSON response.

    Returns:
        The status code, content type, and content.
    """
    return status, "application/json", json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _open_browser(url: str) -> None:
    """Open an address in the default browser, including the Windows browser under WSL."""
    if _is_wsl():
        command = _windows_browser_command(url)
        if command is not None:
            # A Windows drive as working directory avoids the warning cmd.exe prints for Linux paths.
            directory = "/mnt/c" if Path("/mnt/c").is_dir() else None
            subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
                command, cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return
    if not webbrowser.open(url):
        logger.warning("Could not open a browser; open %s yourself", url)


def _windows_browser_command(url: str) -> list[str] | None:
    """Return the command that opens an address in the Windows default browser.

    Windows Explorer does not accept addresses with a query, so the viewer uses
    ``wslview`` or the ``start`` command of ``cmd.exe``. The viewer address
    contains only letters, digits, and ``-_:/?=#.``, which ``cmd.exe`` passes on
    unchanged.

    Returns:
        The command, or ``None`` if neither program exists.
    """
    wslview = shutil.which("wslview")
    if wslview is not None:
        return [wslview, url]
    shell = shutil.which("cmd.exe")
    if shell is not None:
        return [shell, "/c", "start", "", url]
    return None


def _is_wsl() -> bool:
    """Return whether Python runs inside the Windows Subsystem for Linux."""
    return "WSL_DISTRO_NAME" in os.environ or "microsoft" in platform.release().lower()


def _wait_if_needed(url: str, *, wait: bool | None) -> None:
    """Keep a plain script alive so its viewer keeps running."""
    if wait is None:
        wait = not _is_interactive()
    if not wait:
        return
    logger.warning("The IonShuttler viewer runs at %s until you press Ctrl+C.", url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def _is_interactive() -> bool:
    """Return whether Python runs a notebook or an interactive shell."""
    return "ipykernel" in sys.modules or hasattr(sys, "ps1") or bool(sys.flags.interactive)


__all__ = ["open_viewer", "show_in_viewer"]
