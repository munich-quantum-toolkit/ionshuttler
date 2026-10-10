# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the local result viewer and its server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote, urlsplit

import pytest

import mqt.ionshuttler.visualization.viewer as viewer_module
from mqt.ionshuttler.core.result import RESULT_SCHEMA, CompilationResult, CompilationStatus
from mqt.ionshuttler.core.schedule import Schedule, ScheduledAction
from mqt.ionshuttler.grid import GridArchitecture, GridMachineState, Junction, JunctionMove, Segment
from mqt.ionshuttler.visualization import GridVisualizer, open_viewer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mqt.ionshuttler.core.actions import Action

# The sandboxed test environment may set proxy variables; local requests must bypass them.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _grid_result() -> CompilationResult:
    left = Segment("left")
    right = Segment("right")
    architecture = GridArchitecture(
        (left, right),
        (Junction("west", (left.start,)), Junction("center", (left.end, right.start)), Junction("east", (right.end,))),
    )
    move = JunctionMove(left.end, right.start, (0,))
    scheduled: ScheduledAction[Action] = ScheduledAction(0, move, 0, architecture.action_duration(move))
    schedule: Schedule[Action, GridMachineState] = Schedule(
        (scheduled,), scheduled.end_time, architecture.initial_state({"left": (0,)})
    )
    return CompilationResult(
        status=CompilationStatus.SUCCESS,
        schedule=schedule,
        architecture=architecture,
        final_state=architecture.replay_schedule(schedule),
    )


def _request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    assert url.startswith("http://127.0.0.1:")
    method = "POST" if data else "GET"
    # ruff: ignore[suspicious-url-open-usage] - The assertion above limits requests to the local viewer.
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with _LOCAL.open(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        status = error.code
    return status, json.loads(body) if body.startswith(("{", "[")) else body


def _address(server_url: str, path: str) -> str:
    parts = urlsplit(server_url)
    return f"{parts.scheme}://{parts.netloc}{path}?{parts.query}"


@pytest.fixture
def server() -> Iterator[viewer_module._ViewerServer]:
    """Run a viewer server whose views are plain dictionaries."""

    def open_result(data: object, title: str) -> dict[str, object]:
        if data == "unsupported":
            msg = "cannot show this result"
            raise ValueError(msg)
        return {"title": title, "data": data}

    running = viewer_module._ViewerServer(lambda token: f"<html>{token}</html>", open_result)
    yield running
    running.close()


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record browser addresses instead of opening a browser."""
    addresses: list[str] = []
    monkeypatch.setattr(viewer_module, "_open_browser", addresses.append)
    return addresses


def test_server_listens_only_on_this_computer(server: viewer_module._ViewerServer) -> None:
    """Bind the viewer to the loopback address."""
    assert urlsplit(server.url).hostname == "127.0.0.1"


def test_server_requires_the_access_token(server: viewer_module._ViewerServer) -> None:
    """Refuse requests without the random access token."""
    parts = urlsplit(server.url)
    base = f"{parts.scheme}://{parts.netloc}"

    assert _request(f"{base}/")[0] == 403
    assert _request(f"{base}/views?token=wrong")[0] == 403
    assert _request(_address(server.url, "/")) == (200, f"<html>{server.token}</html>")


def test_server_lists_and_returns_views(server: viewer_module._ViewerServer) -> None:
    """Give the page a list of views and each view's data."""
    first = server.add_view({"title": "first", "value": 1})
    second = server.add_view({"title": "second", "value": 2})

    listing = [{"id": 0, "title": "first"}, {"id": 1, "title": "second"}]
    assert _request(_address(server.url, "/views")) == (200, listing)
    assert _request(_address(server.url, f"/views/{second}")) == (200, {"title": "second", "value": 2})
    assert _request(_address(server.url, "/views/7"))[0] == 404
    assert server.view_url(first).endswith("#view=0")


def test_server_turns_uploaded_files_into_views(server: viewer_module._ViewerServer) -> None:
    """Name new views after the uploaded file and report unusable files."""
    headers = {"Content-Type": "application/json", "X-File-Name": quote("run 1.json")}

    status, body = _request(_address(server.url, "/results"), b'{"x": 1}', headers)

    assert status == 200
    assert _request(_address(server.url, f"/views/{body['id']}")) == (200, {"title": "run 1", "data": {"x": 1}})
    status, body = _request(_address(server.url, "/results"), b"{not json", headers)
    assert status == 400
    assert "not valid JSON" in body["error"]
    status, body = _request(_address(server.url, "/results"), b'"unsupported"', headers)
    assert (status, body) == (400, {"error": "cannot show this result"})


def test_viewer_opens_saved_grid_results() -> None:
    """Replay saved Grid results and refuse other files with a clear message."""
    view = viewer_module._open_result(json.loads(_grid_result().to_json()), "saved")

    assert view["title"] == "saved"
    assert view["drawing_name"] == "GridDrawing"
    assert cast("dict[str, Any]", view["drawing"])["panels"][0]["end"] == 1
    with pytest.raises(ValueError, match="not an IonShuttler compilation result"):
        viewer_module._open_result({"schema": "other"}, "other")
    with pytest.raises(ValueError, match="Linear results are not supported yet"):
        viewer_module._open_result({"schema": RESULT_SCHEMA, "architecture": {"num_sites": 2}}, "linear")


def test_viewer_page_contains_the_player_and_the_grid_drawing() -> None:
    """Serve one page with the player, the file picker, and every drawing script."""
    page = viewer_module._page("secret")

    assert "const GridDrawing" in page
    assert 'Viewer.start("secret", { GridDrawing });' in page
    assert 'class="viewer-file"' in page
    assert '<template class="viewer-player"><main class="player"' in page


def test_grid_visualizer_opens_a_result_in_the_viewer(opened: list[str]) -> None:
    """Add the result to the running viewer and open its address."""
    url = GridVisualizer(theme="dark").open(_grid_result(), wait=False)

    assert opened == [url]
    view_id = int(url.rsplit("#view=", 1)[1])
    status, view = _request(_address(url.split("#")[0], f"/views/{view_id}"))
    assert status == 200
    assert view["settings"]["theme"] == "dark"
    assert view["drawing_name"] == "GridDrawing"


@pytest.mark.parametrize(
    ("programs", "expected"),
    [
        ({"wslview": "/usr/bin/wslview", "cmd.exe": "/mnt/c/cmd.exe"}, ["/usr/bin/wslview", "URL"]),
        ({"cmd.exe": "/mnt/c/cmd.exe"}, ["/mnt/c/cmd.exe", "/c", "start", "", "URL"]),
        ({}, None),
    ],
)
def test_wsl_opens_addresses_in_the_windows_browser(
    monkeypatch: pytest.MonkeyPatch,
    programs: dict[str, str],
    expected: list[str] | None,
) -> None:
    """Use a Windows program that opens addresses with a query, not Windows Explorer."""
    monkeypatch.setattr(viewer_module.shutil, "which", programs.get)

    assert viewer_module._windows_browser_command("URL") == expected


def test_open_viewer_opens_the_page(opened: list[str]) -> None:
    """Open the viewer page of this process."""
    url = open_viewer(wait=False)

    assert opened == [url]
    status, page = _request(url)
    assert status == 200
    assert "IonShuttler viewer" in page
