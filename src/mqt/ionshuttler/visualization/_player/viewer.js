// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Local viewer page. It lists the views that the Python viewer server holds,
// shows one of them in the player, and sends opened result files to the server.
// Python replays each result; this script only asks for the finished views.

"use strict";

const Viewer = (() => {
  function start(token, drawings) {
    const root = document.querySelector(".viewer");
    const element = (name) => root.querySelector(`.viewer-${name}`);
    const select = element("views");
    const message = element("message");
    const stage = element("stage");
    const template = element("player");
    let stopPlayer = null;
    let shown = null;

    async function request(path, options = {}) {
      let response;
      try {
        response = await fetch(
          `${path}?token=${encodeURIComponent(token)}`,
          options,
        );
      } catch {
        throw new Error(
          "the viewer is no longer running; open it again from Python",
        );
      }
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || response.statusText);
      return body;
    }

    async function refresh(selected = null) {
      const views = await request("/views");
      select.replaceChildren(
        ...views.map((view) => new Option(view.title, String(view.id))),
      );
      element("choice").hidden = views.length === 0;
      element("empty").hidden = views.length > 0;
      const target = selected ?? shown ?? views.at(-1)?.id;
      if (target === undefined || target === null) return;
      select.value = String(target);
      if (target !== shown) await show(target);
    }

    async function show(id) {
      const view = await request(`/views/${id}`);
      const drawing = drawings[view.drawing_name];
      if (drawing === undefined)
        throw new Error(`this viewer cannot draw ${view.drawing_name}`);
      if (stopPlayer !== null) stopPlayer();
      stage.replaceChildren(template.content.cloneNode(true));
      stopPlayer = Player.mount(stage.querySelector(".player"), drawing, view);
      shown = id;
      select.value = String(id);
      document.title = `${view.title} · IonShuttler viewer`;
      history.replaceState(
        null,
        "",
        `${location.pathname}${location.search}#view=${id}`,
      );
    }

    async function open(file) {
      message.textContent = `Opening ${file.name}…`;
      try {
        const opened = await request("/results", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-File-Name": encodeURIComponent(file.name),
          },
          body: await file.text(),
        });
        await refresh(opened.id);
        message.textContent = "";
      } catch (error) {
        message.textContent = `Could not open ${file.name}: ${error.message}`;
      }
    }

    async function run(action) {
      try {
        await action();
      } catch (error) {
        message.textContent = `Error: ${error.message}`;
      }
    }

    root.addEventListener("player-theme", (event) => {
      root.dataset.theme = event.detail;
    });
    select.addEventListener("change", () =>
      run(() => show(Number(select.value))),
    );
    element("file").addEventListener("change", (event) => {
      const [file] = event.target.files;
      if (file !== undefined) open(file);
      event.target.value = "";
    });
    document.addEventListener("dragover", (event) => {
      event.preventDefault();
      root.classList.add("viewer-dropping");
    });
    document.addEventListener("dragleave", (event) => {
      if (event.relatedTarget === null)
        root.classList.remove("viewer-dropping");
    });
    document.addEventListener("drop", (event) => {
      event.preventDefault();
      root.classList.remove("viewer-dropping");
      const [file] = event.dataTransfer.files;
      if (file !== undefined) open(file);
    });
    // Python can add views while this page is open, for example from a notebook.
    window.addEventListener("focus", () => run(() => refresh()));

    const requested = /^#view=(\d+)$/.exec(location.hash);
    run(() => refresh(requested === null ? null : Number(requested[1])));
  }

  return { start };
})();
