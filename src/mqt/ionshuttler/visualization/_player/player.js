// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Playback controls, video range, and video export for schedule drawings.
//
// A drawing object supplies four functions:
//   prepare(data) -> view            index the data that Python sent
//   timeBounds(view) -> [start, end] first and last schedule time
//   stepTimes(view) -> [times]       sorted times for the layer buttons
//   drawAtTime(context, view, time, settings, width, height)
// The player calls `drawAtTime` for the visible canvas and for every video
// frame, so playback, the time slider, and export show the same picture. The
// settings it passes always name a concrete theme, "light" or "dark".
//
// A view can name a side bar as `[setting, label]`. The setting is true while
// the side bar is open. The drawing then supplies one more function:
//   details(view, time, settings) -> [{ title, rows: [[label, [lines]]] }]
// A line is a string or a list of parts. A part is a string or a pill
// `{ text, border, fill, ink }`: text in a rounded frame with optional border,
// background, and text colors. The player writes all parts as text.

"use strict";

const Player = (() => {
  const STEP_TOLERANCE = 1e-6;
  const VIDEO_CODECS = [
    { codec: "vp09.00.10.08", codecId: "V_VP9" },
    { codec: "vp8", codecId: "V_VP8" },
  ];

  function decodeData(text) {
    const bytes = Uint8Array.from(atob(text.trim()), (character) =>
      character.charCodeAt(0),
    );
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  function videoFrameTimes(
    startTime,
    endTime,
    timestepsPerSecond,
    framesPerSecond,
  ) {
    const duration = (endTime - startTime) / timestepsPerSecond;
    const count = Math.round(duration * framesPerSecond) + 1;
    if (count === 1) return [startTime];
    return Array.from(
      { length: count },
      (_, index) => startTime + ((endTime - startTime) * index) / (count - 1),
    );
  }

  function clampRange(start, end, changed) {
    if (start <= end) return [start, end];
    return changed === "start" ? [end, end] : [start, start];
  }

  function stepTarget(times, time, direction, startTime, endTime) {
    if (direction > 0) {
      const next = times.find((value) => value > time + STEP_TOLERANCE);
      return next === undefined ? endTime : next;
    }
    let previous = startTime;
    for (const value of times) {
      if (value < time - STEP_TOLERANCE) previous = value;
      else break;
    }
    return previous;
  }

  function resolveTheme(theme) {
    if (theme !== "auto") return theme;
    const query =
      globalThis.matchMedia &&
      globalThis.matchMedia("(prefers-color-scheme: dark)");
    return query && query.matches ? "dark" : "light";
  }

  // ---------------------------------------------------------------------------
  // Video export
  // ---------------------------------------------------------------------------

  async function chooseVideoCodec(width, height, framesPerSecond) {
    if (
      typeof VideoEncoder === "undefined" ||
      typeof VideoFrame === "undefined"
    )
      return null;
    for (const candidate of VIDEO_CODECS) {
      const config = {
        codec: candidate.codec,
        width,
        height,
        framerate: framesPerSecond,
        bitrate: Math.max(
          500000,
          Math.round(width * height * framesPerSecond * 0.1),
        ),
      };
      try {
        const support = await VideoEncoder.isConfigSupported(config);
        if (support.supported)
          return { config: support.config, codecId: candidate.codecId };
      } catch {
        // Try the next codec.
      }
    }
    return null;
  }

  function waitForEncoder(encoder) {
    return new Promise((resolve) => {
      const done = () => {
        clearTimeout(timer);
        if (typeof encoder.removeEventListener === "function")
          encoder.removeEventListener("dequeue", done);
        resolve();
      };
      const timer = setTimeout(done, 20);
      if (typeof encoder.addEventListener === "function")
        encoder.addEventListener("dequeue", done);
    });
  }

  // Draw each frame at a fixed schedule time and encode it with a fixed
  // timestamp. The export never waits for real time, so it can run faster
  // than the video plays.
  async function exportVideo(drawing, view, settings, report, isCancelled) {
    const width = settings.width;
    const height = settings.height;
    const framesPerSecond = settings.frames_per_second;
    const choice = await chooseVideoCodec(width, height, framesPerSecond);
    if (choice === null) {
      throw new Error(
        "This browser cannot encode WebM video. Use a current Chromium-based browser, " +
          "or export the video in Python.",
      );
    }
    const times = videoFrameTimes(
      settings.video_start_time,
      settings.video_end_time,
      settings.timesteps_per_second,
      framesPerSecond,
    );
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    const writer = new WebMWriter(
      choice.codecId,
      width,
      height,
      framesPerSecond,
    );
    let failure = null;
    const encoder = new VideoEncoder({
      output: (chunk) => writer.addChunk(chunk),
      error: (error) => {
        failure = error;
      },
    });
    encoder.configure(choice.config);
    const keyFrameInterval = Math.max(1, Math.round(framesPerSecond * 2));
    const frameDuration = 1e6 / framesPerSecond;
    try {
      for (let index = 0; index < times.length; index += 1) {
        if (isCancelled()) return null;
        if (failure !== null) throw failure;
        drawing.drawAtTime(
          context,
          view,
          times[index],
          settings,
          width,
          height,
        );
        const frame = new VideoFrame(canvas, {
          timestamp: Math.round(index * frameDuration),
          duration: Math.round(frameDuration),
        });
        encoder.encode(frame, { keyFrame: index % keyFrameInterval === 0 });
        frame.close();
        report(index + 1, times.length);
        while (encoder.encodeQueueSize > 4) await waitForEncoder(encoder);
        if (index % 10 === 9)
          await new Promise((resolve) => setTimeout(resolve, 0));
      }
      await encoder.flush();
      if (failure !== null) throw failure;
      return writer.finish();
    } finally {
      if (encoder.state !== "closed") encoder.close();
    }
  }

  function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  // ---------------------------------------------------------------------------
  // Page controls
  // ---------------------------------------------------------------------------

  let mounted = 0;

  // Connect the controls inside `root` to one view. The returned function stops
  // playback and releases the observers, so a page can replace the view.
  function mount(root, drawing, data) {
    mounted += 1;
    const view = drawing.prepare(data.drawing);
    const settings = { ...data.settings, editing: false, highlight: null };
    const [startTime, endTime] = drawing.timeBounds(view);
    const stepTimes = drawing.stepTimes(view);
    const element = (name) => root.querySelector(`.player-${name}`);
    const canvas = element("canvas");
    canvas.style.aspectRatio = `${settings.width} / ${settings.height}`;
    const context = canvas.getContext("2d");
    const timeInput = element("time");
    const timeLabel = element("time-label");
    const playButton = element("play");
    const rangeStart = element("range-start");
    const rangeEnd = element("range-end");
    const exportPanel = element("export");
    const exportToggle = element("export-toggle");
    const exportStart = element("export-start");
    const exportCancel = element("export-cancel");
    const editToggle = element("edit-toggle");
    const editBar = element("edit-bar");
    const progress = element("progress");
    const message = element("export-message");
    const canEdit = typeof drawing.pick === "function";
    const sidebar =
      data.sidebar !== undefined && typeof drawing.details === "function"
        ? data.sidebar
        : null;
    const sidebarPanel = element("sidebar");
    const sidebarToggle = element("sidebar-toggle");
    let shownDetails = null;
    let time = startTime;
    let playing = false;
    let lastFrame = null;
    let cancelled = false;
    let dragging = null;
    let frameRequested = false;
    // Every listener uses this signal, so the stop function removes them all.
    const listeners = new AbortController();
    const { signal } = listeners;

    function render() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (width === 0 || height === 0) return;
      const ratio = globalThis.devicePixelRatio || 1;
      const bitmapWidth = Math.round(width * ratio);
      const bitmapHeight = Math.round(height * ratio);
      if (canvas.width !== bitmapWidth || canvas.height !== bitmapHeight) {
        canvas.width = bitmapWidth;
        canvas.height = bitmapHeight;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const shown = { ...settings, theme: resolveTheme(settings.theme) };
      drawing.drawAtTime(context, view, time, shown, width, height);
      renderDetails(shown);
      timeInput.value = String(time);
      const span = Math.max(endTime - startTime, 1e-9);
      timeInput.style.setProperty(
        "--progress",
        `${((time - startTime) / span) * 100}%`,
      );
      timeLabel.textContent = `${time.toFixed(1)} / ${endTime}`;
    }

    // Write the drawing's details into the open side bar. The page changes
    // only when the details change, not on every frame.
    function renderDetails(shown) {
      if (sidebar === null || !settings[sidebar[0]]) return;
      const sections = drawing.details(view, time, shown) ?? [];
      const key = JSON.stringify(sections);
      if (key === shownDetails) return;
      shownDetails = key;
      element("sidebar-body").replaceChildren(
        ...sections.map(({ title, rows }) => {
          const section = document.createElement("section");
          section.className = "player-details";
          if (title) {
            const heading = document.createElement("h3");
            heading.textContent = title;
            section.append(heading);
          }
          const list = document.createElement("dl");
          for (const [label, lines] of rows) {
            const term = document.createElement("dt");
            term.textContent = label;
            const value = document.createElement("dd");
            value.append(
              ...(lines.length === 0 ? ["none"] : lines).map((parts) => {
                const line = document.createElement("div");
                line.append(
                  ...(typeof parts === "string" ? [parts] : parts).map(
                    detailPart,
                  ),
                );
                return line;
              }),
            );
            list.append(term, value);
          }
          section.append(list);
          return section;
        }),
      );
    }

    // Each part is its own element, so the line keeps a gap between parts.
    function detailPart(part) {
      if (typeof part === "string") {
        const text = document.createElement("span");
        text.textContent = part;
        return text;
      }
      const pill = document.createElement("span");
      pill.className = "player-pill";
      pill.textContent = part.text;
      for (const [name, color] of [
        ["border", part.border],
        ["fill", part.fill],
        ["ink", part.ink],
      ]) {
        if (color) pill.style.setProperty(`--pill-${name}`, color);
      }
      return pill;
    }

    function setSidebar(open) {
      settings[sidebar[0]] = open;
      sidebarPanel.hidden = !open;
      sidebarToggle.setAttribute("aria-expanded", String(open));
      shownDetails = null;
      render();
    }

    // Redraw once per animation frame, however many events arrive.
    function requestRender() {
      if (frameRequested) return;
      frameRequested = true;
      requestAnimationFrame(() => {
        frameRequested = false;
        render();
      });
    }

    // The page background follows the player theme, so no light border shows
    // around a dark player. Surrounding page parts can follow the event.
    function applyTheme() {
      root.dataset.theme = settings.theme;
      document.body.style.background = getComputedStyle(root).backgroundColor;
      root.dispatchEvent(
        new CustomEvent("player-theme", {
          bubbles: true,
          detail: settings.theme,
        }),
      );
      render();
    }

    function setTime(value) {
      time = Math.min(Math.max(value, startTime), endTime);
      render();
    }

    function setPlaying(value) {
      playing = value;
      const label = playing ? "Pause" : "Play";
      playButton.setAttribute("aria-label", label);
      playButton.title = label;
      element("play-icon").toggleAttribute("hidden", playing);
      element("pause-icon").toggleAttribute("hidden", !playing);
      if (playing) {
        if (time >= endTime) time = startTime;
        lastFrame = null;
        requestAnimationFrame(tick);
      }
    }

    function tick(now) {
      if (!playing) return;
      if (lastFrame !== null)
        time += ((now - lastFrame) / 1000) * settings.timesteps_per_second;
      lastFrame = now;
      if (time >= endTime) {
        time = endTime;
        setPlaying(false);
      }
      render();
      if (playing) requestAnimationFrame(tick);
    }

    function updateRange(changed) {
      const [start, end] = clampRange(
        Number(rangeStart.value),
        Number(rangeEnd.value),
        changed,
      );
      rangeStart.value = String(start);
      rangeEnd.value = String(end);
      settings.video_start_time = start;
      settings.video_end_time = end;
      const span = Math.max(endTime - startTime, 1);
      const fill = element("range-fill");
      fill.style.left = `${((start - startTime) / span) * 100}%`;
      fill.style.right = `${((endTime - end) / span) * 100}%`;
      element("range-label").textContent = `Timesteps ${start} – ${end}`;
      updateDuration();
    }

    function updateDuration() {
      const frames = videoFrameTimes(
        settings.video_start_time,
        settings.video_end_time,
        settings.timesteps_per_second,
        settings.frames_per_second,
      ).length;
      element("duration").textContent =
        `${frames} frames · ${(frames / settings.frames_per_second).toFixed(1)} s`;
    }

    function readPositive(input, fallback) {
      const value = Number(input.value);
      return Number.isFinite(value) && value > 0 ? value : fallback;
    }

    // Layout editing: drag junctions on the canvas. The drawing decides what a
    // canvas point hits and how a drag changes the view.
    function setEditing(value) {
      settings.editing = value;
      closePaint();
      editToggle.setAttribute("aria-pressed", String(value));
      editBar.hidden = !value;
      canvas.classList.toggle("player-editing", value);
      canvas.classList.remove(
        "player-grab",
        "player-grabbing",
        "player-pointer",
      );
      render();
    }

    function canvasPoint(event) {
      const box = canvas.getBoundingClientRect();
      return [event.clientX - box.left, event.clientY - box.top];
    }

    function sameHandle(first, second) {
      if (first === null || second === null) return first === second;
      return (
        first.kind === second.kind &&
        first.panel === second.panel &&
        first.junction === second.junction &&
        first.ion === second.ion &&
        first.zone === second.zone
      );
    }

    // A small color editor for the picked ion or processing zone.
    function openPaint(handle, point) {
      const shown = { ...settings, theme: resolveTheme(settings.theme) };
      const content = drawing.paintParts(view, handle, shown, time);
      if (content === null) return;
      const panel = element("paint");
      element("paint-title").textContent = content.title;
      const fields = element("paint-fields");
      fields.replaceChildren(
        ...content.parts.map(({ part, label, value }) => {
          const field = document.createElement("label");
          field.className = "player-paint-field";
          const input = document.createElement("input");
          input.type = "color";
          input.value = value;
          input.addEventListener(
            "input",
            () => {
              drawing.paint(view, handle, part, input.value);
              requestRender();
            },
            { signal },
          );
          field.append(input, label);
          return field;
        }),
      );
      element("paint-reset").onclick = () => {
        drawing.paint(view, handle, null, null);
        closePaint();
        render();
      };
      const stage = element("stage").getBoundingClientRect();
      panel.hidden = false;
      const left = Math.min(
        Math.max(point[0] + 14, 8),
        stage.width - panel.offsetWidth - 8,
      );
      const top = Math.min(
        Math.max(point[1] - panel.offsetHeight / 2, 8),
        stage.height - panel.offsetHeight - 8,
      );
      panel.style.left = `${left}px`;
      panel.style.top = `${top}px`;
      settings.highlight = handle;
      render();
    }

    function closePaint() {
      element("paint").hidden = true;
      settings.highlight = null;
    }

    function onPointerDown(event) {
      if (!settings.editing) return;
      const point = canvasPoint(event);
      const handle = drawing.pick(view, ...point);
      if (handle === null) {
        closePaint();
        render();
        return;
      }
      event.preventDefault();
      if (handle.kind !== "junction") {
        openPaint(handle, point);
        return;
      }
      closePaint();
      dragging = handle;
      drawing.beginDrag(view, handle);
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add("player-grabbing");
      settings.highlight = handle;
      requestRender();
    }

    function onPointerMove(event) {
      if (!settings.editing) return;
      if (dragging !== null) {
        drawing.dragTo(view, dragging, ...canvasPoint(event));
        requestRender();
        return;
      }
      const handle = drawing.pick(view, ...canvasPoint(event));
      canvas.classList.toggle(
        "player-grab",
        handle !== null && handle.kind === "junction",
      );
      canvas.classList.toggle(
        "player-pointer",
        handle !== null && handle.kind !== "junction",
      );
      if (!element("paint").hidden) return;
      if (!sameHandle(handle, settings.highlight)) {
        settings.highlight = handle;
        requestRender();
      }
    }

    function onPointerUp(event) {
      if (dragging === null) return;
      drawing.endDrag(view, dragging);
      dragging = null;
      canvas.releasePointerCapture(event.pointerId);
      canvas.classList.remove("player-grabbing");
      requestRender();
    }

    async function copySettings() {
      const text = drawing.settingsText(view, settings);
      const button = element("copy");
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
      } catch {
        globalThis.prompt("Copy these visualizer settings:", text);
      }
      setTimeout(() => {
        button.textContent = "Copy settings";
      }, 1500);
    }

    timeInput.min = String(startTime);
    timeInput.max = String(endTime);
    for (const input of [rangeStart, rangeEnd]) {
      input.min = String(startTime);
      input.max = String(endTime);
    }
    rangeStart.value = String(settings.video_start_time);
    rangeEnd.value = String(settings.video_end_time);
    element("speed").value = String(settings.timesteps_per_second);
    element("fps").value = String(settings.frames_per_second);
    for (const radio of root.querySelectorAll(".player-segmented input")) {
      radio.name = `player-theme-${mounted}`;
      radio.checked = radio.value === settings.theme;
      radio.addEventListener(
        "change",
        () => {
          settings.theme = radio.value;
          applyTheme();
        },
        { signal },
      );
    }
    for (const [setting, label, choices] of data.options) {
      if (choices !== undefined) {
        // A choice between named values, shown as a segmented switch.
        const group = document.createElement("div");
        group.className = "player-segmented";
        group.setAttribute("role", "radiogroup");
        group.setAttribute("aria-label", label);
        for (const [value, text] of choices) {
          const option = document.createElement("label");
          const radio = document.createElement("input");
          radio.type = "radio";
          radio.name = `player-${setting}-${mounted}`;
          radio.value = value;
          radio.checked = settings[setting] === value;
          radio.addEventListener(
            "change",
            () => {
              settings[setting] = value;
              render();
            },
            { signal },
          );
          const caption = document.createElement("span");
          caption.textContent = text;
          option.append(radio, caption);
          group.append(option);
        }
        const title = document.createElement("span");
        title.className = "player-choice-label";
        title.textContent = label;
        element("choices").append(title, group);
        continue;
      }
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.setting = setting;
      checkbox.checked = settings[setting];
      checkbox.addEventListener(
        "change",
        () => {
          settings[setting] = checkbox.checked;
          render();
        },
        { signal },
      );
      // A switch: the checkbox stays in the page for keyboards and screen readers.
      checkbox.setAttribute("role", "switch");
      const track = document.createElement("span");
      track.className = "player-switch-track";
      track.setAttribute("aria-hidden", "true");
      const toggle = document.createElement("label");
      toggle.className = "player-switch";
      toggle.append(checkbox, track, label);
      element("chips").append(toggle);
    }

    playButton.addEventListener("click", () => setPlaying(!playing), {
      signal,
    });
    element("previous").addEventListener(
      "click",
      () => {
        setPlaying(false);
        setTime(stepTarget(stepTimes, time, -1, startTime, endTime));
      },
      { signal },
    );
    element("next").addEventListener(
      "click",
      () => {
        setPlaying(false);
        setTime(stepTarget(stepTimes, time, 1, startTime, endTime));
      },
      { signal },
    );
    timeInput.addEventListener(
      "input",
      () => setTime(Number(timeInput.value)),
      { signal },
    );
    element("speed").addEventListener(
      "change",
      (event) => {
        settings.timesteps_per_second = readPositive(
          event.target,
          settings.timesteps_per_second,
        );
        event.target.value = String(settings.timesteps_per_second);
        updateDuration();
      },
      { signal },
    );
    element("fps").addEventListener(
      "change",
      (event) => {
        settings.frames_per_second = Math.max(
          1,
          Math.round(readPositive(event.target, settings.frames_per_second)),
        );
        event.target.value = String(settings.frames_per_second);
        updateDuration();
      },
      { signal },
    );
    rangeStart.addEventListener("input", () => updateRange("start"), {
      signal,
    });
    rangeEnd.addEventListener("input", () => updateRange("end"), { signal });
    exportToggle.addEventListener(
      "click",
      () => {
        exportPanel.hidden = !exportPanel.hidden;
        exportToggle.setAttribute("aria-expanded", String(!exportPanel.hidden));
        // The canvas can fill the window, so the panel can open below its
        // bottom edge. Scroll the opened panel into view.
        if (!exportPanel.hidden)
          exportPanel.scrollIntoView({ block: "nearest", behavior: "smooth" });
      },
      { signal },
    );
    if (canEdit) {
      editToggle.hidden = false;
      // The drawing says what the editor can change.
      if (typeof drawing.editHint === "string")
        element("edit-hint").textContent = drawing.editHint;
      editToggle.addEventListener(
        "click",
        () => setEditing(!settings.editing),
        { signal },
      );
      element("done").addEventListener("click", () => setEditing(false), {
        signal,
      });
      // A drawing with a `marker_size` setting gets a size slider. It scales
      // the drawn ions and junctions without moving any junction.
      const sizeInput = element("size-input");
      const hasSize = typeof settings.marker_size === "number";
      element("size").hidden = !hasSize;
      if (hasSize) {
        sizeInput.value = String(settings.marker_size);
        sizeInput.addEventListener(
          "input",
          () => {
            settings.marker_size = Number(sizeInput.value);
            requestRender();
          },
          { signal },
        );
      }
      element("reset").addEventListener(
        "click",
        () => {
          drawing.resetLayout(view);
          if (hasSize) {
            settings.marker_size = data.settings.marker_size;
            sizeInput.value = String(settings.marker_size);
          }
          render();
        },
        { signal },
      );
      element("copy").addEventListener("click", copySettings, { signal });
      element("paint-close").addEventListener(
        "click",
        () => {
          closePaint();
          render();
        },
        { signal },
      );
      canvas.addEventListener("pointerdown", onPointerDown, { signal });
      canvas.addEventListener("pointermove", onPointerMove, { signal });
      canvas.addEventListener("pointerup", onPointerUp, { signal });
      canvas.addEventListener("pointercancel", onPointerUp, { signal });
    }
    if (sidebar !== null) {
      const [setting, label] = sidebar;
      element("sidebar-label").textContent = label;
      element("sidebar-title").textContent = label;
      sidebarPanel.setAttribute("aria-label", label);
      sidebarToggle.hidden = false;
      sidebarToggle.addEventListener(
        "click",
        () => setSidebar(!settings[setting]),
        { signal },
      );
      element("sidebar-close").addEventListener(
        "click",
        () => setSidebar(false),
        { signal },
      );
      sidebarPanel.hidden = !settings[setting];
      sidebarToggle.setAttribute("aria-expanded", String(!sidebarPanel.hidden));
    }
    exportCancel.addEventListener(
      "click",
      () => {
        cancelled = true;
      },
      { signal },
    );
    exportStart.addEventListener(
      "click",
      async () => {
        cancelled = false;
        exportStart.disabled = true;
        exportCancel.hidden = false;
        progress.hidden = false;
        progress.value = 0;
        message.textContent = "Encoding…";
        const exported = {
          ...settings,
          theme: resolveTheme(settings.theme),
          editing: false,
          highlight: null,
        };
        try {
          const blob = await exportVideo(
            drawing,
            view,
            exported,
            (done, total) => {
              progress.max = total;
              progress.value = done;
            },
            () => cancelled,
          );
          if (blob === null) {
            message.textContent = "Export cancelled.";
          } else {
            download(
              blob,
              `schedule-${exported.video_start_time}-${exported.video_end_time}.webm`,
            );
            message.textContent = `Saved ${(blob.size / 1e6).toFixed(1)} MB.`;
          }
        } catch (error) {
          message.textContent = `Export failed: ${error.message}`;
        } finally {
          exportStart.disabled = false;
          exportCancel.hidden = true;
          progress.hidden = true;
        }
      },
      { signal },
    );

    const query =
      globalThis.matchMedia &&
      globalThis.matchMedia("(prefers-color-scheme: dark)");
    const followSystemTheme = () => {
      if (settings.theme === "auto") applyTheme();
    };
    if (query && typeof query.addEventListener === "function")
      query.addEventListener("change", followSystemTheme, { signal });
    const resizeObserver = new ResizeObserver(render);
    resizeObserver.observe(canvas);
    updateRange("end");
    applyTheme();

    return () => {
      playing = false;
      cancelled = true;
      resizeObserver.disconnect();
      listeners.abort();
    };
  }

  // Inside a notebook, the page lives in a same-origin frame. Grow the frame
  // with the page, for example when the export panel opens.
  function fitFrame() {
    try {
      const frame = globalThis.frameElement;
      if (!frame) return;
      new ResizeObserver(() => {
        frame.style.height = `${document.documentElement.scrollHeight}px`;
      }).observe(document.body);
    } catch {
      // The page is not inside a same-origin frame.
    }
  }

  // Start every player on a standalone page from the view it embeds.
  function start(drawing) {
    for (const root of document.querySelectorAll(".player")) {
      mount(
        root,
        drawing,
        decodeData(root.querySelector(".player-data").textContent),
      );
    }
    fitFrame();
  }

  return {
    clampRange,
    decodeData,
    exportVideo,
    mount,
    resolveTheme,
    start,
    stepTarget,
    videoFrameTimes,
  };
})();

if (typeof module === "object" && module.exports) module.exports = Player;
