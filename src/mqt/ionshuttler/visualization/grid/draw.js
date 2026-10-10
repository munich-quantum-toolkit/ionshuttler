// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// Draw Grid results for the schedule player.
//
// Python replays the schedule and sends each ion's movements as places: a
// segment and a fraction of the way along it. This script turns places into
// coordinates from the current junction positions, so moving a junction moves
// every segment and ion attached to it. It never decides whether an action is
// legal. The Matplotlib renderer in `_matplotlib.py` uses the same layout rules.

"use strict";

const GridDrawing = (() => {
  const ION_RADIUS_PER_SPACING = 0.35;
  const MAX_ION_RADIUS = 0.06;
  const MARGIN_PER_ION_RADIUS = 2.5;
  const FONT =
    "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif";

  // ---------------------------------------------------------------------------
  // Colors; same rules as `_colors.py`
  // ---------------------------------------------------------------------------

  const ION_PALETTE = [
    "#3b6ea8",
    "#d1752e",
    "#c4484f",
    "#4f9a94",
    "#5a9a4b",
    "#b8962e",
    "#8e6aa0",
    "#c76d8e",
    "#8a6a52",
    "#6f7782",
    "#2f8fb8",
    "#7d8a35",
  ];
  const ZONE_PALETTE = [
    "#c08a2e",
    "#3f8f88",
    "#b05d78",
    "#6f68a8",
    "#6f8f3a",
    "#3f7fa8",
    "#b3653a",
    "#8a5a5a",
  ];
  const NEUTRAL_BORDER = "#8a919c";
  const WHITE = "#ffffff";
  const DARK_TEXT = "#1c2430";

  function hsl(hue, saturation, lightness) {
    const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const second = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
    const sector = Math.floor(hue / 60);
    const [red, green, blue] = [
      [chroma, second, 0],
      [second, chroma, 0],
      [0, chroma, second],
      [0, second, chroma],
      [second, 0, chroma],
    ][sector] ?? [chroma, 0, second];
    const shift = lightness - chroma / 2;
    return `#${[red, green, blue]
      .map((value) =>
        Math.floor((value + shift) * 255 + 0.5)
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")}`;
  }

  function distinctColor(index, palette) {
    return index < palette.length
      ? palette[index]
      : hsl((index * 137.508) % 360, 0.42, 0.48);
  }

  function textColor(fill) {
    const linear = (channel) =>
      channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    const [red, green, blue] = [1, 3, 5].map((index) =>
      linear(parseInt(fill.slice(index, index + 2), 16) / 255),
    );
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.35
      ? DARK_TEXT
      : WHITE;
  }

  // Ring, fill, and text color of an ion. Colors chosen in the view win. Ions
  // without a chosen fill use `plainFill`, which depends on the theme.
  function ionColors(view, ion, time, mode, single, plainFill = WHITE) {
    let border = NEUTRAL_BORDER;
    let fill = plainFill;
    if (mode === "distinct") border = distinctColor(ion, ION_PALETTE);
    else if (mode === "single") border = single;
    else {
      const timeline = view.ionChanges[ion] ?? [];
      const index = bisectRight(
        timeline.map((change) => change[0]),
        time,
      );
      if (index > 0) {
        const [, chosenBorder, chosenFill] = timeline[index - 1];
        border = chosenBorder ?? border;
        fill = chosenFill ?? fill;
      }
    }
    const edit = view.ionEdits[ion];
    if (edit !== undefined) {
      border = edit.border ?? border;
      fill = edit.fill ?? fill;
    }
    return { border, fill, text: textColor(fill) };
  }

  function legendEntries(view, plainFill) {
    const entries = new Map();
    for (const ion of Object.keys(view.ionChanges)
      .map(Number)
      .sort((a, b) => a - b)) {
      for (const [, border, fill, label] of view.ionChanges[ion]) {
        if (label !== null && !entries.has(label))
          entries.set(label, [border ?? NEUTRAL_BORDER, fill ?? plainFill]);
      }
    }
    return [...entries].map(([label, [border, fill]]) => ({
      label,
      border,
      fill,
    }));
  }

  function zoneColor(view, panel, zone) {
    return view.zoneEdits[zone] ?? panel.zoneColors[zone];
  }

  // ---------------------------------------------------------------------------
  // Drawing data and time queries
  // ---------------------------------------------------------------------------

  function place(segment, fraction) {
    return { segment, fraction };
  }

  function preparePanel(raw) {
    const ions = raw.ions.map(([ion, segment, fraction, movements]) => ({
      ion,
      initial: place(segment, fraction),
      movements: movements.map((item) => ({
        start: item[0],
        end: item[1],
        target: place(item[2], item[3]),
        via: item.length > 4 ? item[4] : null,
      })),
      starts: movements.map((item) => item[0]),
    }));
    const layers = raw.layers.map(([start, end, description, gates]) => ({
      start,
      end,
      description,
      gates: gates.map(([gateIons, zone, gateStart, gateEnd]) => ({
        ions: gateIons,
        zone,
        start: gateStart,
        end: gateEnd,
      })),
    }));
    const gates = layers.flatMap((layer) => layer.gates);
    const end = ([junction, x, y]) => ({ junction, offset: [x, y] });
    const junctions = raw.junctions.map(([, x, y]) => [x, y]);
    return {
      title: raw.title,
      start: raw.start,
      end: raw.end,
      junctionIds: raw.junctions.map(([id]) => id),
      junctions,
      originalJunctions: junctions.map((point) => [...point]),
      segments: raw.segments.map(([id, first, second, capacity, zones]) => ({
        id,
        start: end(first),
        end: end(second),
        capacity,
        zones,
      })),
      zones: raw.processing_zones.map(([id, segment, fraction]) => ({
        id,
        place: place(segment, fraction),
      })),
      ions,
      layers,
      layerStarts: layers.map((layer) => layer.start),
      gates,
      gateStarts: gates.map((gate) => gate.start),
      longestGate: gates.reduce(
        (longest, gate) => Math.max(longest, gate.end - gate.start),
        0,
      ),
      frozen: null,
      layout: null,
    };
  }

  function prepare(data) {
    const zoneOverrides = data.zone_colors ?? {};
    const panels = data.panels.map(preparePanel);
    for (const panel of panels) {
      panel.zoneColors = Object.fromEntries(
        panel.zones.map((zone, index) => [
          zone.id,
          zoneOverrides[zone.id] ?? distinctColor(index, ZONE_PALETTE),
        ]),
      );
    }
    return {
      colors: data.colors,
      panels,
      ionChanges: data.ion_colors ?? {},
      ionEdits: {},
      zoneEdits: {},
    };
  }

  function timeBounds(view) {
    return [
      Math.min(...view.panels.map((panel) => panel.start)),
      Math.max(...view.panels.map((panel) => panel.end)),
    ];
  }

  function stepTimes(view) {
    return [...new Set(view.panels.flatMap((panel) => panel.layerStarts))].sort(
      (a, b) => a - b,
    );
  }

  function bisectRight(values, value) {
    let low = 0;
    let high = values.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (values[middle] <= value) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function isRunning(start, end, time) {
    return (start <= time && time < end) || start === time;
  }

  function interpolate(start, end, progress) {
    return [
      start[0] + progress * (end[0] - start[0]),
      start[1] + progress * (end[1] - start[1]),
    ];
  }

  function distance(first, second) {
    return Math.hypot(second[0] - first[0], second[1] - first[1]);
  }

  // Same rules as `geometry` in `_scene.py`.
  function geometry(panel) {
    const locate = (end) => {
      if (end.junction === null) return end.offset;
      const [x, y] = panel.junctions[end.junction];
      return [x + end.offset[0], y + end.offset[1]];
    };
    const segments = panel.segments.map((segment) => [
      locate(segment.start),
      locate(segment.end),
    ]);
    const points = [...panel.junctions, ...segments.flat()];
    const xs = points.map((point) => point[0]);
    const ys = points.map((point) => point[1]);
    const [minX, maxX, minY, maxY] = [
      Math.min(...xs),
      Math.max(...xs),
      Math.min(...ys),
      Math.max(...ys),
    ];
    const size = Math.max(maxX - minX, maxY - minY) || 1;
    const spacings = panel.segments
      .map(
        (segment, index) =>
          distance(...segments[index]) / (segment.capacity + 1),
      )
      .filter((spacing, index) => distance(...segments[index]) > 0);
    const ionRadius = spacings.length
      ? Math.min(
          ION_RADIUS_PER_SPACING * Math.min(...spacings),
          MAX_ION_RADIUS * size,
        )
      : size / 40;
    const margin = MARGIN_PER_ION_RADIUS * ionRadius;
    return {
      junctions: panel.junctions,
      segments,
      bounds: [minX - margin, minY - margin, maxX + margin, maxY + margin],
      ionRadius,
    };
  }

  function position(shape, where) {
    const [start, end] = shape.segments[where.segment];
    return interpolate(start, end, where.fraction);
  }

  function alongPath(source, via, target, progress) {
    if (via === null) return interpolate(source, target, progress);
    const first = distance(source, via);
    const second = distance(via, target);
    const total = first + second;
    if (total === 0) return target;
    const travelled = progress * total;
    if (travelled <= first)
      return first > 0 ? interpolate(source, via, travelled / first) : via;
    return interpolate(via, target, (travelled - first) / second);
  }

  function trackPosition(track, time, shape) {
    const index = bisectRight(track.starts, time) - 1;
    if (index < 0) return position(shape, track.initial);
    const movement = track.movements[index];
    const target = position(shape, movement.target);
    if (time >= movement.end) return target;
    const source = position(
      shape,
      index === 0 ? track.initial : track.movements[index - 1].target,
    );
    const via = movement.via === null ? null : shape.junctions[movement.via];
    const progress = (time - movement.start) / (movement.end - movement.start);
    return alongPath(source, via, target, progress);
  }

  function runningGates(panel, time) {
    const running = [];
    let index = bisectRight(panel.gateStarts, time);
    const earliestStart = time - panel.longestGate;
    while (index > 0 && panel.gates[index - 1].start >= earliestStart) {
      index -= 1;
      const gate = panel.gates[index];
      if (isRunning(gate.start, gate.end, time)) running.push(gate);
    }
    return running.reverse();
  }

  function currentLayer(panel, time) {
    const index = bisectRight(panel.layerStarts, time) - 1;
    if (index < 0) return null;
    const layer = panel.layers[index];
    return isRunning(layer.start, layer.end, time) ? layer : null;
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  // Same rules as `panel_layout` in `_scene.py`.
  function panelLayout(shape, left, top, width, height) {
    const fontSize = Math.max(
      11,
      Math.min(17, Math.round(Math.min(width, height) * 0.026)),
    );
    const headerHeight = Math.round(fontSize * 3.6);
    const padding = Math.round(Math.min(width, height) * 0.03);
    const availableWidth = Math.max(width - 2 * padding, 1);
    const availableHeight = Math.max(height - headerHeight - 2 * padding, 1);
    const [minX, minY, maxX, maxY] = shape.bounds;
    const extentX = Math.max(maxX - minX, 1e-9);
    const extentY = Math.max(maxY - minY, 1e-9);
    const scale = Math.min(availableWidth / extentX, availableHeight / extentY);
    const origin = [
      left + padding + (availableWidth - extentX * scale) / 2,
      top + headerHeight + padding + (availableHeight - extentY * scale) / 2,
    ];
    return {
      left,
      top,
      width,
      height,
      scale,
      fontSize,
      headerHeight,
      padding,
      ionRadius: Math.min(
        shape.ionRadius * scale,
        0.045 * Math.min(width, height),
      ),
      point: ([x, y]) => [
        origin[0] + (x - minX) * scale,
        origin[1] + (maxY - y) * scale,
      ],
      coordinate: ([x, y]) => [
        minX + (x - origin[0]) / scale,
        maxY - (y - origin[1]) / scale,
      ],
    };
  }

  function labelNormal(start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const length = Math.hypot(dx, dy);
    let normal = length === 0 ? [0, 1] : [-dy / length, dx / length];
    if (normal[1] < 0 || (normal[1] === 0 && normal[0] > 0))
      normal = [-normal[0], -normal[1]];
    return normal;
  }

  function ionLabelFontSize(ionRadius, label) {
    return Math.min(
      0.85 * ionRadius,
      (2.3 * ionRadius) / Math.max(label.length, 2),
    );
  }

  function headerLines(panel, time) {
    const shownTime = Math.min(Math.max(time, panel.start), panel.end);
    const layer = currentLayer(panel, shownTime);
    return [
      panel.title,
      `t ${shownTime.toFixed(1)} / ${panel.end}`,
      layer === null ? "" : layer.description,
    ];
  }

  function truncate(text, width, fontSize) {
    const limit = Math.max(Math.floor(width / (0.55 * fontSize)), 1);
    return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
  }

  // ---------------------------------------------------------------------------
  // Drawing
  // ---------------------------------------------------------------------------

  function drawAtTime(context, view, time, settings, width, height) {
    const colors = view.colors[settings.theme];
    context.save();
    context.fillStyle = colors.background;
    context.fillRect(0, 0, width, height);
    const panelWidth = width / view.panels.length;
    view.panels.forEach((panel, index) => {
      const shape = geometry(panel);
      if (panel.frozen !== null) Object.assign(shape, panel.frozen);
      const layout = panelLayout(
        shape,
        index * panelWidth,
        0,
        panelWidth,
        height,
      );
      if (index > 0) {
        context.fillStyle = colors.chip;
        context.fillRect(
          Math.round(index * panelWidth),
          layout.padding,
          1,
          height - 2 * layout.padding,
        );
      }
      panel.layout = layout;
      drawPanel(context, view, panel, shape, layout, time, settings, colors);
    });
    context.restore();
  }

  function drawPanel(
    context,
    view,
    panel,
    shape,
    layout,
    time,
    settings,
    colors,
  ) {
    const shownTime = Math.min(Math.max(time, panel.start), panel.end);
    const gates = runningGates(panel, shownTime);
    // Gates that take no time, such as virtual rz gates, get no ring.
    const gateRings = new Map();
    for (const gate of gates) {
      if (gate.end === gate.start) continue;
      const ring =
        gate.zone === null ? colors.gate : zoneColor(view, panel, gate.zone);
      for (const ion of gate.ions) gateRings.set(ion, ring);
    }
    const activeZones = new Set(
      gates.map((gate) => gate.zone).filter((zone) => zone !== null),
    );
    const radius = layout.ionRadius;
    const segments = shape.segments.map(([start, end]) => [
      layout.point(start),
      layout.point(end),
    ]);

    context.lineCap = "round";
    panel.segments.forEach((segment, index) => {
      const color =
        segment.zones.length > 0
          ? zoneColor(view, panel, segment.zones[0])
          : colors.segment;
      const active = segment.zones.some((zone) => activeZones.has(zone));
      const [[x0, y0], [x1, y1]] = segments[index];
      context.strokeStyle = color;
      context.lineWidth = segmentWidth(segment, active, radius);
      line(context, x0, y0, x1, y1);
    });

    const zoneMarker = 0.55 * radius;
    for (const zone of panel.zones) {
      const [x, y] = layout.point(position(shape, zone.place));
      context.fillStyle = zoneColor(view, panel, zone.id);
      context.beginPath();
      context.moveTo(x, y - zoneMarker);
      context.lineTo(x + zoneMarker, y);
      context.lineTo(x, y + zoneMarker);
      context.lineTo(x - zoneMarker, y);
      context.closePath();
      context.fill();
    }
    if (settings.show_processing_zone_labels) {
      const labelSize = Math.max(9, 0.78 * layout.fontSize);
      for (const zone of panel.zones) {
        const [start, end] = segments[zone.place.segment];
        const normal = labelNormal(...shape.segments[zone.place.segment]);
        context.fillStyle = zoneColor(view, panel, zone.id);
        context.font = `500 ${labelSize}px ${FONT}`;
        parallelText(
          context,
          zone.id,
          layout.point(position(shape, zone.place)),
          start,
          end,
          [normal[0], -normal[1]],
          zoneLabelGap(radius, labelSize),
        );
      }
    }

    const junctionRadius = Math.max(1.5, 0.3 * radius);
    context.fillStyle = colors.junction;
    for (const point of shape.junctions) {
      const [x, y] = layout.point(point);
      circle(context, x, y, junctionRadius);
      context.fill();
    }
    if (settings.show_hardware_ids || settings.editing) {
      drawHardwareIds(
        context,
        panel,
        shape,
        segments,
        layout,
        colors,
        settings.show_hardware_ids,
      );
    }
    if (settings.editing)
      drawHandles(context, panel, shape, layout, colors, settings.highlight);

    const points = panel.ions.map((track) =>
      layout.point(trackPosition(track, shownTime, shape)),
    );
    panel.ionPoints = points;
    // A running gate draws a thin outer ring in the color of its processing zone.
    context.lineWidth = Math.max(1.25, 0.15 * radius);
    panel.ions.forEach((track, index) => {
      const ring = gateRings.get(track.ion);
      if (ring === undefined) return;
      context.strokeStyle = ring;
      circle(
        context,
        points[index][0],
        points[index][1],
        radius + Math.max(2.5, 0.3 * radius),
      );
      context.stroke();
    });
    const ringWidth = Math.max(1.25, 0.18 * radius);
    const styles = panel.ions.map((track) =>
      ionColors(
        view,
        track.ion,
        shownTime,
        settings.ion_colors,
        colors.ion,
        colors.ion_fill,
      ),
    );
    panel.ions.forEach((track, index) => {
      const [x, y] = points[index];
      const style = styles[index];
      context.fillStyle = style.fill;
      context.strokeStyle = style.border;
      context.lineWidth = ringWidth;
      circle(context, x, y, radius - ringWidth / 2);
      context.fill();
      context.stroke();
      const highlight = settings.highlight;
      if (
        highlight !== null &&
        highlight.panel === panel &&
        highlight.ion === track.ion
      ) {
        context.strokeStyle = colors.text;
        context.lineWidth = 1.5;
        circle(context, x, y, radius + 3);
        context.stroke();
      }
    });
    if (settings.show_ion_labels) {
      context.textAlign = "center";
      context.textBaseline = "middle";
      panel.ions.forEach((track, index) => {
        const label = `q${track.ion}`;
        const labelSize = ionLabelFontSize(radius, label);
        if (labelSize < 6) return;
        context.fillStyle = styles[index].text;
        context.font = `500 ${labelSize}px ${FONT}`;
        context.fillText(
          label,
          points[index][0],
          points[index][1] + labelSize * 0.04,
        );
      });
    }
    if (settings.ion_colors === "custom")
      drawLegend(context, view, layout, colors);
    drawHeader(context, panel, layout, time, colors);
  }

  // Distance from a processing zone to its label, clear of the gate ring.
  // Same rule as `_zone_label_gap` in `_matplotlib.py`.
  function zoneLabelGap(radius, labelSize) {
    return radius + Math.max(2.5, 0.3 * radius) + 0.6 * labelSize + 2;
  }

  // Same rules as `_segment_width` in `_matplotlib.py`.
  function segmentWidth(segment, active, radius) {
    if (segment.zones.length === 0) return Math.max(1.25, 0.3 * radius);
    return active ? Math.max(2.5, 0.6 * radius) : Math.max(2, 0.4 * radius);
  }

  function drawLegend(context, view, layout, colors) {
    const size = Math.max(9, 0.8 * layout.fontSize);
    const entries = legendEntries(view, colors.ion_fill).reverse();
    context.font = `400 ${size}px ${FONT}`;
    context.textAlign = "left";
    context.textBaseline = "middle";
    entries.forEach((entry, index) => {
      const x = layout.left + layout.padding + size / 2;
      const y =
        layout.top +
        layout.height -
        layout.padding -
        index * 1.6 * size -
        size / 2;
      context.fillStyle = entry.fill;
      context.strokeStyle = entry.border;
      context.lineWidth = 1.5;
      circle(context, x, y, size / 2 - 1);
      context.fill();
      context.stroke();
      context.fillStyle = colors.text;
      context.fillText(entry.label, x + size, y);
    });
  }

  // Write text along a segment, moved away from it along `normal` (pixels, y
  // down). The text stays readable from left to right.
  function parallelText(context, text, anchor, start, end, normal, gap) {
    let angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
    if (angle > Math.PI / 2) angle -= Math.PI;
    else if (angle <= -Math.PI / 2) angle += Math.PI;
    context.save();
    context.translate(anchor[0] + normal[0] * gap, anchor[1] + normal[1] * gap);
    context.rotate(angle);
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, 0, 0);
    context.restore();
  }

  function drawHeader(context, panel, layout, time, colors) {
    const [title, timeText, description] = headerLines(panel, time);
    const fontSize = layout.fontSize;
    const left = layout.left + layout.padding;
    const right = layout.left + layout.width - layout.padding;
    const top = layout.top + layout.padding * 0.6;
    context.textBaseline = "top";
    context.font = `400 ${0.9 * fontSize}px ${FONT}`;
    const chipWidth = context.measureText(timeText).width + fontSize;
    context.fillStyle = colors.muted_text;
    context.textAlign = "right";
    context.fillText(timeText, right, top);
    context.textAlign = "left";
    if (title) {
      context.font = `600 ${fontSize}px ${FONT}`;
      context.fillText(
        truncate(title, right - left - chipWidth - fontSize, fontSize),
        left,
        top,
      );
    }
    // Without a title, the action line moves up beside the time chip.
    const descriptionTop = title
      ? top + 1.55 * fontSize
      : top + 0.12 * fontSize;
    const descriptionWidth = title
      ? right - left
      : right - left - chipWidth - fontSize;
    context.font = `400 ${0.85 * fontSize}px ${FONT}`;
    context.fillStyle = colors.muted_text;
    context.fillText(
      truncate(description, descriptionWidth, 0.85 * fontSize),
      left,
      descriptionTop,
    );
  }

  // Segment IDs run along their segment, on the side away from the
  // processing-zone labels. Junction IDs go above and left of their junction.
  function drawHardwareIds(
    context,
    panel,
    shape,
    segments,
    layout,
    colors,
    showSegments,
  ) {
    const radius = layout.ionRadius;
    const fontSize = Math.max(8, 0.72 * layout.fontSize);
    context.fillStyle = colors.muted_text;
    context.font = `400 ${fontSize}px ${FONT}`;
    if (showSegments) {
      panel.segments.forEach((segment, index) => {
        const [start, end] = segments[index];
        const normal = labelNormal(...shape.segments[index]);
        const middle = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
        parallelText(
          context,
          segment.id,
          middle,
          start,
          end,
          [-normal[0], normal[1]],
          0.6 * radius + 0.6 * fontSize,
        );
      });
    }
    context.textAlign = "right";
    context.textBaseline = "bottom";
    shape.junctions.forEach((point, index) => {
      const [x, y] = layout.point(point);
      context.fillText(
        panel.junctionIds[index],
        x - 0.7 * radius,
        y - 0.7 * radius,
      );
    });
  }

  function drawHandles(context, panel, shape, layout, colors, highlight) {
    const size = Math.max(7, 0.8 * layout.ionRadius);
    shape.junctions.forEach((point, index) => {
      const [x, y] = layout.point(point);
      const active =
        highlight !== null &&
        highlight.kind === "junction" &&
        highlight.panel === panel &&
        highlight.junction === index;
      context.lineWidth = active ? 3 : 2;
      context.strokeStyle = colors.ion;
      context.fillStyle = colors.background;
      context.globalAlpha = active ? 1 : 0.85;
      circle(context, x, y, active ? size * 1.25 : size);
      context.fill();
      context.stroke();
      context.globalAlpha = 1;
    });
  }

  function line(context, x0, y0, x1, y1) {
    context.beginPath();
    context.moveTo(x0, y0);
    context.lineTo(x1, y1);
    context.stroke();
  }

  function circle(context, x, y, radius) {
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
  }

  // ---------------------------------------------------------------------------
  // Layout editing
  // ---------------------------------------------------------------------------

  // Return what lies under a canvas point in the last drawn frame: an ion, a
  // junction, or a processing zone, in that order.
  function pick(view, x, y) {
    for (const panel of view.panels) {
      const layout = panel.layout;
      if (layout === null) continue;
      if (x < layout.left || x > layout.left + layout.width) continue;
      const closest = (points) => {
        let best = null;
        points.forEach((point, index) => {
          const gap = Math.hypot(point[0] - x, point[1] - y);
          if (best === null || gap < best.gap) best = { index, gap };
        });
        return best;
      };
      const ion = closest(panel.ionPoints ?? []);
      if (ion !== null && ion.gap <= Math.max(8, layout.ionRadius)) {
        return { kind: "ion", panel, ion: panel.ions[ion.index].ion };
      }
      const junction = closest(
        panel.junctions.map((point) => layout.point(point)),
      );
      if (
        junction !== null &&
        junction.gap <= Math.max(12, 1.2 * layout.ionRadius)
      ) {
        return { kind: "junction", panel, junction: junction.index };
      }
      const shape = geometry(panel);
      const zone = closest(
        panel.zones.map((item) => layout.point(position(shape, item.place))),
      );
      if (zone !== null && zone.gap <= Math.max(10, layout.ionRadius)) {
        return { kind: "zone", panel, zone: panel.zones[zone.index].id };
      }
    }
    return null;
  }

  // The colors that the view can change for a picked ion or processing zone.
  function paintParts(view, handle, settings, time) {
    if (handle.kind === "ion") {
      const theme = view.colors[settings.theme];
      const style = ionColors(
        view,
        handle.ion,
        time,
        settings.ion_colors,
        theme.ion,
        theme.ion_fill,
      );
      return {
        title: `Ion q${handle.ion}`,
        parts: [
          { part: "border", label: "Ring", value: style.border },
          { part: "fill", label: "Fill", value: style.fill },
        ],
      };
    }
    if (handle.kind === "zone") {
      return {
        title: `Processing zone ${handle.zone}`,
        parts: [
          {
            part: "color",
            label: "Color",
            value: zoneColor(view, handle.panel, handle.zone),
          },
        ],
      };
    }
    return null;
  }

  // Set one color of an ion or processing zone, or clear the changes with `part === null`.
  function paint(view, handle, part, value) {
    if (handle.kind === "ion") {
      if (part === null) delete view.ionEdits[handle.ion];
      else
        view.ionEdits[handle.ion] = {
          ...view.ionEdits[handle.ion],
          [part]: value,
        };
    } else if (handle.kind === "zone") {
      if (part === null) delete view.zoneEdits[handle.zone];
      else view.zoneEdits[handle.zone] = value;
    }
  }

  // Keep the scale and position fixed while a junction moves.
  function beginDrag(view, handle) {
    const shape = geometry(handle.panel);
    handle.panel.frozen = { bounds: shape.bounds, ionRadius: shape.ionRadius };
  }

  function dragTo(view, handle, x, y) {
    const [sx, sy] = handle.panel.layout.coordinate([x, y]);
    handle.panel.junctions[handle.junction] = [
      Math.round(sx * 1e4) / 1e4,
      Math.round(sy * 1e4) / 1e4,
    ];
  }

  function endDrag(view, handle) {
    handle.panel.frozen = null;
  }

  // Undo every layout and color change made in the view.
  function resetLayout(view) {
    for (const panel of view.panels) {
      panel.junctions = panel.originalJunctions.map((point) => [...point]);
      panel.frozen = null;
    }
    view.ionEdits = {};
    view.zoneEdits = {};
  }

  // Python keyword arguments that reproduce the current layout and the colors
  // changed in the view. Comparisons get one coordinate mapping per panel.
  function settingsText(view) {
    const number = (value) => String(Math.round(value * 1000) / 1000);
    const mapping = (panel, indent) =>
      [
        "{",
        ...panel.junctionIds.map(
          (id, index) =>
            `${indent}    ${JSON.stringify(id)}: (${number(panel.junctions[index][0])}, ${number(panel.junctions[index][1])}),`,
        ),
        `${indent}}`,
      ].join("\n");
    const lines =
      view.panels.length === 1
        ? [`junction_coordinates=${mapping(view.panels[0], "")},`]
        : [
            "junction_coordinates={",
            ...view.panels.map(
              (panel) =>
                `    ${JSON.stringify(panel.title)}: ${mapping(panel, "    ")},`,
            ),
            "},",
          ];
    const ions = Object.keys(view.ionEdits)
      .map(Number)
      .sort((a, b) => a - b);
    if (ions.length > 0) {
      lines.push("ion_colors={");
      for (const ion of ions) {
        const edit = view.ionEdits[ion];
        const parts = ["border", "fill"]
          .filter((part) => edit[part] !== undefined)
          .map((part) => `${part}=${JSON.stringify(edit[part])}`);
        lines.push(`    ${ion}: IonColor(${parts.join(", ")}),`);
      }
      lines.push("},");
    }
    const zones = Object.keys(view.zoneEdits).sort();
    if (zones.length > 0) {
      lines.push("processing_zone_colors={");
      for (const zone of zones)
        lines.push(
          `    ${JSON.stringify(zone)}: ${JSON.stringify(view.zoneEdits[zone])},`,
        );
      lines.push("},");
    }
    return lines.join("\n");
  }

  return {
    beginDrag,
    currentLayer,
    dragTo,
    drawAtTime,
    endDrag,
    geometry,
    headerLines,
    ionColors,
    paint,
    paintParts,
    panelLayout,
    pick,
    prepare,
    resetLayout,
    settingsText,
    runningGates,
    stepTimes,
    timeBounds,
    trackPosition,
  };
})();

if (typeof module === "object" && module.exports) module.exports = GridDrawing;
